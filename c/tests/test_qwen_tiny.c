/* End-to-end strutturale del motore qwen: modello 2-layer con pesi casuali
 * scritto al volo (config.json + model.safetensors), generate greedy 8 token:
 * niente NaN, output deterministico, embedding legati, QBITS=8 non esplode. */
#define QWEN_TEST
#include "../qwen.c"
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#define MKDIR(p) mkdir(p, 0755)
#endif

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static uint64_t rng = 7;
static float frnd(void) {
    rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17;
    return (float)((rng>>11)*(1.0/9007199254740992.0)) - 0.5f;
}

/* ---- scrittura di un safetensors F32 minimale ---- */
typedef struct { char name[128]; char shape[64]; int64_t numel; float *data; } TT;
static TT g_t[64]; static int g_nt = 0;

static float *add_t(const char *name, const char *shape, int64_t numel, float scale, int ones) {
    TT *t = &g_t[g_nt++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    snprintf(t->shape, sizeof(t->shape), "%s", shape);
    t->numel = numel;
    t->data = falloc(numel);
    for (int64_t i = 0; i < numel; i++) t->data[i] = ones ? 1.f : frnd()*scale;
    return t->data;
}

static void write_st_file(const char *path) {
    char *hdr = malloc(1<<16); int hl = 0;
    hdr[hl++] = '{';
    int64_t off = 0;
    for (int i = 0; i < g_nt; i++) {
        hl += snprintf(hdr+hl, (1<<16)-hl,
            "%s\"%s\":{\"dtype\":\"F32\",\"shape\":%s,\"data_offsets\":[%lld,%lld]}",
            i ? "," : "", g_t[i].name, g_t[i].shape,
            (long long)off, (long long)(off + g_t[i].numel*4));
        off += g_t[i].numel*4;
    }
    hdr[hl++] = '}';
    FILE *f = fopen(path, "wb"); CHECK(f);
    uint64_t hlen = (uint64_t)hl;
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, hl, f);
    for (int i = 0; i < g_nt; i++) fwrite(g_t[i].data, 4, g_t[i].numel, f);
    fclose(f); free(hdr);
}

static void write_cfg(const char *path) {
    FILE *f = fopen(path, "wb"); CHECK(f);
    fputs("{\"hidden_size\":16,\"num_hidden_layers\":2,\"num_attention_heads\":4,"
          "\"num_key_value_heads\":2,\"head_dim\":8,\"intermediate_size\":32,"
          "\"vocab_size\":32,\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
          "\"tie_word_embeddings\":true,\"eos_token_id\":0,"
          "\"max_position_embeddings\":64}", f);
    fclose(f);
}

static void build_model_dir(const char *dir) {
    MKDIR(dir);
    char p[600];
    snprintf(p, sizeof(p), "%s/config.json", dir); write_cfg(p);
    g_nt = 0; rng = 7;
    add_t("model.embed_tokens.weight", "[32,16]", 32*16, 0.5f, 0);
    add_t("model.norm.weight", "[16]", 16, 0, 1);
    char nm[128];
    for (int i = 0; i < 2; i++) {
        #define AT(suffix, shape, numel, sc, ones) \
            do { snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); add_t(nm,shape,numel,sc,ones); } while(0)
        AT("input_layernorm.weight", "[16]", 16, 0, 1);
        AT("post_attention_layernorm.weight", "[16]", 16, 0, 1);
        AT("self_attn.q_norm.weight", "[8]", 8, 0, 1);
        AT("self_attn.k_norm.weight", "[8]", 8, 0, 1);
        AT("self_attn.q_proj.weight", "[32,16]", 32*16, 0.3f, 0);
        AT("self_attn.k_proj.weight", "[16,16]", 16*16, 0.3f, 0);
        AT("self_attn.v_proj.weight", "[16,16]", 16*16, 0.3f, 0);
        AT("self_attn.o_proj.weight", "[16,32]", 16*32, 0.3f, 0);
        AT("mlp.gate_proj.weight", "[32,16]", 32*16, 0.3f, 0);
        AT("mlp.up_proj.weight",   "[32,16]", 32*16, 0.3f, 0);
        AT("mlp.down_proj.weight", "[16,32]", 16*32, 0.3f, 0);
        #undef AT
    }
    snprintf(p, sizeof(p), "%s/model.safetensors", dir); write_st_file(p);
}

/* greedy 8 token dal prompt {1,2,3}; out deve avere spazio per 11 */
static void run8(const char *dir, int qbits, int *out) {
    Model m;
    model_init(&m, dir, qbits);
    CHECK(m.lm_tied);                       /* tie_word_embeddings=true */
    CHECK(m.c.head_dim == 8);               /* esplicito, non hidden/heads=4 */
    kv_alloc(&m, 16);
    int prompt[3] = {1,2,3};
    memcpy(out, prompt, sizeof(prompt));
    float *logit = step(&m, prompt, 3, 0);
    int len = 3;
    for (int s = 0; s < 8; s++) {
        for (int i = 0; i < m.c.vocab; i++) CHECK(isfinite(logit[i]));
        int best = argmax_v(logit, m.c.vocab);
        free(logit);
        out[len++] = best;
        if (s == 7) break;
        logit = step(&m, &out[len-1], 1, len-1);
    }
}

int main(void) {
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    char dir[512]; snprintf(dir, sizeof(dir), "%s/qwen_tiny_model", tmp);
    build_model_dir(dir);

    int a[16], b[16], q[16];
    run8(dir, 0, a);
    run8(dir, 0, b);                        /* due run indipendenti: stessi id */
    for (int i = 0; i < 11; i++) CHECK(a[i]==b[i]);

    run8(dir, 8, q);                        /* QBITS=8: gira e resta finito */
    fprintf(stderr, "f32 ids:"); for (int i=3;i<11;i++) fprintf(stderr," %d",a[i]);
    fprintf(stderr, "\nint8 ids:"); for (int i=3;i<11;i++) fprintf(stderr," %d",q[i]);
    fprintf(stderr, "\n");

    printf("test_qwen_tiny: ok\n");
    return 0;
}
