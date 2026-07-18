/* Logica dei test del motore gemma (C puro; glue gtest in gemma_gtest.cc).
 * Ritorni: 0=ok, 1=fail. Include il motore intero. */
#define GEMMA_TEST
#include "../gemma.c"
#include "tiny_st.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static uint64_t gm_rng = 21;
static float gm_frnd(void) {
    gm_rng ^= gm_rng<<13; gm_rng ^= gm_rng>>7; gm_rng ^= gm_rng<<17;
    return (float)((gm_rng>>11)*(1.0/9007199254740992.0)) - 0.5f;
}

/* ---- p-RoPE: coppie < rot_angles ruotate con inv_freq su hd INTERO, oltre identita' ---- */
int gm_prope(void) {
    int hd = 8, rot = 2; float theta = 1000000.f;
    float x[8], x0[8];
    gm_rng = 21;
    for (int i = 0; i < hd; i++) x[i] = x0[i] = gm_frnd();
    gemma_rope_head(x, 13, theta, hd, rot);
    int h = hd/2;
    for (int j = 0; j < h; j++) {
        if (j < rot) {
            double inv = pow((double)theta, -2.0*j/hd);     /* denominatore = hd, non rot */
            double ang = 13.0*inv, cs = cos(ang), sn = sin(ang);
            double a = x0[j], b = x0[j+h];
            CHECK(fabs(x[j]   - (a*cs - b*sn)) < 1e-5);
            CHECK(fabs(x[j+h] - (b*cs + a*sn)) < 1e-5);
        } else {                                            /* frequenza nulla: identita' */
            CHECK(x[j] == x0[j] && x[j+h] == x0[j+h]);
        }
    }
    /* rot = hd/2 -> RoPE pieno, coincide con la rotazione classica */
    memcpy(x, x0, sizeof(x));
    gemma_rope_head(x, 5, theta, hd, hd/2);
    for (int j = 0; j < h; j++) {
        double inv = pow((double)theta, -2.0*j/hd);
        double ang = 5.0*inv, cs = cos(ang), sn = sin(ang);
        CHECK(fabs(x[j] - (x0[j]*cs - x0[j+h]*sn)) < 1e-5);
    }
    return 0;
}

/* ---- GeGLU vs double ---- */
int gm_geglu(void) {
    for (int i = 0; i < 64; i++) {
        double xv = (i-32)*0.31;
        double ref = 0.5*xv*(1.0 + tanh(0.7978845608028654*(xv + 0.044715*xv*xv*xv)));
        CHECK(fabs((double)gelu_tanh((float)xv) - ref) < 1e-5);
    }
    return 0;
}

/* ---- RMSNorm: entrambe le convenzioni ---- */
int gm_rmsnorm(void) {
    GCfg c; memset(&c,0,sizeof c); c.eps = 1e-6f;
    float x[4] = {1, -2, 3, -4}, w[4] = {0.1f, 0.2f, -0.1f, 0.0f}, out[4];
    double ms = (1.0+4+9+16)/4.0;
    double r = 1.0/sqrt(ms + 1e-6);
    c.zc_norm = 1;
    gnorm_row(&c, out, x, w, 4);
    for (int i = 0; i < 4; i++) CHECK(fabs(out[i] - x[i]*r*(1.0+w[i])) < 1e-5);
    c.zc_norm = 0;
    gnorm_row(&c, out, x, w, 4);
    for (int i = 0; i < 4; i++) CHECK(fabs(out[i] - x[i]*r*w[i]) < 1e-5);
    return 0;
}

/* ---- sliding mask: attention() vs riferimento brute-force double con finestra ---- */
int gm_sliding(void) {
    int D = 8, H = 2, KV = 2, hd = 4, S = 5, W = 2;
    GModel m; memset(&m,0,sizeof m);
    m.c.hidden=D; m.c.n_heads=H; m.c.n_kv_heads=KV; m.c.n_gkv=KV;
    m.c.head_dim=hd; m.c.ghd=hd; m.c.rot_angles=hd/2;
    m.c.theta_g=1e6f; m.c.theta_l=1e4f; m.c.eps=1e-6f; m.c.n_layers=1; m.c.window=W;
    m.c.zc_norm=0;                                  /* pesi=1 -> norma pura, piu' semplice per il ref */
    static int lt[1] = {0};                          /* sliding */
    static int ks[1] = {0};
    m.c.ltype = lt; m.c.kv_src = ks;
    GLayer l; memset(&l,0,sizeof l);
    l.type = 0;
    l.q.f=falloc((int64_t)H*hd*D);  l.q.O=H*hd;  l.q.I=D;
    l.k.f=falloc((int64_t)KV*hd*D); l.k.O=KV*hd; l.k.I=D;
    l.v.f=falloc((int64_t)KV*hd*D); l.v.O=KV*hd; l.v.I=D;
    l.o.f=falloc((int64_t)D*H*hd);  l.o.O=D;     l.o.I=H*hd;
    l.qn=falloc(hd); l.kn=falloc(hd);
    gm_rng = 33;
    for (int64_t i=0;i<(int64_t)H*hd*D;i++) l.q.f[i]=gm_frnd();
    for (int64_t i=0;i<(int64_t)KV*hd*D;i++){ l.k.f[i]=gm_frnd(); l.v.f[i]=gm_frnd(); }
    for (int64_t i=0;i<(int64_t)D*H*hd;i++) l.o.f[i]=gm_frnd();
    for (int i=0;i<hd;i++){ l.qn[i]=1; l.kn[i]=1; }
    kv_alloc(&m, 8);
    float *x = falloc((int64_t)S*D);
    for (int64_t i=0;i<(int64_t)S*D;i++) x[i]=gm_frnd();
    float *out = falloc((int64_t)S*D);
    attention(&m, &l, 0, x, S, 0, out);
    /* riferimento double: proiezioni + norma per testa + rope pieno + softmax
     * mascherata alla finestra [qpos-W+1, qpos] */
    double *q = malloc(sizeof(double)*S*H*hd), *k = malloc(sizeof(double)*S*KV*hd), *v = malloc(sizeof(double)*S*KV*hd);
    for (int s = 0; s < S; s++) {
        for (int o = 0; o < H*hd; o++) { double a=0; for (int i=0;i<D;i++) a += (double)l.q.f[(int64_t)o*D+i]*x[s*D+i]; q[(s*H*hd)+o]=a; }
        for (int o = 0; o < KV*hd; o++) { double a=0; for (int i=0;i<D;i++) a += (double)l.k.f[(int64_t)o*D+i]*x[s*D+i]; k[(s*KV*hd)+o]=a; }
        for (int o = 0; o < KV*hd; o++) { double a=0; for (int i=0;i<D;i++) a += (double)l.v.f[(int64_t)o*D+i]*x[s*D+i]; v[(s*KV*hd)+o]=a; }
        for (int hh = 0; hh < H; hh++) {
            double *qh = q + s*H*hd + hh*hd;
            double ms=0; for (int j=0;j<hd;j++) ms += qh[j]*qh[j];
            double r = 1.0/sqrt(ms/hd + 1e-6);
            for (int j=0;j<hd;j++) qh[j] *= r;
            for (int j = 0; j < hd/2; j++) {
                double inv = pow(1e4, -2.0*j/hd), ang = s*inv;
                double a = qh[j], b = qh[j+hd/2];
                qh[j] = a*cos(ang)-b*sin(ang); qh[j+hd/2] = b*cos(ang)+a*sin(ang);
            }
        }
        for (int hh = 0; hh < KV; hh++) {
            double *kh = k + s*KV*hd + hh*hd;
            double ms=0; for (int j=0;j<hd;j++) ms += kh[j]*kh[j];
            double r = 1.0/sqrt(ms/hd + 1e-6);
            for (int j=0;j<hd;j++) kh[j] *= r;
            for (int j = 0; j < hd/2; j++) {
                double inv = pow(1e4, -2.0*j/hd), ang = s*inv;
                double a = kh[j], b = kh[j+hd/2];
                kh[j] = a*cos(ang)-b*sin(ang); kh[j+hd/2] = b*cos(ang)+a*sin(ang);
            }
        }
    }
    double scale = 1.0/sqrt((double)hd);
    for (int s = 0; s < S; s++) {
        double ctx[64];
        for (int hh = 0; hh < H; hh++) {
            int kvh = hh / (H/KV);
            int t0 = s-W+1 > 0 ? s-W+1 : 0;
            double sc[16], mx=-1e30, sum=0;
            for (int t = t0; t <= s; t++) {
                double a=0; for (int j=0;j<hd;j++) a += q[s*H*hd+hh*hd+j]*k[t*KV*hd+kvh*hd+j];
                sc[t-t0]=a*scale; if (sc[t-t0]>mx) mx=sc[t-t0];
            }
            for (int t = t0; t <= s; t++){ sc[t-t0]=exp(sc[t-t0]-mx); sum+=sc[t-t0]; }
            for (int j = 0; j < hd; j++) {
                double a=0;
                for (int t = t0; t <= s; t++) a += sc[t-t0]/sum * v[t*KV*hd+kvh*hd+j];
                ctx[hh*hd+j]=a;
            }
        }
        for (int d = 0; d < D; d++) {
            double a=0; for (int j=0;j<H*hd;j++) a += (double)l.o.f[(int64_t)d*H*hd+j]*ctx[j];
            if (fabs((double)out[s*D+d] - a) > 1e-4) {
                fprintf(stderr, "sliding s=%d d=%d: got %.6f expected %.6f\n", s, d, out[s*D+d], a);
                return 1;
            }
        }
    }
    free(q); free(k); free(v); free(x); free(out);
    return 0;
}

/* ---- tiny end-to-end ibrido: 4 layer [s,s,full,s], finestra 4, prf 0.25 ---- */
static void gm_write_tiny(const char *dir, int kv_shared, int k_eq_v) {
    char cfg[2048];
    snprintf(cfg, sizeof(cfg),
        "{\"text_config\":{\"hidden_size\":16,\"num_hidden_layers\":4,\"num_attention_heads\":4,"
        "\"num_key_value_heads\":2,\"head_dim\":8,\"global_head_dim\":8,\"intermediate_size\":32,"
        "\"vocab_size\":32,\"rope_theta\":1000000.0,\"rope_local_base_freq\":10000.0,"
        "\"rms_norm_eps\":1e-06,\"sliding_window\":4,\"partial_rotary_factor\":0.25,"
        "\"tie_word_embeddings\":true,\"eos_token_id\":0,\"max_position_embeddings\":64,"
        "\"layer_types\":[\"sliding_attention\",\"sliding_attention\",\"full_attention\",\"sliding_attention\"],"
        "\"num_kv_shared_layers\":%d,\"attention_k_eq_v\":%s}}",
        kv_shared, k_eq_v ? "true" : "false");
    tst_write_text(dir, "config.json", cfg);
    tst_reset(17);
    tst_add("model.embed_tokens.weight", "[32,16]", 32*16, 0.5f, 0);
    tst_add("model.norm.weight", "[16]", 16, 0.05f, 0);
    char nm[128];
    for (int i = 0; i < 4; i++) {
        int shared = kv_shared && i >= 4 - kv_shared;
        #define AT(suffix, shape, numel, sc, ones) \
            do { snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); tst_add(nm,shape,numel,sc,ones); } while(0)
        AT("input_layernorm.weight", "[16]", 16, 0.05f, 0);
        AT("post_attention_layernorm.weight", "[16]", 16, 0.05f, 0);
        AT("pre_feedforward_layernorm.weight", "[16]", 16, 0.05f, 0);
        AT("post_feedforward_layernorm.weight", "[16]", 16, 0.05f, 0);
        AT("self_attn.q_norm.weight", "[8]", 8, 0.05f, 0);
        AT("self_attn.k_norm.weight", "[8]", 8, 0.05f, 0);
        AT("self_attn.q_proj.weight", "[32,16]", 32*16, 0.3f, 0);
        AT("self_attn.o_proj.weight", "[16,32]", 16*32, 0.3f, 0);
        if (!shared) {
            AT("self_attn.k_proj.weight", "[16,16]", 16*16, 0.3f, 0);
            if (!k_eq_v) AT("self_attn.v_proj.weight", "[16,16]", 16*16, 0.3f, 0);
        }
        AT("mlp.gate_proj.weight", "[32,16]", 32*16, 0.3f, 0);
        AT("mlp.up_proj.weight",   "[32,16]", 32*16, 0.3f, 0);
        AT("mlp.down_proj.weight", "[16,32]", 16*32, 0.3f, 0);
        #undef AT
    }
    tst_write(dir);
}

static int gm_run8(const char *dir, int qbits, int *out, int expect_shared, int expect_keqv) {
    GModel m;
    model_init(&m, dir, qbits);
    CHECK(m.lm_tied);
    CHECK(m.c.ltype[2] == 1 && m.c.ltype[0] == 0 && m.c.ltype[3] == 0);
    CHECK(m.c.rot_angles == 1);                    /* int(0.25*8)/2 */
    if (expect_shared) {
        CHECK(m.c.kv_src[3] == 1);                 /* ultimo sliding condivide col layer 1 (sliding) */
        CHECK(m.L[3].shared_kv);
    }
    if (expect_keqv) CHECK(m.c.k_eq_v && !m.L[0].v.f && !m.L[0].v.q);
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
    return 0;
}

int gm_tiny(void) {
    const char *dir = tst_dir("gemma_tiny_model");
    gm_write_tiny(dir, 0, 0);
    int a[16], b[16];
    CHECK(gm_run8(dir, 0, a, 0, 0) == 0);
    CHECK(gm_run8(dir, 0, b, 0, 0) == 0);
    for (int i = 0; i < 11; i++) CHECK(a[i]==b[i]);
    CHECK(gm_run8(dir, 8, b, 0, 0) == 0);          /* QBITS=8 */
    return 0;
}

int gm_tiny_shared(void) {
    const char *dir = tst_dir("gemma_tiny_shared");
    gm_write_tiny(dir, 1, 0);                      /* ultimo layer kv-shared */
    int a[16], b[16];
    CHECK(gm_run8(dir, 0, a, 1, 0) == 0);
    CHECK(gm_run8(dir, 0, b, 1, 0) == 0);
    for (int i = 0; i < 11; i++) CHECK(a[i]==b[i]);
    return 0;
}

int gm_tiny_keqv(void) {
    const char *dir = tst_dir("gemma_tiny_keqv");
    gm_write_tiny(dir, 0, 1);                      /* v_proj assente, V=K */
    int a[16];
    return gm_run8(dir, 0, a, 0, 1);
}
