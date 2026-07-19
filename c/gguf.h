/* Lettore GGUF (v2/v3, little-endian) per i motori densi: NIENTE percorso di
 * caricamento parallelo — gguf_index riempie lo STESSO indice shards di st.h
 * con i nomi gia' tradotti nella nomenclatura HF, quindi tutto il downstream
 * (model_init_ex, streaming MEM_GB, micro-RSS, LoRA, REF) funziona invariato.
 *
 *   - dtypes: F32/F16/BF16 diretti; Q8_0/Q4_0/Q4_K/Q5_K/Q6_K via il dequant
 *     hook g_st_dequant_fn (st.h resta ignaro del formato);
 *   - Q4_0 + QBITS=4: gguf_repack_q4_0 e' una PURA permutazione di nibble
 *     (stessa codifica v+8, stessa granularita' 32) -> int4 grouped LOSSLESS
 *     sui kernel gia' validati;
 *   - i metadati (config, tokenizer) restano in GgufMeta per runtime.h/tok.h.
 *
 * Solo host little-endian (x86, aarch64, ppc64le): i GGUF big-endian e la
 * versione 1 vengono rifiutati rumorosamente. */
#ifndef GGUF_H
#define GGUF_H
#include "st.h"

/* ---- tipi dei valori dei metadati (spec GGUF) ---- */
enum { GG_U8=0, GG_I8=1, GG_U16=2, GG_I16=3, GG_U32=4, GG_I32=5, GG_F32=6,
       GG_BOOL=7, GG_STR=8, GG_ARR=9, GG_U64=10, GG_I64=11, GG_F64=12 };

/* valore di un KV: scalari convertiti; stringhe e array puntano nel buffer
 * dell'header (M->hdr), che resta vivo per tutta la vita del processo */
typedef struct {
    int t;                      /* GG_* */
    int64_t i; double f;
    const char *s; int64_t slen;
    int at; int64_t an;         /* array: tipo elemento e conteggio */
    const uint8_t *ap, *aend;   /* array: dati grezzi [ap, aend) nel buffer */
} gval;
typedef struct { char *key; gval v; } gkv;

typedef struct {
    gkv *kv; int nkv;
    uint8_t *hdr;               /* buffer header (posseduto) */
    int64_t data_off;
    char arch[64];
} GgufMeta;

/* ---- cursore con bound-check: un fail marca e ritorna 0 ---- */
typedef struct { const uint8_t *p; int64_t n, off; int fail; } gcur;
static uint64_t gc_u(gcur *c, int sz) {
    if (c->fail || c->off + sz > c->n) { c->fail = 1; return 0; }
    uint64_t v = 0; memcpy(&v, c->p + c->off, sz); c->off += sz;
    return v;
}
static const uint8_t *gc_bytes(gcur *c, int64_t n) {
    if (c->fail || n < 0 || c->off + n > c->n) { c->fail = 1; return NULL; }
    const uint8_t *p = c->p + c->off; c->off += n;
    return p;
}
static const char *gc_str(gcur *c, int64_t *len_out) {
    int64_t l = (int64_t)gc_u(c, 8);
    if (l < 0 || l > (1ll << 31)) { c->fail = 1; return NULL; }
    const uint8_t *p = gc_bytes(c, l);
    *len_out = l;
    return (const char *)p;
}

/* dimensione fissa di un elemento GG_* (0 = stringa/array: dimensione variabile) */
static int gg_esz(int t) {
    switch (t) {
        case GG_U8: case GG_I8: case GG_BOOL: return 1;
        case GG_U16: case GG_I16: return 2;
        case GG_U32: case GG_I32: case GG_F32: return 4;
        case GG_U64: case GG_I64: case GG_F64: return 8;
    }
    return 0;
}

/* legge un valore tipato dal cursore dentro *v (t gia' impostato) */
static void gg_read_val(gcur *c, int t, gval *v) {
    v->t = t;
    switch (t) {
        case GG_U8:  v->i = (int64_t)gc_u(c, 1); break;
        case GG_I8:  v->i = (int8_t)gc_u(c, 1); break;
        case GG_U16: v->i = (int64_t)gc_u(c, 2); break;
        case GG_I16: v->i = (int16_t)gc_u(c, 2); break;
        case GG_U32: v->i = (int64_t)gc_u(c, 4); break;
        case GG_I32: v->i = (int32_t)gc_u(c, 4); break;
        case GG_U64: v->i = (int64_t)gc_u(c, 8); break;
        case GG_I64: v->i = (int64_t)gc_u(c, 8); break;
        case GG_BOOL: v->i = gc_u(c, 1) != 0; break;
        case GG_F32: { uint32_t u = (uint32_t)gc_u(c, 4); float f; memcpy(&f, &u, 4); v->f = f; break; }
        case GG_F64: { uint64_t u = gc_u(c, 8); double d; memcpy(&d, &u, 8); v->f = d; break; }
        case GG_STR: v->s = gc_str(c, &v->slen); break;
        case GG_ARR: {
            v->at = (int)gc_u(c, 4);
            v->an = (int64_t)gc_u(c, 8);
            if (v->an < 0 || v->an > (1ll << 31)) { c->fail = 1; break; }
            v->ap = c->p + c->off;
            int esz = gg_esz(v->at);
            if (esz) { gc_bytes(c, v->an * esz); }
            else if (v->at == GG_STR) {
                for (int64_t k = 0; k < v->an && !c->fail; k++) { int64_t l; gc_str(c, &l); }
            } else c->fail = 1;             /* array di array: non nello spec d'uso */
            v->aend = c->p + c->off;
            break;
        }
        default: c->fail = 1; break;
    }
}

/* ---- lookup nei metadati ---- */
static gkv *gguf_find(GgufMeta *M, const char *key) {
    for (int i = 0; i < M->nkv; i++) if (!strcmp(M->kv[i].key, key)) return &M->kv[i];
    return NULL;
}
static int64_t gguf_int(GgufMeta *M, const char *key, int64_t def) {
    gkv *k = gguf_find(M, key);
    if (!k) return def;
    if (k->v.t == GG_F32 || k->v.t == GG_F64) return (int64_t)k->v.f;
    return k->v.i;
}
static double gguf_float(GgufMeta *M, const char *key, double def) {
    gkv *k = gguf_find(M, key);
    if (!k) return def;
    if (k->v.t == GG_F32 || k->v.t == GG_F64) return k->v.f;
    return (double)k->v.i;
}
/* stringa NON NUL-terminata: torna il puntatore e la lunghezza */
static const char *gguf_str(GgufMeta *M, const char *key, int64_t *len) {
    gkv *k = gguf_find(M, key);
    if (!k || k->v.t != GG_STR) { *len = 0; return NULL; }
    *len = k->v.slen; return k->v.s;
}
static int gguf_str_eq(GgufMeta *M, const char *key, const char *want) {
    int64_t l; const char *s = gguf_str(M, key, &l);
    return s && (int64_t)strlen(want) == l && !memcmp(s, want, l);
}

/* ---- iterazione sugli array (i tokens/merges del tokenizer sono ~150k voci) ---- */
typedef struct { gcur c; int at; int64_t left; } garr;
static int gguf_arr(GgufMeta *M, const char *key, garr *a) {
    gkv *k = gguf_find(M, key);
    if (!k || k->v.t != GG_ARR) return 0;
    a->c.p = k->v.ap; a->c.n = k->v.aend - k->v.ap; a->c.off = 0; a->c.fail = 0;
    a->at = k->v.at; a->left = k->v.an;
    return 1;
}
static const char *garr_next_str(garr *a, int64_t *len) {
    if (a->left <= 0 || a->at != GG_STR) return NULL;
    a->left--;
    return gc_str(&a->c, len);
}
static int64_t garr_next_int(garr *a) {
    if (a->left <= 0) return 0;
    a->left--;
    gval v; gg_read_val(&a->c, a->at, &v);
    return v.i;
}

/* ---- traduzione nomi llama.cpp -> HF (la tabella e' DATI: gemma = righe in
 * piu'). I nomi sconosciuti restano com'erano: l'errore utile e' il "missing
 * tensor <nome HF>" del motore, non un rifiuto qui. ---- */
static int gguf_map_name(const char *g, char *out, int cap) {
    static const struct { const char *from, *to; } fixed[] = {
        { "token_embd.weight",  "model.embed_tokens.weight" },
        { "output.weight",      "lm_head.weight" },
        { "output_norm.weight", "model.norm.weight" },
    };
    for (size_t i = 0; i < sizeof fixed/sizeof fixed[0]; i++)
        if (!strcmp(g, fixed[i].from)) { snprintf(out, cap, "%s", fixed[i].to); return 1; }
    static const struct { const char *from, *to; } blk[] = {
        { "attn_q.weight",      "self_attn.q_proj.weight" },
        { "attn_k.weight",      "self_attn.k_proj.weight" },
        { "attn_v.weight",      "self_attn.v_proj.weight" },
        { "attn_output.weight", "self_attn.o_proj.weight" },
        { "attn_q_norm.weight", "self_attn.q_norm.weight" },
        { "attn_k_norm.weight", "self_attn.k_norm.weight" },
        { "attn_norm.weight",   "input_layernorm.weight" },
        { "ffn_norm.weight",    "post_attention_layernorm.weight" },
        { "ffn_gate.weight",    "mlp.gate_proj.weight" },
        { "ffn_up.weight",      "mlp.up_proj.weight" },
        { "ffn_down.weight",    "mlp.down_proj.weight" },
    };
    int li; char sub[96];
    if (sscanf(g, "blk.%d.%95s", &li, sub) == 2) {
        for (size_t i = 0; i < sizeof blk/sizeof blk[0]; i++)
            if (!strcmp(sub, blk[i].from)) { snprintf(out, cap, "model.layers.%d.%s", li, blk[i].to); return 1; }
    }
    return 0;
}

/* ---- tipo tensore ggml -> dtype st.h ---- */
static int gguf_ggml_dtype(uint32_t t) {
    switch (t) {
        case 0:  return 2;         /* F32 */
        case 1:  return 1;         /* F16 */
        case 30: return 0;         /* BF16 */
        case 8:  return ST_Q8_0;
        case 2:  return ST_Q4_0;
        case 12: return ST_Q4_K;
        case 13: return ST_Q5_K;
        case 14: return ST_Q6_K;
    }
    return -1;
}

/* ---------- dequant dei formati a blocchi (riferimento llama.cpp) ---------- */
static void gguf_dq_q8_0(const uint8_t *b, int64_t nblk, float *out) {
    for (int64_t k = 0; k < nblk; k++, b += 34, out += 32) {
        uint16_t dh; memcpy(&dh, b, 2);
        float d = f16_to_f32(dh);
        const int8_t *q = (const int8_t *)(b + 2);
        for (int i = 0; i < 32; i++) out[i] = d * (float)q[i];
    }
}
static void gguf_dq_q4_0(const uint8_t *b, int64_t nblk, float *out) {
    for (int64_t k = 0; k < nblk; k++, b += 18, out += 32) {
        uint16_t dh; memcpy(&dh, b, 2);
        float d = f16_to_f32(dh);
        const uint8_t *q = b + 2;
        /* byte i = elemento i (nibble basso) ed elemento i+16 (alto), val = nib-8 */
        for (int i = 0; i < 16; i++) {
            out[i]      = d * (float)((int)(q[i] & 0xF) - 8);
            out[i + 16] = d * (float)((int)(q[i] >> 4) - 8);
        }
    }
}
/* scala/min a 6 bit del superblocco K (layout llama.cpp get_scale_min_k4) */
static inline void gg_k4_scale(int j, const uint8_t *s, uint8_t *sc, uint8_t *mn) {
    if (j < 4) { *sc = s[j] & 63; *mn = s[j+4] & 63; }
    else {
        *sc = (uint8_t)((s[j+4] & 0xF) | ((s[j-4] >> 6) << 4));
        *mn = (uint8_t)((s[j+4] >>  4) | ((s[j]   >> 6) << 4));
    }
}
static void gguf_dq_q4k(const uint8_t *b, int64_t nblk, float *out) {
    for (int64_t k = 0; k < nblk; k++, b += 144) {
        uint16_t dh, mh; memcpy(&dh, b, 2); memcpy(&mh, b+2, 2);
        float d = f16_to_f32(dh), dmin = f16_to_f32(mh);
        const uint8_t *scales = b + 4, *q = b + 16;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mn;
            gg_k4_scale(is + 0, scales, &sc, &mn);
            float d1 = d*sc, m1 = dmin*mn;
            gg_k4_scale(is + 1, scales, &sc, &mn);
            float d2 = d*sc, m2 = dmin*mn;
            for (int l = 0; l < 32; l++) *out++ = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *out++ = d2 * (float)(q[l] >>  4) - m2;
            q += 32; is += 2;
        }
    }
}
static void gguf_dq_q5k(const uint8_t *b, int64_t nblk, float *out) {
    for (int64_t k = 0; k < nblk; k++, b += 176) {
        uint16_t dh, mh; memcpy(&dh, b, 2); memcpy(&mh, b+2, 2);
        float d = f16_to_f32(dh), dmin = f16_to_f32(mh);
        const uint8_t *scales = b + 4, *qh = b + 16, *ql = b + 48;
        int is = 0; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mn;
            gg_k4_scale(is + 0, scales, &sc, &mn);
            float d1 = d*sc, m1 = dmin*mn;
            gg_k4_scale(is + 1, scales, &sc, &mn);
            float d2 = d*sc, m2 = dmin*mn;
            for (int l = 0; l < 32; l++) *out++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++) *out++ = d2 * (float)((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}
static void gguf_dq_q6k(const uint8_t *b, int64_t nblk, float *out) {
    for (int64_t k = 0; k < nblk; k++, b += 210) {
        const uint8_t *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        uint16_t dh; memcpy(&dh, b + 208, 2);
        float d = f16_to_f32(dh);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l/16;
                int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l]      = d * sc[is]     * (float)q1;
                out[l + 32] = d * sc[is + 2] * (float)q2;
                out[l + 64] = d * sc[is + 4] * (float)q3;
                out[l + 96] = d * sc[is + 6] * (float)q4;
            }
            out += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

/* hook per st_read_f32/st_read_slice_f32 (numel multiplo del blocco, garantito
 * dai check di allineamento in st.h) */
static void gguf_dequant(int dtype, const void *raw, int64_t numel, float *out) {
    const uint8_t *b = (const uint8_t *)raw;
    switch (dtype) {
        case ST_Q8_0: gguf_dq_q8_0(b, numel/32,  out); return;
        case ST_Q4_0: gguf_dq_q4_0(b, numel/32,  out); return;
        case ST_Q4_K: gguf_dq_q4k(b, numel/256, out); return;
        case ST_Q5_K: gguf_dq_q5k(b, numel/256, out); return;
        case ST_Q6_K: gguf_dq_q6k(b, numel/256, out); return;
    }
    fprintf(stderr, "gguf: dequant dtype %d non supportato\n", dtype); exit(1);
}

/* ---- Q4_0 -> int4 grouped (gs=32) LOSSLESS ----
 * Q4_0: byte i del blocco = elementi i (nibble basso) e i+16 (alto), valore
 * nib-8, scala f16 per blocco di 32. Il nostro layout vuole coppie sequenziali
 * (2k, 2k+1) memorizzate come v+8 — cioe' lo STESSO nibble grezzo. Pura
 * permutazione di nibble + copia scala: nessun dequant, bit-preservante. */
static void gguf_repack_q4_0(const uint8_t *blocks, uint8_t *q4, float *qs, int64_t O, int64_t I) {
    int64_t nb = I / 32, rb = I / 2;      /* I %% 32 == 0 garantito dal chiamante */
    for (int64_t o = 0; o < O; o++) {
        for (int64_t bi = 0; bi < nb; bi++) {
            const uint8_t *src = blocks + (o*nb + bi)*18;
            uint16_t dh; memcpy(&dh, src, 2);
            qs[o*nb + bi] = f16_to_f32(dh);
            const uint8_t *sq = src + 2;
            uint8_t *dst = q4 + o*rb + bi*16;
            for (int k = 0; k < 16; k++) {
                int e0 = 2*k, e1 = 2*k + 1;
                unsigned n0 = e0 < 16 ? (sq[e0] & 0xF) : (sq[e0-16] >> 4);
                unsigned n1 = e1 < 16 ? (sq[e1] & 0xF) : (sq[e1-16] >> 4);
                dst[k] = (uint8_t)(n0 | (n1 << 4));
            }
        }
    }
}

/* ---------- config sintetico ----------
 * Emette una stringa JSON con le chiavi HF a partire dai metadati <arch>.*:
 * cfg_slurp la parsa al posto di config.json, cosi' load_cfg dei motori resta
 * byte-identico (CKR compresi). vocab_size = lunghezza dell'array tokens. */
static char *gguf_synth_config(GgufMeta *M) {
    const char *a = M->arch[0] ? M->arch : "qwen3";
    char k[160];
    #define GKI(nm, def) (snprintf(k, sizeof(k), "%s." nm, a), gguf_int(M, k, def))
    #define GKF(nm, def) (snprintf(k, sizeof(k), "%s." nm, a), gguf_float(M, k, def))
    int64_t D   = GKI("embedding_length", 0);
    int64_t L   = GKI("block_count", 0);
    int64_t H   = GKI("attention.head_count", 0);
    int64_t KV  = GKI("attention.head_count_kv", H);
    int64_t IN  = GKI("feed_forward_length", 0);
    int64_t ctx = GKI("context_length", 32768);
    int64_t hd  = GKI("attention.key_length", 0);
    double th   = GKF("rope.freq_base", 1000000.0);
    double eps  = GKF("attention.layer_norm_rms_epsilon", 1e-6);
    #undef GKI
    #undef GKF
    gkv *tk = gguf_find(M, "tokenizer.ggml.tokens");
    int64_t vocab = (tk && tk->v.t == GG_ARR) ? tk->v.an : 0;
    int64_t eos = gguf_int(M, "tokenizer.ggml.eos_token_id", -1);
    char *buf = malloc(1024);
    int n = snprintf(buf, 1024,
        "{\"hidden_size\":%lld,\"num_hidden_layers\":%lld,\"num_attention_heads\":%lld,"
        "\"num_key_value_heads\":%lld,\"intermediate_size\":%lld,\"vocab_size\":%lld,"
        "\"max_position_embeddings\":%lld,\"rope_theta\":%.9g,\"rms_norm_eps\":%.9g",
        (long long)D, (long long)L, (long long)H, (long long)KV, (long long)IN,
        (long long)vocab, (long long)ctx, th, eps);
    if (hd > 0)   n += snprintf(buf+n, 1024-n, ",\"head_dim\":%lld", (long long)hd);
    if (eos >= 0) n += snprintf(buf+n, 1024-n, ",\"eos_token_id\":%lld", (long long)eos);
    snprintf(buf+n, 1024-n, "}");
    return buf;
}

/* ---------- indicizzazione di un file GGUF dentro shards ---------- */
static void gguf_index(shards *S, GgufMeta *M, const char *path) {
    memset(S, 0, sizeof(*S));
    S->cap = 4096; S->t = calloc(S->cap, sizeof(st_tensor));
    memset(M, 0, sizeof(*M));
    int fd = st_open_fd(S, path);
    struct stat sst;
    if (fstat(fd, &sst) != 0) { perror("fstat gguf"); exit(1); }
    int64_t fsz = (int64_t)sst.st_size;
    /* endianness dell'host: il formato e' little-endian e leggiamo con memcpy */
    const uint16_t one = 1;
    if (*(const uint8_t *)&one != 1) { fprintf(stderr, "%s: host big-endian non supportato\n", path); exit(1); }

    /* l'header ha lunghezza ignota (i metadati del tokenizer possono essere
     * decine di MB): si legge un prefisso crescente e si riparsa da capo
     * finche' il parse non entra tutto nel buffer */
    int64_t cap = 1 << 20;
    uint8_t *buf = NULL;
    gcur c;
    int64_t n_tensors = 0, n_kv = 0;
    for (;;) {
        if (cap > fsz) cap = fsz;
        buf = realloc(buf, cap);
        if (!buf) { fprintf(stderr, "OOM gguf header\n"); exit(1); }
        if (pread(fd, buf, cap, 0) != cap) { perror("pread gguf"); exit(1); }
        c.p = buf; c.n = cap; c.off = 0; c.fail = 0;
        uint32_t magic = (uint32_t)gc_u(&c, 4);
        uint32_t ver   = (uint32_t)gc_u(&c, 4);
        if (c.fail || magic != 0x46554747u) { fprintf(stderr, "%s: non e' un file GGUF\n", path); exit(1); }
        if (ver != 2 && ver != 3) { fprintf(stderr, "%s: versione GGUF %u non supportata (2/3)\n", path, ver); exit(1); }
        n_tensors = (int64_t)gc_u(&c, 8);
        n_kv      = (int64_t)gc_u(&c, 8);
        if (n_tensors < 0 || n_tensors > (1 << 20) || n_kv < 0 || n_kv > (1 << 20)) {
            fprintf(stderr, "%s: header GGUF assurdo (%lld tensori, %lld kv)\n",
                    path, (long long)n_tensors, (long long)n_kv); exit(1); }
        /* KV */
        M->kv = realloc(M->kv, n_kv * sizeof(gkv));
        M->nkv = 0;
        for (int64_t k = 0; k < n_kv && !c.fail; k++) {
            int64_t kl; const char *key = gc_str(&c, &kl);
            int vt = (int)gc_u(&c, 4);
            if (c.fail) break;
            gkv *e = &M->kv[M->nkv];
            e->key = NULL;
            gg_read_val(&c, vt, &e->v);
            if (!c.fail) { e->key = strndup(key, kl); M->nkv++; }
        }
        /* tensor info (secondo giro se il buffer basta) */
        if (!c.fail) break;
        for (int i = 0; i < M->nkv; i++) free(M->kv[i].key);
        M->nkv = 0;
        if (cap >= fsz) { fprintf(stderr, "%s: header GGUF troncato/malformato\n", path); exit(1); }
        cap *= 8;
    }
    /* i tensor info seguono i KV: possono ancora sforare il buffer -> stesso
     * schema a prefisso crescente, ripartendo dal parse completo */
    typedef struct { char *hf; int dtype; int64_t numel, nbytes, off; } ginfo;
    ginfo *ti = NULL;
    for (;;) {
        int64_t save = c.off;
        ti = realloc(ti, (n_tensors > 0 ? n_tensors : 1) * sizeof(ginfo));
        int ok = 1;
        int64_t nti = 0;
        for (int64_t i = 0; i < n_tensors; i++) {
            int64_t nl; const char *nm = gc_str(&c, &nl);
            uint32_t nd = (uint32_t)gc_u(&c, 4);
            if (c.fail || nd > 4) { ok = 0; break; }
            int64_t numel = 1;
            for (uint32_t d = 0; d < nd; d++) numel *= (int64_t)gc_u(&c, 8);
            uint32_t gt = (uint32_t)gc_u(&c, 4);
            int64_t off = (int64_t)gc_u(&c, 8);
            if (c.fail || numel <= 0) { ok = 0; break; }
            int dt = gguf_ggml_dtype(gt);
            if (dt < 0) {
                fprintf(stderr, "%s: tipo ggml %u non supportato (tensore %.*s)\n",
                        path, gt, (int)nl, nm); exit(1); }
            int64_t nbytes;
            int be = 1, bb = 1;
            if (dt >= ST_DTYPE_QBLOCK) {
                st_qblock(dt, &be, &bb);   /* dt filtrato da gguf_ggml_dtype: sempre noto */
                if (numel % be) { fprintf(stderr, "%s: numel %lld non multiplo del blocco %d\n",
                                          path, (long long)numel, be); exit(1); }
                nbytes = numel/be*bb;
            } else nbytes = numel * (dt == 2 ? 4 : 2);
            char raw[256], hf[256];
            snprintf(raw, sizeof(raw), "%.*s", (int)(nl < 255 ? nl : 255), nm);
            if (!gguf_map_name(raw, hf, sizeof(hf))) snprintf(hf, sizeof(hf), "%s", raw);
            ti[nti].hf = strdup(hf); ti[nti].dtype = dt;
            ti[nti].numel = numel; ti[nti].nbytes = nbytes; ti[nti].off = off;
            nti++;
        }
        if (ok && !c.fail) break;
        for (int64_t i = 0; i < nti; i++) free(ti[i].hf);
        if (cap >= fsz) { fprintf(stderr, "%s: tensor info GGUF troncati\n", path); exit(1); }
        cap *= 8; if (cap > fsz) cap = fsz;
        buf = realloc(buf, cap);
        if (pread(fd, buf, cap, 0) != cap) { perror("pread gguf"); exit(1); }
        c.p = buf; c.n = cap; c.fail = 0; c.off = save;
        /* NB: save resta valido: i KV sono gia' parsati e il prefisso riletto
         * e' identico byte a byte */
    }
    int64_t align = gguf_int(M, "general.alignment", 32);
    if (align <= 0 || (align & (align - 1))) { fprintf(stderr, "%s: alignment %lld non valido\n", path, (long long)align); exit(1); }
    M->data_off = (c.off + align - 1) & ~(align - 1);
    M->hdr = buf;
    int64_t al; const char *as = gguf_str(M, "general.architecture", &al);
    snprintf(M->arch, sizeof(M->arch), "%.*s", (int)(as ? (al < 63 ? al : 63) : 0), as ? as : "");
    /* riempi l'indice shards */
    for (int64_t i = 0; i < n_tensors; i++) {
        if (M->data_off + ti[i].off + ti[i].nbytes > fsz) {
            fprintf(stderr, "%s: tensore %s fuori dal file\n", path, ti[i].hf); exit(1); }
        if (S->n == S->cap) { S->cap *= 2; S->t = realloc(S->t, S->cap*sizeof(st_tensor)); }
        st_tensor *t = &S->t[S->n++];
        t->name = ti[i].hf; t->fd = fd;
        t->off = M->data_off + ti[i].off;
        t->nbytes = ti[i].nbytes; t->dtype = ti[i].dtype; t->numel = ti[i].numel;
    }
    free(ti);
    st_hash_build(S);
    g_st_dequant_fn = gguf_dequant;
    fprintf(stderr, "[gguf] %s: arch=%s, %d tensori, data @ %lld\n",
            path, M->arch[0] ? M->arch : "?", S->n, (long long)M->data_off);
}

#endif /* GGUF_H */
