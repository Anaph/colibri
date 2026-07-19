/* Writer GGUF v3 minimale, SOLO per le fixture dei test (gemello di tiny_st.h).
 * Non e' un writer generale: liste statiche, niente error-handling raffinato. */
#ifndef TINY_GGUF_H
#define TINY_GGUF_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TG_MAX_KV 64
#define TG_MAX_T  64
#define TG_ALIGN  32

typedef struct {
    char key[96];
    int t;                      /* GG_* di gguf.h */
    int64_t i; double f;
    char s[128];
    int at; int64_t an;
    uint8_t *adata; size_t abytes;   /* array gia' serializzato (posseduto) */
} tg_kv_rec;
typedef struct {
    char name[160];
    uint32_t gt; int nd; int64_t dims[4];
    uint8_t *data; size_t nbytes;    /* copia posseduta */
} tg_t_rec;

static tg_kv_rec tg_kvs[TG_MAX_KV]; static int tg_nkv;
static tg_t_rec  tg_ts[TG_MAX_T];   static int tg_nt;

static void tg_reset(void) {
    for (int i = 0; i < tg_nkv; i++) free(tg_kvs[i].adata);
    for (int i = 0; i < tg_nt; i++) free(tg_ts[i].data);
    tg_nkv = 0; tg_nt = 0;
}
static tg_kv_rec *tg_kv_new(const char *k, int t) {
    tg_kv_rec *e = &tg_kvs[tg_nkv++];
    memset(e, 0, sizeof(*e));
    snprintf(e->key, sizeof(e->key), "%s", k);
    e->t = t;
    return e;
}
static void tg_kv_u32(const char *k, uint32_t v) { tg_kv_new(k, 4)->i = v; }
static void tg_kv_f32(const char *k, float v)    { tg_kv_new(k, 6)->f = v; }
static void tg_kv_str(const char *k, const char *v) {
    tg_kv_rec *e = tg_kv_new(k, 8);
    snprintf(e->s, sizeof(e->s), "%s", v);
}
static void tg_kv_arr_str(const char *k, const char **items, int n) {
    tg_kv_rec *e = tg_kv_new(k, 9);
    e->at = 8; e->an = n;
    size_t tot = 0;
    for (int i = 0; i < n; i++) tot += 8 + strlen(items[i]);
    e->adata = malloc(tot ? tot : 1); e->abytes = tot;
    uint8_t *p = e->adata;
    for (int i = 0; i < n; i++) {
        uint64_t l = strlen(items[i]);
        memcpy(p, &l, 8); p += 8;
        memcpy(p, items[i], l); p += l;
    }
}
static void tg_kv_arr_i32(const char *k, const int32_t *items, int n) {
    tg_kv_rec *e = tg_kv_new(k, 9);
    e->at = 5; e->an = n;
    e->abytes = (size_t)n*4;
    e->adata = malloc(e->abytes ? e->abytes : 1);
    memcpy(e->adata, items, e->abytes);
}
static void tg_tensor(const char *name, uint32_t ggml_type, int nd, const int64_t *dims,
                      const void *data, size_t nbytes) {
    tg_t_rec *e = &tg_ts[tg_nt++];
    memset(e, 0, sizeof(*e));
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->gt = ggml_type; e->nd = nd;
    for (int i = 0; i < nd; i++) e->dims[i] = dims[i];
    e->data = malloc(nbytes ? nbytes : 1); e->nbytes = nbytes;
    memcpy(e->data, data, nbytes);
}
/* comodo per i tensori f32: dims[0]=I (ne0, contigua), dims[1]=O */
static void tg_tensor_f32(const char *name, int64_t O, int64_t I, const float *data) {
    int64_t dims[2] = { I, O };
    tg_tensor(name, 0, 2, dims, data, (size_t)(O*I)*4);
}

static void tg_w(FILE *f, const void *p, size_t n) { fwrite(p, 1, n, f); }
static void tg_wu32(FILE *f, uint32_t v) { tg_w(f, &v, 4); }
static void tg_wu64(FILE *f, uint64_t v) { tg_w(f, &v, 8); }
static void tg_wstr(FILE *f, const char *s) { tg_wu64(f, strlen(s)); tg_w(f, s, strlen(s)); }

static void tg_write(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    tg_wu32(f, 0x46554747u);         /* "GGUF" */
    tg_wu32(f, 3);
    tg_wu64(f, (uint64_t)tg_nt);
    tg_wu64(f, (uint64_t)tg_nkv);
    for (int i = 0; i < tg_nkv; i++) {
        tg_kv_rec *e = &tg_kvs[i];
        tg_wstr(f, e->key);
        tg_wu32(f, (uint32_t)e->t);
        switch (e->t) {
            case 4: tg_wu32(f, (uint32_t)e->i); break;
            case 6: { float v = (float)e->f; tg_w(f, &v, 4); break; }
            case 8: tg_wstr(f, e->s); break;
            case 9: tg_wu32(f, (uint32_t)e->at); tg_wu64(f, (uint64_t)e->an);
                    tg_w(f, e->adata, e->abytes); break;
            default: fprintf(stderr, "tiny_gguf: tipo kv %d non implementato\n", e->t); exit(1);
        }
    }
    /* offset dei dati: allineati a TG_ALIGN dentro la sezione dati */
    uint64_t off = 0, offs[TG_MAX_T];
    for (int i = 0; i < tg_nt; i++) {
        off = (off + TG_ALIGN - 1) & ~(uint64_t)(TG_ALIGN - 1);
        offs[i] = off;
        off += tg_ts[i].nbytes;
    }
    for (int i = 0; i < tg_nt; i++) {
        tg_t_rec *e = &tg_ts[i];
        tg_wstr(f, e->name);
        tg_wu32(f, (uint32_t)e->nd);
        for (int d = 0; d < e->nd; d++) tg_wu64(f, (uint64_t)e->dims[d]);
        tg_wu32(f, e->gt);
        tg_wu64(f, offs[i]);
    }
    /* pad dell'header fino all'allineamento, poi i blob (ognuno gia' allineato) */
    long hend = ftell(f);
    long dstart = (hend + TG_ALIGN - 1) & ~(long)(TG_ALIGN - 1);
    for (long p = hend; p < dstart; p++) fputc(0, f);
    uint64_t cur = 0;
    for (int i = 0; i < tg_nt; i++) {
        while (cur < offs[i]) { fputc(0, f); cur++; }
        tg_w(f, tg_ts[i].data, tg_ts[i].nbytes);
        cur += tg_ts[i].nbytes;
    }
    fclose(f);
}

#endif /* TINY_GGUF_H */
