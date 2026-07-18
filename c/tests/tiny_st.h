/* Scrittura di checkpoint sintetici per i test: config.json lo scrive il
 * chiamante, qui si accumulano tensori F32 casuali (LCG deterministico) e si
 * emette un model.safetensors minimale. Usato dai test qwen/gemma/memknob. */
#ifndef TINY_ST_H
#define TINY_ST_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define TST_MKDIR(p) _mkdir(p)
#else
#define TST_MKDIR(p) mkdir(p, 0755)
#endif

typedef struct { char name[128]; char shape[64]; int64_t numel; float *data; } TstTensor;
static TstTensor tst_t[96]; static int tst_nt = 0;
static uint64_t tst_rng = 7;

static float tst_frnd(void) {
    tst_rng ^= tst_rng<<13; tst_rng ^= tst_rng>>7; tst_rng ^= tst_rng<<17;
    return (float)((tst_rng>>11)*(1.0/9007199254740992.0)) - 0.5f;
}

static void tst_reset(uint64_t seed) { tst_nt = 0; tst_rng = seed; }

static float *tst_add(const char *name, const char *shape, int64_t numel, float scale, int ones) {
    if (tst_nt >= 96) { fprintf(stderr, "tiny_st: troppi tensori\n"); exit(1); }
    TstTensor *t = &tst_t[tst_nt++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    snprintf(t->shape, sizeof(t->shape), "%s", shape);
    t->numel = numel;
    t->data = (float*)malloc(numel*sizeof(float));
    if (!t->data) { fprintf(stderr, "tiny_st: OOM\n"); exit(1); }
    for (int64_t i = 0; i < numel; i++) t->data[i] = ones ? 1.f : tst_frnd()*scale;
    return t->data;
}

static void tst_write(const char *dir) {
    char path[640];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    char *hdr = (char*)malloc(1<<16); int hl = 0;
    hdr[hl++] = '{';
    int64_t off = 0;
    for (int i = 0; i < tst_nt; i++) {
        hl += snprintf(hdr+hl, (1<<16)-hl,
            "%s\"%s\":{\"dtype\":\"F32\",\"shape\":%s,\"data_offsets\":[%lld,%lld]}",
            i ? "," : "", tst_t[i].name, tst_t[i].shape,
            (long long)off, (long long)(off + tst_t[i].numel*4));
        off += tst_t[i].numel*4;
    }
    hdr[hl++] = '}';
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); exit(1); }
    uint64_t hlen = (uint64_t)hl;
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, hl, f);
    for (int i = 0; i < tst_nt; i++) { fwrite(tst_t[i].data, 4, tst_t[i].numel, f); free(tst_t[i].data); }
    fclose(f); free(hdr);
    tst_nt = 0;
}

static void tst_write_text(const char *dir, const char *fname, const char *body) {
    char path[640];
    snprintf(path, sizeof(path), "%s/%s", dir, fname);
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); exit(1); }
    fputs(body, f); fclose(f);
}

static const char *tst_dir(const char *name) {
    static char dir[512];
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    snprintf(dir, sizeof(dir), "%s/%s", tmp, name);
    TST_MKDIR(dir);
    return dir;
}

#endif /* TINY_ST_H */
