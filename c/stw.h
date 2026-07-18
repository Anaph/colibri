/* Scrittore minimale di safetensors F32 con buffering in memoria: si accumulano
 * tensori con stw_add (i dati vengono COPIATI), poi stw_write emette il file
 * (8 byte di hlen little-endian + header JSON + blob contigui) e azzera lo
 * stato. Usato dal trainer LoRA per salvare gli adattatori e dai test. */
#ifndef STW_H
#define STW_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct { char name[128]; char shape[80]; int64_t numel; float *data; } StwT;
static StwT stw_t[256]; static int stw_n;

/* accoda un tensore F32 [shape[0],...,shape[ndim-1]]; i dati vengono copiati */
static void stw_add(const char *name, int ndim, const int64_t *shape, const float *data) {
    if (stw_n >= 256) { fprintf(stderr, "stw: troppi tensori (>256)\n"); exit(1); }
    StwT *t = &stw_t[stw_n++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    int sl = snprintf(t->shape, sizeof(t->shape), "[");
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        sl += snprintf(t->shape + sl, sizeof(t->shape) - sl, "%s%lld", i ? "," : "", (long long)shape[i]);
        n *= shape[i];
    }
    snprintf(t->shape + sl, sizeof(t->shape) - sl, "]");
    t->numel = n;
    t->data = malloc((size_t)n * sizeof(float));
    if (!t->data) { fprintf(stderr, "stw: OOM tensore %s\n", name); exit(1); }
    memcpy(t->data, data, (size_t)n * sizeof(float));
}

/* scrive il file safetensors e resetta lo stato (libera le copie) */
static void stw_write(const char *path) {
    char *hdr = malloc(1 << 16);
    if (!hdr) { fprintf(stderr, "stw: OOM header\n"); exit(1); }
    int hl = 0;
    hdr[hl++] = '{';
    int64_t off = 0;
    for (int i = 0; i < stw_n; i++) {
        hl += snprintf(hdr + hl, (1 << 16) - hl,
            "%s\"%s\":{\"dtype\":\"F32\",\"shape\":%s,\"data_offsets\":[%lld,%lld]}",
            i ? "," : "", stw_t[i].name, stw_t[i].shape,
            (long long)off, (long long)(off + stw_t[i].numel * 4));
        off += stw_t[i].numel * 4;
    }
    hdr[hl++] = '}';
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    uint64_t hlen = (uint64_t)hl;
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, hl, f);
    for (int i = 0; i < stw_n; i++) {
        fwrite(stw_t[i].data, 4, stw_t[i].numel, f);
        free(stw_t[i].data);
    }
    fclose(f);
    free(hdr);
    stw_n = 0;
}

#endif /* STW_H */
