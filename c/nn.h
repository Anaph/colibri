/* Nucleo condiviso dei motori densi (qwen.c, gemma.c): kernel matmul f32/int8,
 * quantizzazione per riga, norme, sampler temperatura+nucleus, stop-set.
 * Tutto static, un'istanza per translation unit (stesso pattern di st.h).
 * Niente di architetturale qui: attention/mlp/step restano per-motore. */
#ifndef NN_H
#define NN_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#include "simd.h"

/* peso denso: f32 oppure int8+scala per riga (QBITS=8) */
typedef struct { float *f; int8_t *q; float *qs; int O, I; } Mat;

/* tetto sulla riga di attivazione quantizzabile al volo in matmul_q */
#define NN_QROW_MAX 16384

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
#endif
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* y[S,O] = x[S,I] @ W^T,  W e' [O,I] row-major */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            y[(int64_t)s * O + o] = dot_f32(xs, w, I);
        }
    }
}

/* y[1,O] = x[1,I] @ W^T con W int8 + scala per riga: schema Q8_0 di glm.c su
 * TUTTE le piattaforme (attivazione quantizzata per riga, dot INTERO via
 * dot_i8i8). xi vive sullo stack del thread chiamante ed e' letto in
 * condivisione dentro la regione omp: nessuna copia per thread. */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I <= NN_QROW_MAX) {
        int8_t xi[NN_QROW_MAX];
        float sx = qrow_i8(x, xi, I);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++)
            y[o] = scale[o] * sx * (float)dot_i8i8(q + (int64_t)o*I, xi, I);
        return;
    }
    /* IDOT=0: percorso esatto f32*int8, invariato */
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

/* quantizzazione simmetrica per riga (come olmoe.c) */
static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax / qmax; if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v >  qmax) v =  qmax;
            if (v < -qmax-1) v = -qmax-1;
            qr[i] = (int8_t)v;
        }
    }
}

/* y[S,O] = x[S,I] @ W^T qualunque sia lo storage del peso */
static void mat_apply(float *y, const float *x, const Mat *w, int S) {
    if (w->q) { for (int s = 0; s < S; s++) matmul_q(y + (int64_t)s*w->O, x + (int64_t)s*w->I, w->q, w->qs, w->I, w->O); }
    else matmul(y, x, w->f, S, w->I, w->O);
}

static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- sampling (temperatura + nucleus), portato da glm.c ---------- */
static float g_temp = 0.7f, g_nuc = 0.95f;
static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static inline double rndu(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>7; g_rng^=g_rng<<17;
    return (double)(g_rng>>11)*(1.0/9007199254740992.0); }
static inline int argmax_v(const float *lo, int V){
    int b=0; float bv=lo[0];
    for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;}
    return b;
}
static float *g_pbuf=NULL; static int *g_pidx=NULL;
static int cmp_pdesc(const void *a,const void *b){
    float pa=g_pbuf[*(const int*)a], pb=g_pbuf[*(const int*)b];
    return pa<pb ? 1 : pa>pb ? -1 : 0; }
static void dist_build(const float *lo, int V){
    if(!g_pbuf){ g_pbuf=falloc(V); g_pidx=malloc(V*sizeof(int)); }
    float mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double s=0; float invt=1.f/(g_temp>1e-4f?g_temp:1e-4f);
    for(int i=0;i<V;i++){ g_pbuf[i]=expf((lo[i]-mx)*invt); s+=g_pbuf[i]; }
    for(int i=0;i<V;i++) g_pbuf[i]/=(float)s;
    if(g_nuc>0 && g_nuc<1.f){
        for(int i=0;i<V;i++) g_pidx[i]=i;
        qsort(g_pidx,V,sizeof(int),cmp_pdesc);
        double cum=0; int keep=V;
        for(int i=0;i<V;i++){ cum+=g_pbuf[g_pidx[i]]; if(cum>=g_nuc){ keep=i+1; break; } }
        double s2=0; for(int i=keep;i<V;i++) g_pbuf[g_pidx[i]]=0;
        for(int i=0;i<keep;i++) s2+=g_pbuf[g_pidx[i]];
        for(int i=0;i<keep;i++) g_pbuf[g_pidx[i]]/=(float)s2;
    }
}
static int dist_sample(int V){
    double u = rndu(), cum=0;
    for(int i=0;i<V;i++){ cum+=g_pbuf[i]; if(cum>=u) return i; }
    for(int i=V-1;i>=0;i--) if(g_pbuf[i]>0) return i;
    return 0;
}
static int pick_tok(const float *lo, int V){
    if(g_temp<=0) return argmax_v(lo,V);
    dist_build(lo,V);
    return dist_sample(V);
}

/* ---------- stop-set ---------- */
static int g_stop[9], g_nstop=0;
static inline int is_stop(int t){ for(int i=0;i<g_nstop;i++) if(t==g_stop[i]) return 1; return 0; }
static void stop_add(int t){ if(t>=0 && !is_stop(t) && g_nstop<9) g_stop[g_nstop++]=t; }

#endif /* NN_H */
