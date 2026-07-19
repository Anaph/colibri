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

/* OpenMP: header reale se compilato con -fopenmp, altrimenti stub inline a
 * un thread, cosi' i chiamanti (THREADS, scratch per-thread) non hanno
 * bisogno di #ifdef sparsi. */
#ifdef _OPENMP
#include <omp.h>
#else
static inline int  omp_get_max_threads(void) { return 1; }
static inline int  omp_get_thread_num(void)  { return 0; }
static inline void omp_set_num_threads(int n) { (void)n; }
#endif

/* peso denso: f32, int8+scala per riga (QBITS=8), oppure STREAMATO dal disco
 * (micro-RSS: f/q NULL, sh/sname puntano al descrittore safetensors; il
 * matmul legge le righe a blocchi e non tiene nulla residente). nn.h non
 * conosce st.h: il dispatch passa per il puntatore g_mat_stream_fn, che
 * runtime.h installa quando attiva il micro-RSS. */
typedef struct Mat { float *f; int8_t *q; float *qs; int O, I;
                     uint8_t *q4; int gs;       /* int4 impacchettato (QBITS=4); le scale
                                                 * stanno in qs: [O] se gs==0 (per riga),
                                                 * [O][ceil(I/gs)] se gs>0 (per gruppo) */
                     const void *sh; const char *sname; } Mat;
static void (*g_mat_stream_fn)(float *y, const float *x, const struct Mat *w, int S) = NULL;

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

/* y[S,O] = x[S,I] @ W^T con W int8 + scala per riga: schema Q8_0 di glm.c su
 * TUTTE le piattaforme (attivazione quantizzata per riga, dot INTERO via
 * dot_i8i8). Le S righe di attivazione vengono quantizzate UNA volta e ogni
 * riga di peso viene letta UNA volta per tutte le S righe: in prefill il
 * traffico sui pesi non cresce con S. Scratch statico che cresce e basta:
 * contratto di chiamata SERIALE (mai da dentro una regione parallela), come
 * tutti i kernel di questo header. */
static void matmul_q_s(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O) {
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I <= NN_QROW_MAX) {
        static int8_t *xi = NULL; static float *sx = NULL;
        static int64_t xcap = 0, scap = 0;
        if ((int64_t)S*I > xcap) {
            xcap = (int64_t)S*I;
            xi = realloc(xi, xcap);
            if (!xi) { fprintf(stderr, "OOM %ld\n", (long)xcap); exit(1); }
        }
        if (S > scap) {
            scap = S;
            sx = realloc(sx, scap*sizeof(float));
            if (!sx) { fprintf(stderr, "OOM %ld\n", (long)scap); exit(1); }
        }
        for (int s = 0; s < S; s++) sx[s] = qrow_i8(x + (int64_t)s*I, xi + (int64_t)s*I, I);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o*I;
            for (int s = 0; s < S; s++)
                y[(int64_t)s*O + o] = scale[o] * sx[s] * (float)dot_i8i8(w, xi + (int64_t)s*I, I);
        }
        return;
    }
    /* IDOT=0: percorso esatto f32*int8, invariato */
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s*I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * (float)w[i];
            y[(int64_t)s*O + o] = acc * scale[o];
        }
    }
}

/* forma a riga singola, per i chiamanti che restano token-per-token */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    matmul_q_s(y, x, q, scale, 1, I, O);
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

/* ---------- int4 impacchettato (layout glm.c: 2 valori/byte, indice pari nel
 * nibble basso, memorizzati come v+8, stride di riga (I+1)/2) ---------- */

/* quantizza w[O,I] f32 -> int4 + scala[O] simmetrica per riga (glm pack_int4) */
static void pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I) {
    int rb = (I+1)/2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o*I; float amax = 0;
        for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax/7.f; if (s < 1e-8f) s = 1e-8f; scale[o] = s;
        uint8_t *qr = q4 + (int64_t)o*rb;
        for (int i = 0; i < I; i += 2) {
            int v0 = (int)lrintf(wr[i]/s); if (v0 > 7) v0 = 7; if (v0 < -8) v0 = -8;
            int v1 = 0;
            if (i+1 < I) { v1 = (int)lrintf(wr[i+1]/s); if (v1 > 7) v1 = 7; if (v1 < -8) v1 = -8; }
            qr[i>>1] = (uint8_t)((v0+8) | ((v1+8)<<4));
        }
    }
}

/* quantizza w[O,I] f32 -> int4 + scale PER GRUPPO [O][ng], ng=ceil(I/gs).
 * gs multiplo di 16 (lane AVX2 del kernel grouped, come glm fmt=4). */
static void pack_int4_grouped(const float *w, uint8_t *q4, float *scale, int O, int I, int gs) {
    if (gs % 16) { fprintf(stderr, "pack_int4_grouped: gs=%d non multiplo di 16\n", gs); exit(1); }
    int rb = (I+1)/2, ng = (I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o*I;
        uint8_t *qr = q4 + (int64_t)o*rb;
        float *scl = scale + (int64_t)o*ng;
        for (int g = 0; g*gs < I; g++) {
            int base = g*gs, glen = gs; if (base+glen > I) glen = I-base;
            float amax = 0;
            for (int i = base; i < base+glen; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
            float s = amax/7.f; if (s < 1e-8f) s = 1e-8f; scl[g] = s;
            for (int i = base; i < base+glen; i += 2) {
                int v0 = (int)lrintf(wr[i]/s); if (v0 > 7) v0 = 7; if (v0 < -8) v0 = -8;
                int v1 = 0;
                if (i+1 < base+glen) { v1 = (int)lrintf(wr[i+1]/s); if (v1 > 7) v1 = 7; if (v1 < -8) v1 = -8; }
                qr[i>>1] = (uint8_t)((v0+8) | ((v1+8)<<4));
            }
        }
    }
}

/* y[S,O] = x[S,I] @ W^T con W int4 + scala per riga: kernel ESATTO f32×int4
 * (dequant al volo, attivazioni f32 — glm matmul_i4). Niente IDOT qui: sulle
 * matrici int4 l'errore extra della doppia quantizzazione non ripaga. */
static void matmul_i4_s(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O) {
    int rb = (I+1)/2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o*rb; float sc = scale[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s*I; float a = 0; int i = 0;
#ifdef __AVX2__
            const __m128i m4 = _mm_set1_epi8(0x0F); const __m256i b8 = _mm256_set1_epi32(8);
            __m256 acc = _mm256_setzero_ps();
            for (; i+16 <= I; i += 16) {
                __m128i by = _mm_loadl_epi64((const __m128i*)(w+(i>>1)));   /* 8 byte = 16 nibble */
                __m128i lo = _mm_and_si128(by,m4), hi = _mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i nib = _mm_unpacklo_epi8(lo,hi);                     /* nibble in ordine */
                __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc);
            }
            a = simd_hsum256_f32(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m4 = vdup_n_u8(0x0F); const int8x8_t b8 = vdup_n_s8(8);
            float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
            for (; i+16 <= I; i += 16) {
                uint8x8_t by = vld1_u8(w+(i>>1));                           /* 8 byte = 16 nibble */
                uint8x8x2_t z = vzip_u8(vand_u8(by,m4), vshr_n_u8(by,4));   /* nibble in ordine */
                int16x8_t w0 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]),b8));
                int16x8_t w1 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]),b8));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
            }
            a = vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for (; i+1 < I; i += 2) {
                uint8_t byte = w[i>>1]; int lo = (int)(byte&0xF)-8, hi = (int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi;
            }
            if (i < I) { uint8_t byte = w[i>>1]; a += xs[i]*(float)((int)(byte&0xF)-8); }
            y[(int64_t)s*O + o] = a*sc;
        }
    }
}

/* come sopra ma con scale per GRUPPO: l'accumulatore riparte a ogni confine di
 * gruppo e il parziale viene scalato con scl[g] (glm matmul_i4_grouped). */
static void matmul_i4_grouped_s(float *y, const float *x, const uint8_t *q4, const float *scale,
                                int S, int I, int O, int gs) {
    int rb = (I+1)/2, ng = (I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o*rb;
        const float *scl = scale + (int64_t)o*ng;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s*I; float a = 0;
            for (int g = 0; g*gs < I; g++) {
                int base = g*gs, glen = gs; if (base+glen > I) glen = I-base;
                float sc = scl[g];
                int i = base;
#ifdef __AVX2__
                const __m128i m4 = _mm_set1_epi8(0x0F); const __m256i b8 = _mm256_set1_epi32(8);
                __m256 acc = _mm256_setzero_ps();
                for (; i+16 <= base+glen; i += 16) {
                    __m128i by = _mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo = _mm_and_si128(by,m4), hi = _mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib = _mm_unpacklo_epi8(lo,hi);
                    __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                    __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc);
                }
                a += simd_hsum256_f32(acc)*sc;
#endif
                for (; i < base+glen; i += 2) {   /* coda scalare del gruppo */
                    if (i+1 < base+glen) {
                        uint8_t byte = w[i>>1];
                        a += (xs[i]*(float)((int)(byte&0xF)-8) + xs[i+1]*(float)((int)(byte>>4)-8))*sc;
                    } else {
                        uint8_t byte = w[i>>1];
                        a += xs[i]*(float)((int)(byte&0xF)-8)*sc;
                    }
                }
            }
            y[(int64_t)s*O + o] = a;
        }
    }
}

/* y[S,O] = x[S,I] @ W^T qualunque sia lo storage del peso */
static void mat_apply(float *y, const float *x, const Mat *w, int S) {
    if (g_mat_stream_fn && w->sh) { g_mat_stream_fn(y, x, w, S); return; }
    if (w->q4) {
        if (w->gs > 0) matmul_i4_grouped_s(y, x, w->q4, w->qs, S, w->I, w->O, w->gs);
        else matmul_i4_s(y, x, w->q4, w->qs, S, w->I, w->O);
        return;
    }
    if (w->q) matmul_q_s(y, x, w->q, w->qs, S, w->I, w->O);
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
