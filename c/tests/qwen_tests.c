/* Logica dei test del motore qwen (C puro; glue gtest in qwen_gtest.cc).
 * Ogni funzione qt_* ritorna 0=ok, 1=fail; dettagli su stderr.
 * Include il motore intero: accesso diretto a static e strutture. */
#define QWEN_TEST
#include "../qwen.c"
#include "tiny_st.h"
#include "tiny_gguf.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static uint64_t qt_rng_s = 42;
static float qt_frnd(void) {
    qt_rng_s ^= qt_rng_s<<13; qt_rng_s ^= qt_rng_s>>7; qt_rng_s ^= qt_rng_s<<17;
    return (float)((qt_rng_s>>11)*(1.0/9007199254740992.0)) - 0.5f;
}
static void qt_fill(float *p, int64_t n, float sc) { for (int64_t i = 0; i < n; i++) p[i] = qt_frnd()*sc; }

/* ---- RoPE: pos=0 identita', norma preservata, valori vs riferimento double ---- */
int qt_rope(void) {
    int hd = 8; float theta = 1000000.f;
    float x[8], x0[8];
    qt_rng_s = 42;
    for (int i = 0; i < hd; i++) x[i] = x0[i] = qt_frnd();
    rope_head(x, 0, theta, hd);
    for (int i = 0; i < hd; i++) CHECK(fabsf(x[i]-x0[i]) < 1e-6f);
    memcpy(x, x0, sizeof(x));
    rope_head(x, 17, theta, hd);
    double n0 = 0, n1 = 0;
    for (int i = 0; i < hd; i++) { n0 += (double)x0[i]*x0[i]; n1 += (double)x[i]*x[i]; }
    CHECK(fabs(n0-n1) < 1e-5);
    int h = hd/2;
    for (int j = 0; j < h; j++) {
        double inv = pow((double)theta, -2.0*j/hd);
        double ang = 17.0*inv, cs = cos(ang), sn = sin(ang);
        double a = x0[j], b = x0[j+h];
        CHECK(fabs(x[j]   - (a*cs - b*sn)) < 1e-5);
        CHECK(fabs(x[j+h] - (b*cs + a*sn)) < 1e-5);
    }
    return 0;
}

/* ---- GQA: attenzione con KV<H deve coincidere con la MHA a teste kv replicate ---- */
int qt_gqa(void) {
    int D = 16, H = 4, KV = 2, hd = 8, G = H/KV, S = 5;
    Model A, B; memset(&A,0,sizeof A); memset(&B,0,sizeof B);
    A.c.hidden=D; A.c.n_heads=H; A.c.n_kv_heads=KV; A.c.head_dim=hd; A.c.rot=hd;
    A.c.theta=1e6f; A.c.eps=1e-6f; A.c.n_layers=1;
    static int lt[1] = {LT_FULL}; A.c.ltype = lt;
    B.c = A.c; B.c.n_kv_heads = H;
    Layer la, lb; memset(&la,0,sizeof la); memset(&lb,0,sizeof lb);
    la.q.f=falloc((int64_t)H*hd*D);  la.q.O=H*hd;  la.q.I=D;
    la.k.f=falloc((int64_t)KV*hd*D); la.k.O=KV*hd; la.k.I=D;
    la.v.f=falloc((int64_t)KV*hd*D); la.v.O=KV*hd; la.v.I=D;
    la.o.f=falloc((int64_t)D*H*hd);  la.o.O=D;     la.o.I=H*hd;
    la.qn=falloc(hd); la.kn=falloc(hd);
    qt_rng_s = 42;
    qt_fill(la.q.f,(int64_t)H*hd*D,1); qt_fill(la.k.f,(int64_t)KV*hd*D,1); qt_fill(la.v.f,(int64_t)KV*hd*D,1);
    qt_fill(la.o.f,(int64_t)D*H*hd,1); qt_fill(la.qn,hd,1); qt_fill(la.kn,hd,1);
    lb = la;
    lb.k.f=falloc((int64_t)H*hd*D); lb.k.O=H*hd; lb.k.I=D;
    lb.v.f=falloc((int64_t)H*hd*D); lb.v.O=H*hd; lb.v.I=D;
    for (int hh = 0; hh < H; hh++) {
        memcpy(lb.k.f + (int64_t)hh*hd*D, la.k.f + (int64_t)(hh/G)*hd*D, (int64_t)hd*D*sizeof(float));
        memcpy(lb.v.f + (int64_t)hh*hd*D, la.v.f + (int64_t)(hh/G)*hd*D, (int64_t)hd*D*sizeof(float));
    }
    kv_alloc(&A, 16); kv_alloc(&B, 16);
    float *x = falloc((int64_t)S*D); qt_fill(x,(int64_t)S*D,1);
    float *oa = falloc((int64_t)S*D), *ob = falloc((int64_t)S*D);
    attention(&A, &la, 0, x, S, 0, oa);
    attention(&B, &lb, 0, x, S, 0, ob);
    for (int64_t i = 0; i < (int64_t)S*D; i++) CHECK(fabsf(oa[i]-ob[i]) < 1e-5f);
    float *xt = falloc(D); qt_fill(xt,D,1);
    float *oa1 = falloc(D), *ob1 = falloc(D);
    attention(&A, &la, 0, xt, 1, S, oa1);
    attention(&B, &lb, 0, xt, 1, S, ob1);
    for (int i = 0; i < D; i++) CHECK(fabsf(oa1[i]-ob1[i]) < 1e-5f);
    return 0;
}

/* ---- quantizzazione: matmul_q entro tolleranza dal f32 ---- */
int qt_quant(void) {
    int O = 24, I = 32;
    float *w = falloc((int64_t)O*I), *x = falloc(I), *y = falloc(O), *yq = falloc(O);
    qt_rng_s = 43;
    qt_fill(w,(int64_t)O*I,1); qt_fill(x,I,1);
    int8_t *q = malloc((int64_t)O*I); float *qs = falloc(O);
    quantize_rows(w, q, qs, O, I, 8);
    matmul(y, x, w, 1, I, O);
    matmul_q(yq, x, q, qs, I, O);
    for (int o = 0; o < O; o++) CHECK(fabsf(y[o]-yq[o]) < 0.05f);
    return 0;
}

/* ---- matmul_q_s: batch S=4 bit-esatto rispetto a 4 chiamate S=1 ---- */
int qt_quant_batch(void) {
    int O = 24, I = 32, S = 4;
    float *w = falloc((int64_t)O*I), *x = falloc((int64_t)S*I);
    qt_rng_s = 45;
    qt_fill(w,(int64_t)O*I,1); qt_fill(x,(int64_t)S*I,1);
    int8_t *q = malloc((int64_t)O*I); float *qs = falloc(O);
    quantize_rows(w, q, qs, O, I, 8);
    float *yb = falloc((int64_t)S*O), *y1 = falloc((int64_t)S*O);
    matmul_q_s(yb, x, q, qs, S, I, O);
    for (int s = 0; s < S; s++) matmul_q(y1 + (int64_t)s*O, x + (int64_t)s*I, q, qs, I, O);
    CHECK(memcmp(yb, y1, (size_t)S*O*sizeof(float)) == 0);
    free(w); free(x); free(q); free(qs); free(yb); free(y1);
    return 0;
}

/* ---- int4: pack/dequant roundtrip + ordine dei nibble ---- */
int qt_int4_pack(void) {
    static const int Is[] = {15, 16, 17, 32};
    int O = 6;
    for (size_t k = 0; k < sizeof Is/sizeof Is[0]; k++) {
        int I = Is[k], rb = (I+1)/2;
        float *w = falloc((int64_t)O*I); float *qs = falloc(O);
        uint8_t *q4 = malloc((size_t)O*rb);
        qt_rng_s = 100 + I;
        qt_fill(w, (int64_t)O*I, 2.f);
        pack_int4(w, q4, qs, O, I);
        for (int o = 0; o < O; o++) {
            float s = qs[o];
            for (int i = 0; i < I; i++) {
                uint8_t b = q4[(int64_t)o*rb + (i>>1)];
                int v = (i & 1) ? (int)(b>>4)-8 : (int)(b&0xF)-8;   /* pari -> nibble basso */
                CHECK(v >= -8 && v <= 7);
                CHECK(fabsf(w[(int64_t)o*I+i] - s*(float)v) <= s*0.5f + 1e-6f);
            }
        }
        free(w); free(qs); free(q4);
    }
    return 0;
}

/* ---- int4 grouped: roundtrip per gruppo + gs=I equivale al per-riga ---- */
int qt_int4_grouped(void) {
    int O = 5, I = 80, gs = 16, ng = (I+gs-1)/gs, rb = (I+1)/2;
    float *w = falloc((int64_t)O*I); float *qs = falloc((int64_t)O*ng);
    uint8_t *q4 = malloc((size_t)O*rb);
    qt_rng_s = 321;
    qt_fill(w, (int64_t)O*I, 2.f);
    pack_int4_grouped(w, q4, qs, O, I, gs);
    for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
        float s = qs[(int64_t)o*ng + i/gs];
        uint8_t b = q4[(int64_t)o*rb + (i>>1)];
        int v = (i & 1) ? (int)(b>>4)-8 : (int)(b&0xF)-8;
        CHECK(fabsf(w[(int64_t)o*I+i] - s*(float)v) <= s*0.5f + 1e-6f);
    }
    /* gs che copre l'intera riga: stessi byte e stesse scale del per-riga */
    int gs2 = 80;   /* multiplo di 16 */
    float *qs1 = falloc(O); uint8_t *q41 = malloc((size_t)O*rb);
    float *qs2 = falloc(O); uint8_t *q42 = malloc((size_t)O*rb);
    pack_int4(w, q41, qs1, O, I);
    pack_int4_grouped(w, q42, qs2, O, I, gs2);
    CHECK(memcmp(q41, q42, (size_t)O*rb) == 0);
    CHECK(memcmp(qs1, qs2, (size_t)O*sizeof(float)) == 0);
    /* e i due matmul coincidono numericamente (tolleranza: ordine di accumulo) */
    int S = 3;
    float *x = falloc((int64_t)S*I), *y1 = falloc((int64_t)S*O), *y2 = falloc((int64_t)S*O);
    qt_fill(x, (int64_t)S*I, 1.f);
    matmul_i4_s(y1, x, q41, qs1, S, I, O);
    matmul_i4_grouped_s(y2, x, q42, qs2, S, I, O, gs2);
    for (int64_t i = 0; i < (int64_t)S*O; i++) CHECK(fabsf(y1[i]-y2[i]) <= 1e-4f*(1.f+fabsf(y1[i])));
    free(w); free(qs); free(q4); free(qs1); free(q41); free(qs2); free(q42);
    free(x); free(y1); free(y2);
    return 0;
}

/* ---- matmul int4 vs riferimento double su pesi dequantizzati: il kernel
 * calcola ESATTAMENTE dot(x, s*v) a meno dell'ordine di accumulo f32 ---- */
int qt_int4_matmul_ref(void) {
    int O = 24, I = 50, S = 4, gs = 16, ng = (I+gs-1)/gs, rb = (I+1)/2;
    float *w = falloc((int64_t)O*I), *x = falloc((int64_t)S*I);
    qt_rng_s = 654;
    qt_fill(w, (int64_t)O*I, 1.f); qt_fill(x, (int64_t)S*I, 1.f);
    float *qsr = falloc(O); uint8_t *q4 = malloc((size_t)O*rb);
    float *qsg = falloc((int64_t)O*ng); uint8_t *q4g = malloc((size_t)O*rb);
    pack_int4(w, q4, qsr, O, I);
    pack_int4_grouped(w, q4g, qsg, O, I, gs);
    float *yr = falloc((int64_t)S*O), *yg = falloc((int64_t)S*O);
    matmul_i4_s(yr, x, q4, qsr, S, I, O);
    matmul_i4_grouped_s(yg, x, q4g, qsg, S, I, O, gs);
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double rr = 0, rg = 0;
        for (int i = 0; i < I; i++) {
            uint8_t b = q4[(int64_t)o*rb + (i>>1)];
            int v = (i & 1) ? (int)(b>>4)-8 : (int)(b&0xF)-8;
            rr += (double)x[(int64_t)s*I+i] * (double)qsr[o] * v;
            uint8_t bg = q4g[(int64_t)o*rb + (i>>1)];
            int vg = (i & 1) ? (int)(bg>>4)-8 : (int)(bg&0xF)-8;
            rg += (double)x[(int64_t)s*I+i] * (double)qsg[(int64_t)o*ng + i/gs] * vg;
        }
        CHECK(fabs((double)yr[(int64_t)s*O+o] - rr) <= 1e-4*(1.0+fabs(rr)));
        CHECK(fabs((double)yg[(int64_t)s*O+o] - rg) <= 1e-4*(1.0+fabs(rg)));
    }
    /* batch-invarianza: S=4 in un colpo == 4 chiamate S=1, bit-esatto */
    float *y1 = falloc((int64_t)S*O);
    for (int s = 0; s < S; s++) matmul_i4_s(y1 + (int64_t)s*O, x + (int64_t)s*I, q4, qsr, 1, I, O);
    CHECK(memcmp(yr, y1, (size_t)S*O*sizeof(float)) == 0);
    float *y1g = falloc((int64_t)S*O);
    for (int s = 0; s < S; s++) matmul_i4_grouped_s(y1g + (int64_t)s*O, x + (int64_t)s*I, q4g, qsg, 1, I, O, gs);
    CHECK(memcmp(yg, y1g, (size_t)S*O*sizeof(float)) == 0);
    /* mat_apply instrada sul kernel int4 giusto */
    Mat mq; memset(&mq, 0, sizeof(mq));
    mq.q4 = q4g; mq.qs = qsg; mq.gs = gs; mq.O = O; mq.I = I;
    float *ya = falloc((int64_t)S*O);
    mat_apply(ya, x, &mq, S);
    CHECK(memcmp(ya, yg, (size_t)S*O*sizeof(float)) == 0);
    free(w); free(x); free(qsr); free(q4); free(qsg); free(q4g);
    free(yr); free(yg); free(y1); free(y1g); free(ya);
    return 0;
}

/* ---- sampler: stesso seed -> stessa sequenza; greedy = argmax ---- */
int qt_sampler(void) {
    int V = 100; float lo[100];
    qt_rng_s = 44;
    for (int i = 0; i < V; i++) lo[i] = qt_frnd()*4;
    float t0 = g_temp, n0 = g_nuc;
    g_temp = 0; CHECK(pick_tok(lo,V) == argmax_v(lo,V));
    g_temp = 0.8f; g_nuc = 0.9f;
    int a[16], b[16];
    g_rng = 12345; for (int i = 0; i < 16; i++) a[i] = pick_tok(lo,V);
    g_rng = 12345; for (int i = 0; i < 16; i++) b[i] = pick_tok(lo,V);
    g_temp = t0; g_nuc = n0;
    for (int i = 0; i < 16; i++) CHECK(a[i]==b[i]);
    return 0;
}

/* ---- edge-case numerici del deltanet ---- */
int qt_edges(void) {
    CHECK(fabsf(softplusf(50.f) - 50.f) < 1e-4f);
    CHECK(fabsf(softplusf(0.f) - logf(2.f)) < 1e-6f);
    CHECK(softplusf(-50.f) >= 0.f && softplusf(-50.f) < 1e-6f);
    CHECK(fabsf(sigmoidf(0.f) - 0.5f) < 1e-7f);
    float z[4] = {0,0,0,0};
    l2norm_head(z, 4);
    for (int i = 0; i < 4; i++) CHECK(isfinite(z[i]) && z[i]==0.f);
    return 0;
}

/* ---- layout q_proj gated: gate=0 (sigmoid 0.5) => output = 0.5 * non-gated ---- */
int qt_gated_layout(void) {
    int D = 16, H = 2, KV = 2, hd = 8, S = 4;
    Model A, B; memset(&A,0,sizeof A); memset(&B,0,sizeof B);
    A.c.hidden=D; A.c.n_heads=H; A.c.n_kv_heads=KV; A.c.head_dim=hd; A.c.rot=hd/2; /* esercita il partial rope */
    A.c.theta=1e6f; A.c.eps=1e-6f; A.c.n_layers=1;
    static int lt[1] = {LT_FULL}; A.c.ltype = lt;
    B.c = A.c;
    Layer la, lb; memset(&la,0,sizeof la); memset(&lb,0,sizeof lb);
    la.q.f=falloc((int64_t)H*hd*D);  la.q.O=H*hd;  la.q.I=D;
    la.k.f=falloc((int64_t)KV*hd*D); la.k.O=KV*hd; la.k.I=D;
    la.v.f=falloc((int64_t)KV*hd*D); la.v.O=KV*hd; la.v.I=D;
    la.o.f=falloc((int64_t)D*H*hd);  la.o.O=D;     la.o.I=H*hd;
    la.qn=falloc(hd); la.kn=falloc(hd);
    qt_rng_s = 99;
    for (int64_t i=0;i<(int64_t)H*hd*D;i++) la.q.f[i]=qt_frnd();
    for (int64_t i=0;i<(int64_t)KV*hd*D;i++){ la.k.f[i]=qt_frnd(); la.v.f[i]=qt_frnd(); }
    for (int64_t i=0;i<(int64_t)D*H*hd;i++) la.o.f[i]=qt_frnd();
    for (int i=0;i<hd;i++){ la.qn[i]=1; la.kn[i]=1; }
    lb = la;
    lb.gated = 1;
    lb.q.f=falloc((int64_t)2*H*hd*D); lb.q.O=2*H*hd; lb.q.I=D;
    for (int hh = 0; hh < H; hh++) {
        memcpy(lb.q.f + (int64_t)hh*2*hd*D,        la.q.f + (int64_t)hh*hd*D, (int64_t)hd*D*sizeof(float));
        memset(lb.q.f + (int64_t)hh*2*hd*D + (int64_t)hd*D, 0, (int64_t)hd*D*sizeof(float));
    }
    kv_alloc(&A, 8); kv_alloc(&B, 8);
    float *x = falloc((int64_t)S*D); for (int64_t i=0;i<(int64_t)S*D;i++) x[i]=qt_frnd();
    float *oa = falloc((int64_t)S*D), *ob = falloc((int64_t)S*D);
    attention(&A, &la, 0, x, S, 0, oa);
    attention(&B, &lb, 0, x, S, 0, ob);
    for (int64_t i = 0; i < (int64_t)S*D; i++) CHECK(fabsf(ob[i] - 0.5f*oa[i]) < 1e-5f);
    return 0;
}

/* ---- Gated DeltaNet vs riferimento indipendente in double ----
 * Riferimento riscritto dalle formule di transformers (torch_recurrent_gated_
 * delta_rule + modular_qwen3_5) in cicli vettoriali double, NON dai loop fusi
 * del motore. LCG identico al vecchio generatore Python (stessi stimoli). */
static uint64_t dn_lcg;
static double dn_rnd(void) {
    dn_lcg = dn_lcg*6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(dn_lcg>>33)/2147483648.0 - 0.5;
}
static double dn_silu(double x){ return x/(1.0+exp(-x)); }
static double dn_softplus(double x){ return x>20.0 ? x : log1p(exp(x)); }
static double dn_sigmoid(double x){ return 1.0/(1.0+exp(-x)); }

static int qt_deltanet_case(int D, int Hv, int Hk, int dk, int dv, int K, int T) {
    int kd = Hk*dk, vd = Hv*dv, cd = 2*kd + vd, R = Hv/Hk;
    dn_lcg = 123456789ULL;
    /* pesi e input in double, nello stesso ordine del vecchio generatore */
    double *W_qkv = malloc(sizeof(double)*cd*D), *W_z = malloc(sizeof(double)*vd*D);
    double *W_b = malloc(sizeof(double)*Hv*D), *W_a = malloc(sizeof(double)*Hv*D);
    double *conv_w = malloc(sizeof(double)*cd*K), *conv_b = malloc(sizeof(double)*cd);
    double *dt_bias = malloc(sizeof(double)*Hv), *A_log = malloc(sizeof(double)*Hv);
    double *norm_w = malloc(sizeof(double)*dv), *W_out = malloc(sizeof(double)*D*vd);
    double *X = malloc(sizeof(double)*T*D);
    for (int i = 0; i < cd*D; i++) W_qkv[i] = dn_rnd();
    for (int i = 0; i < vd*D; i++) W_z[i] = dn_rnd();
    for (int i = 0; i < Hv*D; i++) W_b[i] = dn_rnd();
    for (int i = 0; i < Hv*D; i++) W_a[i] = dn_rnd();
    for (int i = 0; i < cd*K; i++) conv_w[i] = dn_rnd();
    for (int i = 0; i < cd; i++) conv_b[i] = dn_rnd()*0.1;
    for (int i = 0; i < Hv; i++) dt_bias[i] = dn_rnd();
    for (int i = 0; i < Hv; i++) A_log[i] = dn_rnd();
    for (int i = 0; i < dv; i++) norm_w[i] = 1.0 + dn_rnd()*0.1;
    for (int i = 0; i < D*vd; i++) W_out[i] = dn_rnd();
    for (int i = 0; i < T*D; i++) X[i] = dn_rnd();

    /* --- riferimento double --- */
    double *cs = calloc((size_t)cd*K, sizeof(double));         /* conv window */
    double *S  = calloc((size_t)Hv*dk*dv, sizeof(double));     /* stato ricorrente */
    double *expect = malloc(sizeof(double)*T*D);
    double *qkv = malloc(sizeof(double)*cd), *z = malloc(sizeof(double)*vd);
    double *b = malloc(sizeof(double)*Hv), *a = malloc(sizeof(double)*Hv);
    double *kvbuf = malloc(sizeof(double)*dv), *delta = malloc(sizeof(double)*dv);
    double *o_all = malloc(sizeof(double)*vd);
    for (int t = 0; t < T; t++) {
        const double *x = X + t*D;
        for (int o = 0; o < cd; o++) { double s=0; for (int i=0;i<D;i++) s += W_qkv[o*D+i]*x[i]; qkv[o]=s; }
        for (int o = 0; o < vd; o++) { double s=0; for (int i=0;i<D;i++) s += W_z[o*D+i]*x[i]; z[o]=s; }
        for (int o = 0; o < Hv; o++) { double s=0; for (int i=0;i<D;i++) s += W_b[o*D+i]*x[i]; b[o]=s; }
        for (int o = 0; o < Hv; o++) { double s=0; for (int i=0;i<D;i++) s += W_a[o*D+i]*x[i]; a[o]=s; }
        for (int ch = 0; ch < cd; ch++) {
            double *w = cs + (size_t)ch*K;
            for (int j = 0; j < K-1; j++) w[j] = w[j+1];
            w[K-1] = qkv[ch];
            double v = conv_b[ch];
            for (int j = 0; j < K; j++) v += w[j]*conv_w[ch*K+j];
            qkv[ch] = dn_silu(v);
        }
        double *q = qkv, *k = qkv + kd, *v = qkv + 2*kd;
        for (int h = 0; h < Hk; h++) {
            for (int pass = 0; pass < 2; pass++) {
                double *seg = (pass ? k : q) + (size_t)h*dk;
                double s = 0; for (int i = 0; i < dk; i++) s += seg[i]*seg[i];
                double r = 1.0/sqrt(s + 1e-6);
                for (int i = 0; i < dk; i++) seg[i] *= r;
            }
        }
        double qs = 1.0/sqrt((double)dk);
        for (int i = 0; i < kd; i++) q[i] *= qs;
        for (int hv = 0; hv < Hv; hv++) {
            int hk = hv / R;
            const double *qh = q + (size_t)hk*dk, *kh = k + (size_t)hk*dk, *vh = v + (size_t)hv*dv;
            double *Sh = S + (size_t)hv*dk*dv;
            double g    = -exp(A_log[hv]) * dn_softplus(a[hv] + dt_bias[hv]);
            double beta = dn_sigmoid(b[hv]);
            double dec  = exp(g);
            for (int i = 0; i < dk*dv; i++) Sh[i] *= dec;
            for (int j = 0; j < dv; j++) { double s=0; for (int i=0;i<dk;i++) s += Sh[i*dv+j]*kh[i]; kvbuf[j]=s; }
            for (int j = 0; j < dv; j++) delta[j] = (vh[j]-kvbuf[j])*beta;
            for (int i = 0; i < dk; i++) for (int j = 0; j < dv; j++) Sh[i*dv+j] += kh[i]*delta[j];
            double *oh = o_all + (size_t)hv*dv;
            for (int j = 0; j < dv; j++) { double s=0; for (int i=0;i<dk;i++) s += Sh[i*dv+j]*qh[i]; oh[j]=s; }
            double ms = 0; for (int j = 0; j < dv; j++) ms += oh[j]*oh[j];
            double r = 1.0/sqrt(ms/dv + 1e-6);
            const double *zh = z + (size_t)hv*dv;
            for (int j = 0; j < dv; j++) oh[j] = oh[j]*r*norm_w[j]*dn_silu(zh[j]);
        }
        for (int d = 0; d < D; d++) { double s=0; for (int j=0;j<vd;j++) s += W_out[d*vd+j]*o_all[j]; expect[t*D+d]=s; }
    }

    /* --- motore in float sugli stessi pesi --- */
    Model m; memset(&m,0,sizeof m);
    m.c.hidden = D; m.c.eps = 1e-6f;
    m.c.lin_hv = Hv; m.c.lin_hk = Hk; m.c.lin_dk = dk; m.c.lin_dv = dv; m.c.lin_conv = K;
    Layer l; memset(&l,0,sizeof l);
    l.type = LT_LINEAR;
    #define MKM(mat, src, O_, I_) do { l.mat.O=O_; l.mat.I=I_; l.mat.q=NULL; l.mat.qs=NULL; \
        l.mat.f=falloc((int64_t)(O_)*(I_)); for (int64_t _i=0;_i<(int64_t)(O_)*(I_);_i++) l.mat.f[_i]=(float)src[_i]; } while(0)
    MKM(aqkv, W_qkv, cd, D);
    MKM(az,   W_z,   vd, D);
    MKM(ab,   W_b,   Hv, D);
    MKM(aa,   W_a,   Hv, D);
    MKM(aout, W_out, D, vd);
    #undef MKM
    l.conv_w = falloc((int64_t)cd*K); for (int i=0;i<cd*K;i++) l.conv_w[i]=(float)conv_w[i];
    l.conv_b = falloc(cd);            for (int i=0;i<cd;i++)   l.conv_b[i]=(float)conv_b[i];
    l.dt_bias = falloc(Hv);           for (int i=0;i<Hv;i++)   l.dt_bias[i]=(float)dt_bias[i];
    l.A_log = falloc(Hv);             for (int i=0;i<Hv;i++)   l.A_log[i]=(float)A_log[i];
    l.dn_norm = falloc(dv);           for (int i=0;i<dv;i++)   l.dn_norm[i]=(float)norm_w[i];
    l.conv_state = calloc((size_t)cd*K, sizeof(float));
    l.Sstate = calloc((size_t)Hv*dk*dv, sizeof(float));
    float *xf = falloc(D), *out = falloc(D);
    int rc = 0;
    for (int t = 0; t < T && !rc; t++) {
        for (int i = 0; i < D; i++) xf[i] = (float)X[t*D+i];
        deltanet_token(&m, &l, xf, out);
        for (int d = 0; d < D; d++) {
            if (fabs((double)out[d] - expect[t*D+d]) > 1e-4) {
                fprintf(stderr, "deltanet D=%d Hv=%d: token %d dim %d: got %.6f expected %.6f\n",
                        D, Hv, t, d, out[d], expect[t*D+d]);
                rc = 1; break;
            }
        }
    }
    free(W_qkv); free(W_z); free(W_b); free(W_a); free(conv_w); free(conv_b);
    free(dt_bias); free(A_log); free(norm_w); free(W_out); free(X);
    free(cs); free(S); free(expect); free(qkv); free(z); free(b); free(a);
    free(kvbuf); free(delta); free(o_all); free(xf); free(out);
    return rc;
}

int qt_deltanet_small(void) { return qt_deltanet_case(4, 2, 1, 4, 4, 3, 3); }
int qt_deltanet_large(void) { return qt_deltanet_case(8, 4, 2, 8, 8, 4, 6); }

/* ---- tiny end-to-end: modello sintetico su disco ---- */
static void qt_write_dense_dir(const char *dir) {
    tst_write_text(dir, "config.json",
        "{\"hidden_size\":16,\"num_hidden_layers\":2,\"num_attention_heads\":4,"
        "\"num_key_value_heads\":2,\"head_dim\":8,\"intermediate_size\":32,"
        "\"vocab_size\":32,\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
        "\"tie_word_embeddings\":true,\"eos_token_id\":0,"
        "\"max_position_embeddings\":64}");
    tst_reset(7);
    tst_add("model.embed_tokens.weight", "[32,16]", 32*16, 0.5f, 0);
    tst_add("model.norm.weight", "[16]", 16, 0, 1);
    char nm[128];
    for (int i = 0; i < 2; i++) {
        #define AT(suffix, shape, numel, sc, ones) \
            do { snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); tst_add(nm,shape,numel,sc,ones); } while(0)
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
    tst_write(dir);
}

static void qt_write_hybrid_dir(const char *dir) {
    tst_write_text(dir, "config.json",
        "{\"hidden_size\":16,\"num_hidden_layers\":2,\"num_attention_heads\":4,"
        "\"num_key_value_heads\":2,\"head_dim\":8,\"intermediate_size\":32,"
        "\"vocab_size\":32,\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
        "\"tie_word_embeddings\":true,\"eos_token_id\":0,"
        "\"max_position_embeddings\":64,"
        "\"partial_rotary_factor\":0.5,"
        "\"layer_types\":[\"linear_attention\",\"full_attention\"],"
        "\"linear_num_value_heads\":4,\"linear_num_key_heads\":2,"
        "\"linear_key_head_dim\":4,\"linear_value_head_dim\":4,"
        "\"linear_conv_kernel_dim\":3}");
    tst_reset(11);
    tst_add("model.embed_tokens.weight", "[32,16]", 32*16, 0.5f, 0);
    tst_add("model.norm.weight", "[16]", 16, 0, 1);
    tst_add("model.layers.0.input_layernorm.weight", "[16]", 16, 0, 1);
    tst_add("model.layers.0.post_attention_layernorm.weight", "[16]", 16, 0, 1);
    tst_add("model.layers.0.linear_attn.in_proj_qkv.weight", "[32,16]", 32*16, 0.3f, 0);
    tst_add("model.layers.0.linear_attn.in_proj_z.weight", "[16,16]", 16*16, 0.3f, 0);
    tst_add("model.layers.0.linear_attn.in_proj_b.weight", "[4,16]", 4*16, 0.3f, 0);
    tst_add("model.layers.0.linear_attn.in_proj_a.weight", "[4,16]", 4*16, 0.3f, 0);
    tst_add("model.layers.0.linear_attn.conv1d.weight", "[32,1,3]", 32*3, 0.3f, 0);
    tst_add("model.layers.0.linear_attn.conv1d.bias", "[32]", 32, 0.1f, 0);
    tst_add("model.layers.0.linear_attn.dt_bias", "[4]", 4, 0.3f, 0);
    tst_add("model.layers.0.linear_attn.A_log", "[4]", 4, 0.3f, 0);
    tst_add("model.layers.0.linear_attn.norm.weight", "[4]", 4, 0, 1);
    tst_add("model.layers.0.linear_attn.out_proj.weight", "[16,16]", 16*16, 0.3f, 0);
    tst_add("model.layers.0.mlp.gate_proj.weight", "[32,16]", 32*16, 0.3f, 0);
    tst_add("model.layers.0.mlp.up_proj.weight",   "[32,16]", 32*16, 0.3f, 0);
    tst_add("model.layers.0.mlp.down_proj.weight", "[16,32]", 16*32, 0.3f, 0);
    tst_add("model.layers.1.input_layernorm.weight", "[16]", 16, 0, 1);
    tst_add("model.layers.1.post_attention_layernorm.weight", "[16]", 16, 0, 1);
    tst_add("model.layers.1.self_attn.q_norm.weight", "[8]", 8, 0, 1);
    tst_add("model.layers.1.self_attn.k_norm.weight", "[8]", 8, 0, 1);
    tst_add("model.layers.1.self_attn.q_proj.weight", "[64,16]", 64*16, 0.3f, 0);
    tst_add("model.layers.1.self_attn.k_proj.weight", "[16,16]", 16*16, 0.3f, 0);
    tst_add("model.layers.1.self_attn.v_proj.weight", "[16,16]", 16*16, 0.3f, 0);
    tst_add("model.layers.1.self_attn.o_proj.weight", "[16,32]", 16*32, 0.3f, 0);
    tst_add("model.layers.1.mlp.gate_proj.weight", "[32,16]", 32*16, 0.3f, 0);
    tst_add("model.layers.1.mlp.up_proj.weight",   "[32,16]", 32*16, 0.3f, 0);
    tst_add("model.layers.1.mlp.down_proj.weight", "[16,32]", 16*32, 0.3f, 0);
    tst_write(dir);
}

/* prefill {1,2,3} + 8 passi greedy con guardia isfinite; out ha spazio per
 * 11 id. Il modello arriva gia' model_init'd + kv_alloc'd: era il ciclo
 * copiato in sei test. */
static int drive_greedy8(Model *m, int *out) {
    int prompt[3] = {1,2,3};
    memcpy(out, prompt, sizeof(prompt));
    float *logit = step(m, prompt, 3, 0);
    int len = 3;
    for (int s = 0; s < 8; s++) {
        for (int i = 0; i < m->c.vocab; i++) CHECK(isfinite(logit[i]));
        int best = argmax_v(logit, m->c.vocab);
        free(logit);
        out[len++] = best;
        if (s == 7) break;
        logit = step(m, &out[len-1], 1, len-1);
    }
    return 0;
}

/* tst_dir ritorna un buffer statico: copia i due percorsi e scrive entrambi
 * i modelli tiny (buffer da almeno 512 byte) */
static void qt_setup_both(char *dense, char *hyb) {
    snprintf(dense, 512, "%s", tst_dir("qwen_tiny_model"));
    snprintf(hyb,   512, "%s", tst_dir("qwen_tiny_hybrid"));
    qt_write_dense_dir(dense);
    qt_write_hybrid_dir(hyb);
}

/* greedy 8 token; verifica anche le strutture (tied, head_dim, ibrido) */
static int qt_run8(const char *dir, int qbits, int hybrid, int *out) {
    Model m;
    model_init(&m, dir, qbits);
    CHECK(m.lm_tied);
    CHECK(m.c.head_dim == 8);
    CHECK(m.c.hybrid == hybrid);
    if (hybrid) {
        CHECK(m.L[0].type == LT_LINEAR && m.L[1].type == LT_FULL);
        CHECK(m.L[1].gated);
        CHECK(m.c.rot == 4);
    }
    kv_alloc(&m, 16);
    return drive_greedy8(&m, out);
}

int qt_tiny_dense(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    int a[16], b[16];
    CHECK(qt_run8(dir, 0, 0, a) == 0);
    CHECK(qt_run8(dir, 0, 0, b) == 0);
    for (int i = 0; i < 11; i++) CHECK(a[i]==b[i]);
    return 0;
}

int qt_tiny_qbits(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    int q[16];
    return qt_run8(dir, 8, 0, q);
}

/* QBITS=4: layer int4 grouped, embed/lm_head restano int8; deterministico
 * su dense e ibrido */
int qt_tiny_qbits4(void) {
    char dense[512], hyb[512];
    qt_setup_both(dense, hyb);
    int a[16], b[16];
    CHECK(qt_run8(dense, 4, 0, a) == 0);
    CHECK(qt_run8(dense, 4, 0, b) == 0);
    for (int i = 0; i < 11; i++) CHECK(a[i]==b[i]);
    CHECK(qt_run8(hyb, 4, 1, a) == 0);
    /* struttura: embed int8, matrici dei layer int4 (gs=QGROUP), testa int8 */
    Model m; model_init(&m, dense, 4);
    CHECK(m.embed == NULL && m.embed_q != NULL);
    CHECK(m.lm_tied && m.lm_head.q == m.embed_q && m.lm_head.q4 == NULL);
    CHECK(m.L[0].q.q4 != NULL && m.L[0].q.q == NULL && m.L[0].q.f == NULL && m.L[0].q.gs == g_qgroup);
    CHECK(m.L[0].down.q4 != NULL && m.L[1].up.q4 != NULL);
    return 0;
}

int qt_tiny_hybrid(void) {
    const char *dir = tst_dir("qwen_tiny_hybrid");
    qt_write_hybrid_dir(dir);
    int a[16], b[16];
    CHECK(qt_run8(dir, 0, 1, a) == 0);
    CHECK(qt_run8(dir, 0, 1, b) == 0);
    for (int i = 0; i < 11; i++) CHECK(a[i]==b[i]);
    CHECK(qt_run8(dir, 8, 1, b) == 0);
    return 0;
}

/* ---- MEM_GB/MEM_FRAC: parita' token con streaming ---- */
static int qt_run8_budget(const char *dir, int qbits, int64_t budget, int *out, int *resident_out) {
    Model m;
    model_init_ex(&m, dir, qbits, budget, 16);
    if (resident_out) *resident_out = m.n_resident;
    kv_alloc(&m, 16);
    return drive_greedy8(&m, out);
}

int qt_memknob_parity(void) {
    /* stessi id greedy con: tutto residente (budget 0), tutto streamato
     * (budget minuscolo -> R=0) e budget enorme (R=n_layers) — sia sul
     * modello denso che sull'ibrido (stati deltanet sempre residenti) */
    char dense[512], hyb[512];
    qt_setup_both(dense, hyb);
    const char *dirs[2] = { dense, hyb };
    for (int d = 0; d < 2; d++) {
        int a[16], b[16], cc[16]; int r0, r1, r2;
        CHECK(qt_run8_budget(dirs[d], 0, 0, a, &r0) == 0);                 /* classico */
        CHECK(qt_run8_budget(dirs[d], 0, 1, b, &r1) == 0);                 /* R=0: tutto stream */
        CHECK(qt_run8_budget(dirs[d], 0, (int64_t)1<<40, cc, &r2) == 0);   /* R=tutti */
        CHECK(r1 == 0 && r0 == 2 && r2 == 2);
        for (int i = 0; i < 11; i++) CHECK(a[i]==b[i] && a[i]==cc[i]);     /* stream f32 == residente f32 */
    }
    return 0;
}

/* QBITS=8 + streaming: le matrici streamate girano int8 come le residenti;
 * la quantizzazione per riga rende i token IDENTICI a tutto-residente */
int qt_memknob_q8_parity(void) {
    char dense[512], hyb[512];
    qt_setup_both(dense, hyb);
    const char *dirs[2] = { dense, hyb };
    for (int d = 0; d < 2; d++) {
        int a[16], b[16]; int r0, r1;
        CHECK(qt_run8_budget(dirs[d], 8, 0, a, &r0) == 0);                 /* int8 residente */
        CHECK(qt_run8_budget(dirs[d], 8, 1, b, &r1) == 0);                 /* int8 streamato (R=0) */
        CHECK(r0 == 2 && r1 == 0);
        for (int i = 0; i < 11; i++) CHECK(a[i]==b[i]);
    }
    return 0;
}

int qt_memknob_env(void) {
    int64_t g8 = (int64_t)8<<30;
    CHECK(budget_from_env("2", "0.5", g8) == (int64_t)2<<30);   /* MEM_GB batte MEM_FRAC */
    CHECK(budget_from_env(NULL, "0.5", g8) == (int64_t)4<<30);
    CHECK(budget_from_env("", "", g8) == 0);
    CHECK(budget_from_env(NULL, NULL, g8) == 0);
    CHECK(budget_from_env("0", "1.5", g8) == 0);                /* valori invalidi -> residente */
    CHECK(budget_from_env("0.5", NULL, g8) == (int64_t)512<<20);
    return 0;
}

/* ---- micro-RSS: embed non residente + matmul streamato, parita' bit-esatta
 * col percorso f32 residente (stesse righe, stesso dot_f32) ---- */
static int qt_run8_micro(const char *dir, int *out) {
    g_micro = 1; g_micro_chunk = 256;   /* blocchi minuscoli: esercita il loop di chunking */
    Model m;
    model_init_ex(&m, dir, 0, 0, 16);
    CHECK(m.embed == NULL);
    CHECK(m.n_resident == 0);
    CHECK(m.stream_buf == NULL);
    CHECK(m.lm_tied && m.lm_head.sh != NULL && m.lm_head.f == NULL);
    kv_alloc(&m, 16);
    int rc = drive_greedy8(&m, out);
    g_micro = 0; g_mat_stream_fn = NULL; g_micro_chunk = 4<<20;
    return rc;
}

/* ---- QBITS=8 quantizza anche l'embed: loader a blocchi bit-identico alla
 * quantizzazione one-shot dell'intera tabella, tied lm_head sul kernel int8 ---- */
int qt_embed_q8(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    int save = g_embed_chunk_rows;
    g_embed_chunk_rows = 5;             /* vocab=32: 7 blocchi, resto compreso */
    Model m;
    model_init(&m, dir, 8);
    g_embed_chunk_rows = save;
    CHECK(m.embed == NULL && m.embed_q != NULL && m.embed_qs != NULL);
    CHECK(m.lm_tied && m.lm_head.q == m.embed_q && m.lm_head.qs == m.embed_qs && m.lm_head.f == NULL);
    int V = m.c.vocab, D = m.c.hidden;
    float *ref = falloc((int64_t)V*D);
    st_read_f32(&m.S, "model.embed_tokens.weight", ref, 0);
    int8_t *rq = malloc((size_t)V*D); float *rs = falloc(V);
    quantize_rows(ref, rq, rs, V, D, 8);
    CHECK(memcmp(rq, m.embed_q, (size_t)V*D) == 0);
    CHECK(memcmp(rs, m.embed_qs, (size_t)V*sizeof(float)) == 0);
    free(ref); free(rq); free(rs);
    /* generazione deterministica sul percorso embed int8 (gather = dequant) */
    int a[16], b[16];
    CHECK(qt_run8(dir, 8, 0, a) == 0);
    CHECK(qt_run8(dir, 8, 0, b) == 0);
    for (int i = 0; i < 11; i++) CHECK(a[i] == b[i]);
    return 0;
}

int qt_micro_parity(void) {
    char dense[512], hyb[512];
    qt_setup_both(dense, hyb);
    const char *dirs[2] = { dense, hyb };
    for (int d = 0; d < 2; d++) {
        int a[16], b[16];
        CHECK(qt_run8(dirs[d], 0, d, a) == 0);
        CHECK(qt_run8_micro(dirs[d], b) == 0);
        for (int i = 0; i < 11; i++) CHECK(a[i] == b[i]);
    }
    return 0;
}

/* ---- GGUF: stesso modello via safetensors e via GGUF -> logits bit-esatti ---- */
int qt_gguf_tiny_parity(void) {
    char dense[512]; snprintf(dense, sizeof(dense), "%s", tst_dir("qwen_tiny_model"));
    qt_write_dense_dir(dense);
    /* riscrivi i tensori del safetensors in un GGUF con i nomi llama.cpp:
     * dati bit-identici per costruzione */
    shards S; st_init(&S, dense);
    tg_reset();
    tg_kv_str("general.architecture", "qwen3");
    tg_kv_u32("qwen3.embedding_length", 16);
    tg_kv_u32("qwen3.block_count", 2);
    tg_kv_u32("qwen3.attention.head_count", 4);
    tg_kv_u32("qwen3.attention.head_count_kv", 2);
    tg_kv_u32("qwen3.attention.key_length", 8);
    tg_kv_u32("qwen3.feed_forward_length", 32);
    tg_kv_u32("qwen3.context_length", 64);
    tg_kv_f32("qwen3.rope.freq_base", 1000000.0f);
    tg_kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    tg_kv_u32("tokenizer.ggml.eos_token_id", 0);
    const char *toks[32]; char tbuf[32][8];              /* vocab_size = len(tokens) */
    for (int i = 0; i < 32; i++) { snprintf(tbuf[i], 8, "t%d", i); toks[i] = tbuf[i]; }
    tg_kv_arr_str("tokenizer.ggml.tokens", toks, 32);
    static const struct { const char *st, *gg; } map[] = {
        { "model.embed_tokens.weight", "token_embd.weight" },
        { "model.norm.weight",         "output_norm.weight" },
        { "model.layers.%d.input_layernorm.weight",          "blk.%d.attn_norm.weight" },
        { "model.layers.%d.post_attention_layernorm.weight", "blk.%d.ffn_norm.weight" },
        { "model.layers.%d.self_attn.q_norm.weight",         "blk.%d.attn_q_norm.weight" },
        { "model.layers.%d.self_attn.k_norm.weight",         "blk.%d.attn_k_norm.weight" },
        { "model.layers.%d.self_attn.q_proj.weight",         "blk.%d.attn_q.weight" },
        { "model.layers.%d.self_attn.k_proj.weight",         "blk.%d.attn_k.weight" },
        { "model.layers.%d.self_attn.v_proj.weight",         "blk.%d.attn_v.weight" },
        { "model.layers.%d.self_attn.o_proj.weight",         "blk.%d.attn_output.weight" },
        { "model.layers.%d.mlp.gate_proj.weight",            "blk.%d.ffn_gate.weight" },
        { "model.layers.%d.mlp.up_proj.weight",              "blk.%d.ffn_up.weight" },
        { "model.layers.%d.mlp.down_proj.weight",            "blk.%d.ffn_down.weight" },
    };
    static float tdata[32*16];
    for (size_t e = 0; e < sizeof map/sizeof map[0]; e++) {
        for (int li = 0; li < (e < 2 ? 1 : 2); li++) {
            char sn[128], gn[128];
            snprintf(sn, sizeof(sn), map[e].st, li);
            snprintf(gn, sizeof(gn), map[e].gg, li);
            int64_t n = st_numel(&S, sn);
            CHECK(n > 0 && n <= 32*16);
            st_read_f32(&S, sn, tdata, 0);
            tg_tensor_f32(gn, 1, n, tdata);              /* i motori validano solo il numel */
        }
    }
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    char gpath[600]; snprintf(gpath, sizeof(gpath), "%s/qwen_tiny.gguf", tmp);
    tg_write(gpath);
    /* run A: snapshot HF; run B: GGUF single-file */
    static const int prompt[3] = {1,2,3};
    Model a; model_init(&a, dense, 0);
    kv_alloc(&a, 16);
    float *la = step(&a, prompt, 3, 0);
    g_gguf = gpath;
    Model b; model_init(&b, NULL, 0);
    g_gguf = NULL;
    CHECK(b.c.hidden == 16 && b.c.n_layers == 2 && b.c.head_dim == 8 && b.c.vocab == 32);
    CHECK(b.lm_tied);                                    /* niente output.weight nel GGUF */
    kv_alloc(&b, 16);
    float *lb = step(&b, prompt, 3, 0);
    CHECK(memcmp(la, lb, a.c.vocab*sizeof(float)) == 0);
    free(la); free(lb);
    g_st_dequant_fn = NULL;
    remove(gpath);
    return 0;
}

/* ---- GGUF Q4_0: fast-path repack lossless in load_mat_bits + dequant f32 ---- */
int qt_gguf_q4_0_load(void) {
    enum { O = 4, I = 64, NB = I/32 };
    /* blocchi Q4_0 costruiti a mano: scala f16 nota + nibble pseudo-casuali */
    uint8_t blocks[O*NB*18];
    static const uint16_t ds[2] = { 0x3C00, 0x3800 };    /* 1.0, 0.5 */
    qt_rng_s = 4242;
    for (int b = 0; b < O*NB; b++) {
        memcpy(blocks + b*18, &ds[b & 1], 2);
        for (int i = 0; i < 16; i++) {
            qt_rng_s ^= qt_rng_s<<13; qt_rng_s ^= qt_rng_s>>7; qt_rng_s ^= qt_rng_s<<17;
            blocks[b*18 + 2 + i] = (uint8_t)(qt_rng_s & 0xFF);
        }
    }
    tg_reset();
    tg_kv_str("general.architecture", "qwen3");
    int64_t dims[2] = { I, O };
    tg_tensor("blk.0.ffn_gate.weight", 2 /* GGML Q4_0 */, 2, dims, blocks, sizeof(blocks));
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    char gpath[600]; snprintf(gpath, sizeof(gpath), "%s/qwen_q40.gguf", tmp);
    tg_write(gpath);
    Model m; memset(&m, 0, sizeof m);
    GgufMeta meta;
    gguf_index(&m.S, &meta, gpath);
    CHECK(st_dtype(&m.S, "model.layers.0.mlp.gate_proj.weight") == ST_Q4_0);
    /* QBITS=4: repack lossless, bit-identico al repack di riferimento */
    Mat w;
    load_mat_bits(&m, &w, "model.layers.0.mlp.gate_proj.weight", O, I, 4);
    CHECK(w.q4 != NULL && w.gs == 32 && w.f == NULL && w.q == NULL);
    uint8_t rq4[O*I/2]; float rqs[O*NB];
    gguf_repack_q4_0(blocks, rq4, rqs, O, I);
    CHECK(memcmp(w.q4, rq4, sizeof(rq4)) == 0);
    CHECK(memcmp(w.qs, rqs, sizeof(rqs)) == 0);
    /* QBITS=0: percorso dequant f32 via hook, uguale a gguf_dq_q4_0 */
    Mat wf;
    load_mat_bits(&m, &wf, "model.layers.0.mlp.gate_proj.weight", O, I, 0);
    float ref[O*I];
    gguf_dq_q4_0(blocks, O*NB, ref);
    CHECK(wf.f != NULL && memcmp(wf.f, ref, sizeof(ref)) == 0);
    g_st_dequant_fn = NULL;
    remove(gpath);
    return 0;
}

/* ---- KV_BITS=8: KV-cache int8 ---- */
int qt_kv_i8_roundtrip(void) {
    static const int ns[] = {4, 8, 64, 128};
    float src[128]; int8_t q[128]; float s;
    qt_rng_s = 999;
    for (size_t k = 0; k < sizeof ns/sizeof ns[0]; k++) {
        int n = ns[k];
        for (int rep = 0; rep < 8; rep++) {
            qt_fill(src, n, 3.f);
            kv_store_row(q, &s, src, n);
            for (int i = 0; i < n; i++) CHECK(fabsf(src[i] - s*(float)q[i]) <= s*0.5f + 1e-7f);
        }
    }
    return 0;
}

/* prefill {1,2,3} + decode sulla STESSA sequenza deterministica di token,
 * logits di ogni step catturati in out[steps][V] (ciclo condiviso col
 * gemello gemma) */
static int drive_capture(Model *m, float *out, int steps, int V) {
    int prompt[3] = {1,2,3};
    float *lo = step(m, prompt, 3, 0);
    memcpy(out, lo, (size_t)V*sizeof(float)); free(lo);
    int len = 3;
    for (int s = 1; s < steps; s++) {
        int t = (s % 5) + 1;
        lo = step(m, &t, 1, len); len++;
        memcpy(out + (int64_t)s*V, lo, (size_t)V*sizeof(float)); free(lo);
    }
    return 0;
}

/* confronto f32-KV vs int8-KV: deve restare entro tolleranza */
static int qt_kv8_drive(const char *dir, int kvbits, float *out, int steps, int V) {
    g_kv_bits = kvbits;
    Model m; model_init(&m, dir, 0);
    kv_alloc(&m, 16);
    if (kvbits == 8) {
        CHECK(m.K8[0] != NULL && m.Ks[0] != NULL && m.K[0] == NULL);
    } else {
        CHECK(m.K[0] != NULL && m.K8[0] == NULL);
    }
    int rc = drive_capture(&m, out, steps, V);
    g_kv_bits = 0;
    return rc;
}

int qt_kv_i8_tolerance(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    enum { STEPS = 6, V = 32 };
    static float a[STEPS*V], b[STEPS*V];
    CHECK(qt_kv8_drive(dir, 0, a, STEPS, V) == 0);
    CHECK(qt_kv8_drive(dir, 8, b, STEPS, V) == 0);
    for (int s = 0; s < STEPS; s++) {
        double d2 = 0, n2 = 0;
        for (int v = 0; v < V; v++) {
            CHECK(isfinite(b[s*V+v]));
            double d = (double)a[s*V+v] - b[s*V+v];
            d2 += d*d; n2 += (double)a[s*V+v]*a[s*V+v];
        }
        CHECK(sqrt(d2) <= 3e-2 * (sqrt(n2) + 1e-6));   /* errore SOLO dalla KV int8 */
    }
    return 0;
}

/* ibrido: i layer deltanet non hanno KV (K8 NULL), quelli full girano int8 */
int qt_kv_i8_hybrid(void) {
    const char *dir = tst_dir("qwen_tiny_hybrid");
    qt_write_hybrid_dir(dir);
    g_kv_bits = 8;
    Model m; model_init(&m, dir, 0);
    kv_alloc(&m, 16);
    CHECK(m.K8[0] == NULL && m.K[0] == NULL);          /* layer 0 = deltanet */
    CHECK(m.K8[1] != NULL && m.Vs[1] != NULL);         /* layer 1 = full int8 */
    int out[16];
    CHECK(drive_greedy8(&m, out) == 0);
    g_kv_bits = 0;
    /* le scale sono state scritte davvero */
    CHECK(m.Ks[1][0] > 0.f && m.Vs[1][0] > 0.f);
    return 0;
}

/* ---- PREFILL_CHUNK: prefill a blocchi bit-esatto al prefill intero ---- */
static int qt_prefill_chunk_dir(const char *dir, int hybrid) {
    static const int prompt[11] = {1,2,3,1,2,3,1,2,3,1,2};
    Model a; model_init(&a, dir, 0);
    CHECK(a.c.hybrid == hybrid);
    kv_alloc(&a, 16);
    float *la = step(&a, prompt, 11, 0);
    Model b; model_init(&b, dir, 0);
    kv_alloc(&b, 16);
    g_prefill_chunk = 3;                    /* 11 = 3+3+3+2: esercita blocchi pieni e coda */
    float *lb = step_chunked(&b, prompt, 11, 0);
    g_prefill_chunk = 0;
    CHECK(memcmp(la, lb, a.c.vocab*sizeof(float)) == 0);
    CHECK(a.kv_len == b.kv_len && b.kv_len == 11);
    /* il decode successivo parte dallo stesso stato: ancora bit-esatto */
    int t = argmax_v(la, a.c.vocab);
    float *da = step(&a, &t, 1, 11);
    float *db = step(&b, &t, 1, 11);
    CHECK(memcmp(da, db, a.c.vocab*sizeof(float)) == 0);
    free(la); free(lb); free(da); free(db);
    return 0;
}

int qt_prefill_chunk(void) {
    char dense[512], hyb[512];
    qt_setup_both(dense, hyb);
    CHECK(qt_prefill_chunk_dir(dense, 0) == 0);
    CHECK(qt_prefill_chunk_dir(hyb, 1) == 0);     /* deltanet: ricorrenza gia' per token */
    return 0;
}

/* ---- LoRA runtime: writer stw, loader, no-op e effetto ---- */

/* parita' stw <-> st: quello che stw_write scrive, st_init_file lo rilegge uguale */
int qt_stw_st_parity(void) {
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    char path[600]; snprintf(path, sizeof(path), "%s/qwen_stw_parity.safetensors", tmp);
    qt_rng_s = 77;
    float a[24], b[6], c[40];
    qt_fill(a, 24, 1); qt_fill(b, 6, 1); qt_fill(c, 40, 1);
    int64_t sa[2] = {4,6}, sb[1] = {6}, sc[3] = {2,4,5};
    stw_add("alpha", 2, sa, a);
    stw_add("beta",  1, sb, b);
    stw_add("gamma", 3, sc, c);
    stw_write(path);
    shards LS; st_init_file(&LS, path);
    CHECK(LS.n == 3);
    CHECK(st_numel(&LS,"alpha") == 24 && st_numel(&LS,"beta") == 6 && st_numel(&LS,"gamma") == 40);
    float ra[24], rb[6], rc[40];
    st_read_f32(&LS, "alpha", ra, 0);
    st_read_f32(&LS, "beta",  rb, 0);
    st_read_f32(&LS, "gamma", rc, 0);
    CHECK(!memcmp(a, ra, sizeof(a)) && !memcmp(b, rb, sizeof(b)) && !memcmp(c, rc, sizeof(c)));
    return 0;
}

/* aggiunge uno slot base.A/base.B al writer; B a zero se zeroB; A[0] in *a0 */
static void qt_lora_add(const char *base, int r, int I, int O, int zeroB, float *a0) {
    char nm[110];   /* < sizeof(StwT.name): niente warning di troncamento */
    float A[128], B[128];
    qt_fill(A, (int64_t)r*I, 0.5f);
    if (zeroB) memset(B, 0, sizeof(B)); else qt_fill(B, (int64_t)O*r, 0.5f);
    int64_t sA[2] = {r, I}, sB[2] = {O, r};
    snprintf(nm, sizeof(nm), "%s.A", base); stw_add(nm, 2, sA, A);
    snprintf(nm, sizeof(nm), "%s.B", base); stw_add(nm, 2, sB, B);
    if (a0) *a0 = A[0];
}

/* un passo greedy dal prompt {1,2,3}; logits (malloc'd) dell'ultimo token */
static float *qt_lora_step3(const char *dir, const char *lora_path) {
    Model m; model_init(&m, dir, 0);
    if (lora_path) {
        setenv("LORA", lora_path, 1);
        lora_load(&m);
        unsetenv("LORA");
    }
    kv_alloc(&m, 16);
    int ids[3] = {1,2,3};
    return step(&m, ids, 3, 0);
}

/* adattatore con B=0: i logits restano identici al modello base */
int qt_lora_zero_noop(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    char lp[600]; snprintf(lp, sizeof(lp), "%s/qwen_lora_zero.safetensors", tmp);
    qt_rng_s = 101;
    qt_lora_add("lora.layers.1.mlp.down_proj", 2, 32, 16, 1, NULL);
    float alpha = 4.f; int64_t s1[1] = {1};
    stw_add("lora.alpha", 1, s1, &alpha);
    stw_write(lp);
    float *base = qt_lora_step3(dir, NULL);
    float *with = qt_lora_step3(dir, lp);
    for (int v = 0; v < 32; v++) CHECK(base[v] == with[v]);
    free(base); free(with);
    return 0;
}

/* lora_apply vs riferimento double + effetto end-to-end con B != 0 */
int qt_lora_effect(void) {
    /* unit: y = y0 + (alpha/r)*B·(A·x) elemento per elemento in double */
    int I = 32, O = 16, r = 3, S = 2;
    float A[3*32], B[16*3], x[2*32], y[2*16], y0[2*16];
    qt_rng_s = 202;
    qt_fill(A, 96, 0.7f); qt_fill(B, 48, 0.7f); qt_fill(x, 64, 1.f); qt_fill(y0, 32, 1.f);
    memcpy(y, y0, sizeof(y));
    Lora lo = {A, B, r, 6.f};
    lora_apply(&lo, y, x, S, I, O);
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double acc = y0[s*O+o];
        for (int j = 0; j < r; j++) {
            double t = 0;
            for (int i = 0; i < I; i++) t += (double)A[j*I+i] * x[s*I+i];
            acc += (6.0/r) * (double)B[o*r+j] * t;
        }
        CHECK(fabs(acc - (double)y[s*O+o]) < 1e-5);
    }
    /* e2e: stesso slot di qt_lora_zero_noop ma B casuale -> logits diversi */
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    char lp[600]; snprintf(lp, sizeof(lp), "%s/qwen_lora_eff.safetensors", tmp);
    qt_rng_s = 101;
    qt_lora_add("lora.layers.1.mlp.down_proj", 2, 32, 16, 0, NULL);
    float alpha = 4.f; int64_t s1[1] = {1};
    stw_add("lora.alpha", 1, s1, &alpha);
    stw_write(lp);
    float *base = qt_lora_step3(dir, NULL);
    float *with = qt_lora_step3(dir, lp);
    int diff = 0;
    for (int v = 0; v < 32; v++) { CHECK(isfinite(with[v])); if (base[v] != with[v]) diff = 1; }
    CHECK(diff);
    free(base); free(with);
    return 0;
}

/* roundtrip completo: 7 slot x 2 layer + lm_head sopravvivono a stw+lora_load */
int qt_lora_roundtrip(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    char lp[600]; snprintf(lp, sizeof(lp), "%s/qwen_lora_rt.safetensors", tmp);
    static const char *subs[7] = {"self_attn.q_proj","self_attn.k_proj","self_attn.v_proj",
                                  "self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"};
    static const int Is[7] = {16,16,16,32,16,16,32};
    static const int Os[7] = {32,16,16,16,32,32,16};
    qt_rng_s = 303;
    float a0[15]; int na = 0;
    char base[96];
    for (int i = 0; i < 2; i++)
        for (int s = 0; s < 7; s++) {
            snprintf(base, sizeof(base), "lora.layers.%d.%s", i, subs[s]);
            qt_lora_add(base, 2, Is[s], Os[s], 0, &a0[na++]);
        }
    qt_lora_add("lora.lm_head", 2, 16, 32, 0, &a0[na]);
    float alpha = 5.f; int64_t s1[1] = {1};
    stw_add("lora.alpha", 1, s1, &alpha);
    stw_write(lp);
    Model m; model_init(&m, dir, 0);
    setenv("LORA", lp, 1);
    lora_load(&m);
    unsetenv("LORA");
    na = 0;
    for (int i = 0; i < 2; i++) {
        CHECK(m.L[i].lo);
        Lora *sl[7] = {&m.L[i].lo->q, &m.L[i].lo->k, &m.L[i].lo->v, &m.L[i].lo->o,
                       &m.L[i].lo->gate, &m.L[i].lo->up, &m.L[i].lo->down};
        for (int s = 0; s < 7; s++) {
            CHECK(sl[s]->r == 2 && sl[s]->alpha == 5.f);
            CHECK(sl[s]->A[0] == a0[na]); na++;
        }
    }
    CHECK(m.lm_lora.r == 2 && m.lm_lora.alpha == 5.f && m.lm_lora.A[0] == a0[na]);
    return 0;
}

/* ---- trainer: primitive backward e gradient-check ---- */

/* rope seguito da rope inverso = identita' (anche con rotazione parziale) */
int qt_bw_rope_inv(void) {
    float x[8], x0[8];
    qt_rng_s = 91;
    for (int rot = 8; rot >= 4; rot -= 4) {
        for (int i = 0; i < 8; i++) x[i] = x0[i] = qt_frnd();
        rope_head(x, 13, 1000000.f, rot);
        rope_head_inv(x, 13, 1000000.f, rot);
        for (int i = 0; i < 8; i++) CHECK(fabsf(x[i] - x0[i]) < 1e-6f);
    }
    return 0;
}

/* riferimento double di sum_j dy_j * rmsnorm_j(x)*w_j */
static double qt_rms_loss_ref(const double *x, const float *w, const float *dy, int D, double eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += x[i]*x[i];
    double r = 1.0 / sqrt(ms/D + eps);
    double L = 0; for (int i = 0; i < D; i++) L += (double)dy[i] * x[i]*r*(double)w[i];
    return L;
}

/* bw_rmsnorm contro finite-difference in double */
int qt_bw_rmsnorm(void) {
    int D = 16;
    float x[16], w[16], dy[16], y[16], dx[16], r;
    qt_rng_s = 92;
    qt_fill(x, D, 1); qt_fill(w, D, 1); qt_fill(dy, D, 1);
    t_rmsnorm_row(y, x, w, D, 1e-6f, &r);
    memset(dx, 0, sizeof(dx));
    bw_rmsnorm(dx, dy, x, w, r, D);
    double xd[16]; for (int i = 0; i < D; i++) xd[i] = x[i];
    for (int k = 0; k < 10; k++) {
        int i = (k * 3) % D;
        double eps = 1e-5, s = xd[i];
        xd[i] = s + eps; double lp = qt_rms_loss_ref(xd, w, dy, D, 1e-6);
        xd[i] = s - eps; double lm = qt_rms_loss_ref(xd, w, dy, D, 1e-6);
        xd[i] = s;
        double gfd = (lp - lm) / (2*eps);
        double rel = fabs(gfd - (double)dx[i]) / fmax(1e-3, fabs(gfd) + fabs(dx[i]));
        CHECK(rel < 1e-3);
    }
    return 0;
}

/* setup comune del trainer sui due layer del modello tiny */
static Model qt_train_m;
static LStash qt_train_st[2];
static LoraLayer *qt_train_gl;
static Lora qt_train_gh;

static void qt_train_setup(int S, int head, int maxt) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    model_init(&qt_train_m, dir, 0);
    kv_alloc(&qt_train_m, maxt);
    train_guard(&qt_train_m, 0);
    lora_train_alloc(&qt_train_m, 0, 2, 4.f, head);
    memset(&qt_train_gh, 0, sizeof(qt_train_gh));
    qt_train_gl = lora_grad_alloc(&qt_train_m, 0, head ? &qt_train_gh : NULL);
    t_params_build(&qt_train_m, 0, qt_train_gl, head ? &qt_train_gh : NULL);
    lstash_alloc(&qt_train_st[0], &qt_train_m.c, S);
    lstash_alloc(&qt_train_st[1], &qt_train_m.c, S);
}

/* IL gate del trainer: ogni parametro adattatore contro central finite
 * difference. B randomizzati (con B=0 i gradienti su A sarebbero nulli). */
int qt_grad_fd(void) {
    qt_train_setup(5, 1, 8);
    Model *m = &qt_train_m;
    qt_rng_s = 505;
    for (int p = 0; p < t_npar; p++)
        for (int64_t i = 0; i < t_par[p].n; i++) t_par[p].w[i] += qt_frnd() * 0.3f;
    int ids[5];
    qt_rng_s = 606;
    for (int i = 0; i < 5; i++) ids[i] = 1 + (int)((qt_frnd() + 0.5f) * 30.99f);
    t_grads_zero();
    train_loss_and_backward(m, ids, 5, 0, qt_train_st, qt_train_gl, &qt_train_gh);
    double maxrel = 0; int nch = 0;
    /* eps 1e-2 (non 1e-3): il forward e' in float, la loss ha ~1e-6 di rumore
     * e con eps piu' piccolo il rumore/(2*eps) supera la tolleranza sui
     * gradienti piccoli. L'errore di troncamento O(eps^2) resta trascurabile. */
    const float eps = 1e-2f;
    for (int p = 0; p < t_npar; p++) {
        for (int64_t i = 0; i < t_par[p].n; i++) {
            float w0 = t_par[p].w[i];
            t_par[p].w[i] = w0 + eps;
            double lp = train_loss_and_backward(m, ids, 5, 0, qt_train_st, NULL, NULL);
            t_par[p].w[i] = w0 - eps;
            double lm = train_loss_and_backward(m, ids, 5, 0, qt_train_st, NULL, NULL);
            t_par[p].w[i] = w0;
            double gfd = (lp - lm) / (2.0 * eps);
            double gan = t_par[p].g[i];
            double rel = fabs(gfd - gan) / fmax(1e-3, fabs(gfd) + fabs(gan));
            if (rel > maxrel) maxrel = rel;
            nch++;
            /* doppio criterio: errore relativo E assoluto. Il pavimento
             * assoluto 5e-4 (~10x il rumore FD) evita falsi negativi sui
             * gradienti minuscoli; un errore di formula vero sposta interi
             * tensori ben oltre entrambe le soglie. */
            if (rel >= 1e-2 && fabs(gfd - gan) >= 5e-4) {
                fprintf(stderr, "grad-fd: par %d el %lld: an=%.6g fd=%.6g rel=%.3g\n",
                        p, (long long)i, gan, gfd, rel);
                return 1;
            }
        }
    }
    fprintf(stderr, "grad-fd: %d parametri verificati, max rel err %.3g\n", nch, maxrel);
    return 0;
}

/* 20 step AdamW su una finestra: la loss deve scendere nettamente */
int qt_train_descends(void) {
    qt_train_setup(16, 0, 24);
    Model *m = &qt_train_m;
    int ids[16];
    for (int i = 0; i < 16; i++) ids[i] = 1 + (i % 7);
    double first = 0, last = 0;
    for (int it = 1; it <= 20; it++) {
        t_grads_zero();
        double L = train_loss_and_backward(m, ids, 16, 0, qt_train_st, qt_train_gl, NULL);
        if (it == 1) first = L;
        last = L;
        t_adamw_all(1e-2f, 0.f, it);
    }
    fprintf(stderr, "train-descends: loss %.4f -> %.4f\n", first, last);
    CHECK(last < first * 0.9 && last < first);
    return 0;
}

/* ---- end-to-end: loop di training su corpus sintetico + adattatori riusabili ----
 * Guida direttamente il loop a finestre (niente train_main: nessun fixture
 * tokenizer), poi salva con lora_save e ricarica in un modello fresco. */
int qt_train_e2e(void) {
    qt_train_setup(16, 0, 24);
    Model *m = &qt_train_m;
    int ids[200];
    for (int i = 0; i < 200; i++) ids[i] = (i % 7) + 1;
    double first = 0, lastm = 0;
    int t = 0;
    for (int ep = 0; ep < 3; ep++) {
        double sum = 0; int nw = 0;
        for (int off = 0; off + 8 <= 200; off += 16) {
            int S = 200 - off < 16 ? 200 - off : 16;
            m->kv_len = 0;
            t_grads_zero();
            sum += train_loss_and_backward(m, ids + off, S, 0, qt_train_st, qt_train_gl, NULL);
            t_adamw_all(1e-2f, 0.f, ++t);
            nw++;
        }
        if (ep == 0) first = sum / nw;
        lastm = sum / nw;
    }
    fprintf(stderr, "train-e2e: loss media epoca %.4f -> %.4f\n", first, lastm);
    CHECK(lastm < first);
    /* salvataggio con lo stesso percorso del trainer, poi ricarica da zero */
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    char lp[600]; snprintf(lp, sizeof(lp), "%s/qwen_lora_e2e.safetensors", tmp);
    lora_save(m, 0, 4.f, lp);
    const char *dir = tst_dir("qwen_tiny_model");
    float *base = qt_lora_step3(dir, NULL);
    float *with = qt_lora_step3(dir, lp);
    int diff = 0;
    for (int v = 0; v < 32; v++) { CHECK(isfinite(with[v])); if (base[v] != with[v]) diff = 1; }
    CHECK(diff);
    free(base); free(with);
    return 0;
}

/* ---- TTA sperimentale: cache neurale e bias sui logit ---- */

/* protocollo di gen_turn guidato a mano: step -> adjust -> (prob del vero
 * successivo) -> observe. Ritorna la somma di -log p(next) su seq[1..n). */
static double tta_drive(Model *m, const int *seq, int n, double *last_p_next) {
    kv_alloc(m, 32);
    double nll = 0;
    float *logit = NULL;
    for (int t = 0; t < n - 1; t++) {
        logit = step(m, &seq[t], 1, t);
        tta_adjust(m, logit);
        float *p = falloc(m->c.vocab);
        memcpy(p, logit, m->c.vocab*sizeof(float));
        softmax_row(p, m->c.vocab);
        nll += -log((double)p[seq[t+1]] + 1e-30);
        if (last_p_next) *last_p_next = p[seq[t+1]];
        free(p);
        tta_observe(m, seq[t+1]);
        free(logit);
    }
    return nll;
}

static void tta_setup(int mode, float lambda, float lr) {
    tta_reset();
    g_tta.init = 1;                 /* niente parsing env nei test */
    g_tta.mode = mode;
    g_tta.n = 64; g_tta.lambda = lambda; g_tta.theta = 4.0f; g_tta.lr = lr;
    g_tta.alloc = 0;                /* ri-alloca alle dimensioni del modello corrente */
}

int qt_tta_off_bitexact(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    static const int seq[8] = {1,2,3,1,2,3,1,2};
    Model m1; model_init(&m1, dir, 0);
    tta_setup(TTA_OFF, 0, 0);
    double a = tta_drive(&m1, seq, 8, NULL);
    /* cache con lambda=0 deve essere identico a spento */
    Model m2; model_init(&m2, dir, 0);
    tta_setup(TTA_CACHE, 0.0f, 0);
    double b = tta_drive(&m2, seq, 8, NULL);
    CHECK(a == b);
    return 0;
}

int qt_tta_cache_boost(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    /* sequenza ripetitiva: il cache deve alzare P del prossimo token ripetuto */
    static const int seq[13] = {4,5,6,4,5,6,4,5,6,4,5,6,4};
    double p_off, p_on;
    Model m1; model_init(&m1, dir, 0);
    tta_setup(TTA_OFF, 0, 0);
    tta_drive(&m1, seq, 13, &p_off);
    Model m2; model_init(&m2, dir, 0);
    tta_setup(TTA_CACHE, 0.3f, 0);
    tta_drive(&m2, seq, 13, &p_on);
    fprintf(stderr, "tta cache: P(next) off=%.4f on=%.4f\n", p_off, p_on);
    CHECK(p_on > p_off);
    return 0;
}

int qt_tta_bias_direction(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    /* osserva sempre il token 7: il bias deve crescere su 7 */
    static const int seq[10] = {7,7,7,7,7,7,7,7,7,7};
    Model m; model_init(&m, dir, 0);
    tta_setup(TTA_BIAS, 0, 0.5f);
    tta_drive(&m, seq, 10, NULL);
    CHECK(g_tta.alloc && g_tta.bias[7] > 0.f);
    for (int v = 0; v < m.c.vocab; v++)
        if (v != 7) CHECK(g_tta.bias[7] > g_tta.bias[v]);
    /* reset: il bias torna a zero */
    state_reset(&m);
    CHECK(g_tta.bias[7] == 0.f);
    return 0;
}

/* ---- TTA=lora: adattatore online low-rank sull'lm_head ---- */

/* protocollo a due step: prefill {1,2,3} + un decode; logits del decode */
static float *qt_tta_two_steps(Model *m) {
    kv_alloc(m, 16);
    int ids[3] = {1,2,3};
    float *l1 = step(m, ids, 3, 0);
    tta_adjust(m, l1);
    free(l1);
    int nxt = 4;
    float *l2 = step(m, &nxt, 1, 3);
    tta_adjust(m, l2);
    return l2;
}

/* B=0 (fresco o dopo reset): adjust aggiunge zeri esatti, logits identici */
int qt_tta_lora_off_noop(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    Model m1; model_init(&m1, dir, 0);
    tta_setup(TTA_OFF, 0, 0);
    float *a = qt_tta_two_steps(&m1);
    Model m2; model_init(&m2, dir, 0);
    tta_setup(TTA_LORA, 0, 1e-3f);
    float *b = qt_tta_two_steps(&m2);
    CHECK(g_tta.alloc && g_tta.lB && g_tta.h_valid);   /* l'adattatore era attivo */
    for (int v = 0; v < 32; v++) CHECK(a[v] == b[v]);
    free(a); free(b);
    tta_setup(TTA_OFF, 0, 0);
    return 0;
}

/* osserva sempre il token 7: la P aggiustata di 7 deve crescere vs base */
int qt_tta_lora_direction(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    static const int seq[12] = {7,7,7,7,7,7,7,7,7,7,7,7};
    double p_off, p_on;
    Model m1; model_init(&m1, dir, 0);
    tta_setup(TTA_OFF, 0, 0);
    tta_drive(&m1, seq, 12, &p_off);
    Model m2; model_init(&m2, dir, 0);
    tta_setup(TTA_LORA, 0, 0.1f);      /* lr alto per un effetto netto sul tiny */
    tta_drive(&m2, seq, 12, &p_on);
    fprintf(stderr, "tta lora: P(next) off=%.4f on=%.4f\n", p_off, p_on);
    CHECK(p_on > p_off);
    tta_setup(TTA_OFF, 0, 0);
    return 0;
}

/* state_reset azzera B: l'output aggiustato torna identico al base */
int qt_tta_lora_reset(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    Model m1; model_init(&m1, dir, 0);
    tta_setup(TTA_OFF, 0, 0);
    float *a = qt_tta_two_steps(&m1);
    Model m2; model_init(&m2, dir, 0);
    tta_setup(TTA_LORA, 0, 0.1f);
    static const int seq[10] = {7,7,7,7,7,7,7,7,7,7};
    tta_drive(&m2, seq, 10, NULL);     /* addestra l'adattatore online */
    double bnorm = 0;
    for (int64_t i = 0; i < (int64_t)32*g_tta.lr_rank; i++) bnorm += fabs(g_tta.lB[i]);
    CHECK(bnorm > 0);                  /* B si e' mosso davvero */
    state_reset(&m2);                  /* B azzerato -> adattatore no-op */
    m2.kv_len = 0;
    float *b = qt_tta_two_steps(&m2);
    for (int v = 0; v < 32; v++) CHECK(a[v] == b[v]);
    free(a); free(b);
    tta_setup(TTA_OFF, 0, 0);
    return 0;
}

int qt_tta_ppl_proxy(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    /* su testo ripetitivo la NLL cumulata deve strettamente migliorare */
    static int seq[25];
    for (int i = 0; i < 25; i++) seq[i] = 8 + (i % 3);
    Model m1; model_init(&m1, dir, 0);
    tta_setup(TTA_OFF, 0, 0);
    double nll_off = tta_drive(&m1, seq, 25, NULL);
    Model m2; model_init(&m2, dir, 0);
    tta_setup(TTA_CACHE, 0.3f, 0);
    double nll_on = tta_drive(&m2, seq, 25, NULL);
    fprintf(stderr, "tta ppl-proxy: nll off=%.3f on=%.3f\n", nll_off, nll_on);
    CHECK(nll_on < nll_off);
    tta_setup(TTA_OFF, 0, 0);       /* non inquinare gli altri test */
    return 0;
}

/* ---- PREFILL_CHUNK + TTA: lo stash h_cur e' quello dell'ULTIMO token del
 * prompt, identico al prefill non a blocchi ---- */
int qt_prefill_chunk_tta(void) {
    const char *dir = tst_dir("qwen_tiny_model");
    qt_write_dense_dir(dir);
    static const int prompt[11] = {1,2,3,1,2,3,1,2,3,1,2};
    float ha[16];
    Model a; model_init(&a, dir, 0);
    tta_setup(TTA_CACHE, 0.1f, 0);
    tta_ensure(&a);
    kv_alloc(&a, 16);
    float *la = step(&a, prompt, 11, 0);
    CHECK(g_tta.h_valid);
    memcpy(ha, g_tta.h_cur, a.c.hidden*sizeof(float));
    Model b; model_init(&b, dir, 0);
    tta_setup(TTA_CACHE, 0.1f, 0);
    tta_ensure(&b);
    kv_alloc(&b, 16);
    g_prefill_chunk = 4;
    float *lb = step_chunked(&b, prompt, 11, 0);
    g_prefill_chunk = 0;
    CHECK(g_tta.h_valid);
    CHECK(memcmp(ha, g_tta.h_cur, b.c.hidden*sizeof(float)) == 0);
    CHECK(memcmp(la, lb, a.c.vocab*sizeof(float)) == 0);
    free(la); free(lb);
    tta_setup(TTA_OFF, 0, 0);       /* non inquinare gli altri test */
    return 0;
}
