/* Unit test del motore qwen: RoPE, attenzione GQA, sampler.
 * Include qwen.c con QWEN_TEST (esclude il main). */
#define QWEN_TEST
#include "../qwen.c"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static uint64_t rng = 42;
static float frnd(void) {   /* uniforme [-0.5,0.5), deterministica */
    rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17;
    return (float)((rng>>11)*(1.0/9007199254740992.0)) - 0.5f;
}

/* ---- RoPE: pos=0 identita', norma preservata, valori vs riferimento double ---- */
static int test_rope(void) {
    int hd = 8; float theta = 1000000.f;
    float x[8], x0[8];
    for (int i = 0; i < hd; i++) x[i] = x0[i] = frnd();
    rope_head(x, 0, theta, hd);
    for (int i = 0; i < hd; i++) CHECK(fabsf(x[i]-x0[i]) < 1e-6f);
    /* pos 17: confronto con la rotazione calcolata in double */
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
    fprintf(stderr, "rope: ok\n");
    return 0;
}

/* ---- GQA: attenzione con KV<H deve coincidere con la MHA a teste kv replicate ---- */
static void fill(float *p, int64_t n, float sc) { for (int64_t i = 0; i < n; i++) p[i] = frnd()*sc; }

static int test_gqa(void) {
    int D = 16, H = 4, KV = 2, hd = 8, G = H/KV, S = 5;
    Model A, B; memset(&A,0,sizeof A); memset(&B,0,sizeof B);
    A.c.hidden=D; A.c.n_heads=H; A.c.n_kv_heads=KV; A.c.head_dim=hd; A.c.rot=hd;
    A.c.theta=1e6f; A.c.eps=1e-6f; A.c.n_layers=1;
    static int lt[1] = {0}; A.c.ltype = lt;
    B.c = A.c; B.c.n_kv_heads = H;
    Layer la, lb; memset(&la,0,sizeof la); memset(&lb,0,sizeof lb);
    la.q.f=falloc((int64_t)H*hd*D);  la.q.O=H*hd;  la.q.I=D;
    la.k.f=falloc((int64_t)KV*hd*D); la.k.O=KV*hd; la.k.I=D;
    la.v.f=falloc((int64_t)KV*hd*D); la.v.O=KV*hd; la.v.I=D;
    la.o.f=falloc((int64_t)D*H*hd);  la.o.O=D;     la.o.I=H*hd;
    la.qn=falloc(hd); la.kn=falloc(hd);
    fill(la.q.f,(int64_t)H*hd*D,1); fill(la.k.f,(int64_t)KV*hd*D,1); fill(la.v.f,(int64_t)KV*hd*D,1);
    fill(la.o.f,(int64_t)D*H*hd,1); fill(la.qn,hd,1); fill(la.kn,hd,1);
    /* B: k/v con le teste kv replicate G volte, stesso q/o/norme */
    lb = la;
    lb.k.f=falloc((int64_t)H*hd*D); lb.k.O=H*hd; lb.k.I=D;
    lb.v.f=falloc((int64_t)H*hd*D); lb.v.O=H*hd; lb.v.I=D;
    for (int hh = 0; hh < H; hh++) {
        memcpy(lb.k.f + (int64_t)hh*hd*D, la.k.f + (int64_t)(hh/G)*hd*D, (int64_t)hd*D*sizeof(float));
        memcpy(lb.v.f + (int64_t)hh*hd*D, la.v.f + (int64_t)(hh/G)*hd*D, (int64_t)hd*D*sizeof(float));
    }
    kv_alloc(&A, 16); kv_alloc(&B, 16);
    float *x = falloc((int64_t)S*D); fill(x,(int64_t)S*D,1);
    float *oa = falloc((int64_t)S*D), *ob = falloc((int64_t)S*D);
    attention(&A, &la, 0, x, S, 0, oa);
    attention(&B, &lb, 0, x, S, 0, ob);
    for (int64_t i = 0; i < (int64_t)S*D; i++) CHECK(fabsf(oa[i]-ob[i]) < 1e-5f);
    /* decode incrementale: un token in coda, stesso risultato */
    float *xt = falloc(D); fill(xt,D,1);
    float *oa1 = falloc(D), *ob1 = falloc(D);
    attention(&A, &la, 0, xt, 1, S, oa1);
    attention(&B, &lb, 0, xt, 1, S, ob1);
    for (int i = 0; i < D; i++) CHECK(fabsf(oa1[i]-ob1[i]) < 1e-5f);
    fprintf(stderr, "gqa vs mha replicata: ok\n");
    return 0;
}

/* ---- quantizzazione: matmul_q entro tolleranza dal f32 ---- */
static int test_quant(void) {
    int O = 24, I = 32;
    float *w = falloc((int64_t)O*I), *x = falloc(I), *y = falloc(O), *yq = falloc(O);
    fill(w,(int64_t)O*I,1); fill(x,I,1);
    int8_t *q = malloc((int64_t)O*I); float *qs = falloc(O);
    quantize_rows(w, q, qs, O, I, 8);
    matmul(y, x, w, 1, I, O);
    matmul_q(yq, x, q, qs, I, O);
    for (int o = 0; o < O; o++) CHECK(fabsf(y[o]-yq[o]) < 0.05f);
    fprintf(stderr, "quant int8: ok\n");
    return 0;
}

/* ---- sampler: stesso seed -> stessa sequenza; greedy = argmax ---- */
static int test_sampler(void) {
    int V = 100; float lo[100];
    for (int i = 0; i < V; i++) lo[i] = frnd()*4;
    g_temp = 0; CHECK(pick_tok(lo,V) == argmax_v(lo,V));
    g_temp = 0.8f; g_nuc = 0.9f;
    int a[16], b[16];
    g_rng = 12345; for (int i = 0; i < 16; i++) a[i] = pick_tok(lo,V);
    g_rng = 12345; for (int i = 0; i < 16; i++) b[i] = pick_tok(lo,V);
    for (int i = 0; i < 16; i++) CHECK(a[i]==b[i]);
    fprintf(stderr, "sampler: ok\n");
    return 0;
}

int main(void) {
    if (test_rope()) return 1;
    if (test_gqa()) return 1;
    if (test_quant()) return 1;
    if (test_sampler()) return 1;
    printf("test_qwen_units: ok\n");
    return 0;
}
