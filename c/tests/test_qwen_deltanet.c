/* Validazione del blocco Gated DeltaNet contro il riferimento Python
 * (deltanet_fixture.h, generato da gen_deltanet_fixture.py con le formule
 * di transformers qwen3_5 in double precision), piu' edge-case e il layout
 * del q_proj gated dell'attenzione. */
#define QWEN_TEST
#include "../qwen.c"
#include "deltanet_fixture.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static Mat mk_mat(const float *w, int O, int I) {
    Mat m; m.q=NULL; m.qs=NULL; m.O=O; m.I=I;
    m.f = falloc((int64_t)O*I);
    memcpy(m.f, w, (int64_t)O*I*sizeof(float));
    return m;
}

/* ---- ricorrenza vs fixture Python ---- */
static int test_fixture(void) {
    int kd = FX_HK*FX_DK, vd = FX_HV*FX_DV, cd = 2*kd + vd;
    Model m; memset(&m,0,sizeof m);
    m.c.hidden = FX_D; m.c.eps = 1e-6f;
    m.c.lin_hv = FX_HV; m.c.lin_hk = FX_HK; m.c.lin_dk = FX_DK; m.c.lin_dv = FX_DV; m.c.lin_conv = FX_K;
    Layer l; memset(&l,0,sizeof l);
    l.type = 1;
    l.aqkv = mk_mat(FX_W_QKV, cd, FX_D);
    l.az   = mk_mat(FX_W_Z,   vd, FX_D);
    l.ab   = mk_mat(FX_W_B,   FX_HV, FX_D);
    l.aa   = mk_mat(FX_W_A,   FX_HV, FX_D);
    l.aout = mk_mat(FX_W_OUT, FX_D, vd);
    l.conv_w = falloc((int64_t)cd*FX_K); memcpy(l.conv_w, FX_CONV_W, sizeof(float)*cd*FX_K);
    l.conv_b = falloc(cd); memcpy(l.conv_b, FX_CONV_B, sizeof(float)*cd);
    l.dt_bias = falloc(FX_HV); memcpy(l.dt_bias, FX_DT_BIAS, sizeof(float)*FX_HV);
    l.A_log   = falloc(FX_HV); memcpy(l.A_log, FX_A_LOG, sizeof(float)*FX_HV);
    l.dn_norm = falloc(FX_DV); memcpy(l.dn_norm, FX_NORM_W, sizeof(float)*FX_DV);
    l.conv_state = falloc((int64_t)cd*FX_K); memset(l.conv_state, 0, sizeof(float)*cd*FX_K);
    l.Sstate = falloc((int64_t)FX_HV*FX_DK*FX_DV); memset(l.Sstate, 0, sizeof(float)*FX_HV*FX_DK*FX_DV);
    float out[FX_D];
    for (int t = 0; t < FX_T; t++) {
        deltanet_token(&m, &l, FX_X + t*FX_D, out);
        for (int d = 0; d < FX_D; d++) {
            float exp_ = FX_EXPECT[t*FX_D + d];
            if (fabsf(out[d]-exp_) > 1e-4f) {
                fprintf(stderr, "token %d dim %d: got %.6f expected %.6f\n", t, d, out[d], exp_);
                return 1;
            }
        }
    }
    fprintf(stderr, "deltanet vs riferimento python: ok\n");
    return 0;
}

/* ---- edge-case numerici ---- */
static int test_edges(void) {
    CHECK(fabsf(softplusf(50.f) - 50.f) < 1e-4f);
    CHECK(fabsf(softplusf(0.f) - logf(2.f)) < 1e-6f);
    CHECK(softplusf(-50.f) >= 0.f && softplusf(-50.f) < 1e-6f);
    CHECK(fabsf(sigmoidf(0.f) - 0.5f) < 1e-7f);
    float z[4] = {0,0,0,0};
    l2norm_head(z, 4);                       /* vettore nullo: resta finito (eps) */
    for (int i = 0; i < 4; i++) CHECK(isfinite(z[i]) && z[i]==0.f);
    fprintf(stderr, "edge-case: ok\n");
    return 0;
}

/* ---- layout q_proj gated: gate >> 0 (sigmoid ~ 1) == attenzione non gated ---- */
static uint64_t rng = 99;
static float frnd(void) { rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17;
    return (float)((rng>>11)*(1.0/9007199254740992.0)) - 0.5f; }

static int test_gated_layout(void) {
    int D = 16, H = 2, KV = 2, hd = 8, S = 4;
    Model A, B; memset(&A,0,sizeof A); memset(&B,0,sizeof B);
    A.c.hidden=D; A.c.n_heads=H; A.c.n_kv_heads=KV; A.c.head_dim=hd; A.c.rot=hd/2;  /* esercita anche il partial rope */
    A.c.theta=1e6f; A.c.eps=1e-6f; A.c.n_layers=1;
    static int lt[1] = {0}; A.c.ltype = lt;
    B.c = A.c;
    Layer la, lb; memset(&la,0,sizeof la); memset(&lb,0,sizeof lb);
    /* base non gated */
    la.q.f=falloc((int64_t)H*hd*D);  la.q.O=H*hd;  la.q.I=D;
    la.k.f=falloc((int64_t)KV*hd*D); la.k.O=KV*hd; la.k.I=D;
    la.v.f=falloc((int64_t)KV*hd*D); la.v.O=KV*hd; la.v.I=D;
    la.o.f=falloc((int64_t)D*H*hd);  la.o.O=D;     la.o.I=H*hd;
    la.qn=falloc(hd); la.kn=falloc(hd);
    for (int64_t i=0;i<(int64_t)H*hd*D;i++) la.q.f[i]=frnd();
    for (int64_t i=0;i<(int64_t)KV*hd*D;i++){ la.k.f[i]=frnd(); la.v.f[i]=frnd(); }
    for (int64_t i=0;i<(int64_t)D*H*hd;i++) la.o.f[i]=frnd();
    for (int i=0;i<hd;i++){ la.qn[i]=1; la.kn[i]=1; }
    /* gated: righe query = base, righe gate = bias enorme via peso costante
     * (x non e' garantito positivo, quindi usiamo pesi 0 e... il gate e' W@x:
     * con W=0 il gate e' 0 -> sigmoid=0.5. Quindi confrontiamo con il contesto
     * scalato 0.5: piu' robusto di un "gate spalancato"). */
    lb = la;
    lb.gated = 1;
    lb.q.f=falloc((int64_t)2*H*hd*D); lb.q.O=2*H*hd; lb.q.I=D;
    for (int hh = 0; hh < H; hh++) {
        memcpy(lb.q.f + (int64_t)hh*2*hd*D,        la.q.f + (int64_t)hh*hd*D, (int64_t)hd*D*sizeof(float));
        memset(lb.q.f + (int64_t)hh*2*hd*D + (int64_t)hd*D, 0, (int64_t)hd*D*sizeof(float));
    }
    kv_alloc(&A, 8); kv_alloc(&B, 8);
    float *x = falloc((int64_t)S*D); for (int64_t i=0;i<(int64_t)S*D;i++) x[i]=frnd();
    float *oa = falloc((int64_t)S*D), *ob = falloc((int64_t)S*D);
    attention(&A, &la, 0, x, S, 0, oa);
    attention(&B, &lb, 0, x, S, 0, ob);
    /* gate=0 -> sigmoid=0.5: l'output gated deve essere l'output base con il
     * contesto dimezzato, e o_proj e' lineare -> ob == 0.5*oa */
    for (int64_t i = 0; i < (int64_t)S*D; i++) CHECK(fabsf(ob[i] - 0.5f*oa[i]) < 1e-5f);
    fprintf(stderr, "gated attention (layout q|gate, partial rope): ok\n");
    return 0;
}

int main(void) {
    if (test_fixture()) return 1;
    if (test_edges()) return 1;
    if (test_gated_layout()) return 1;
    printf("test_qwen_deltanet: ok\n");
    return 0;
}
