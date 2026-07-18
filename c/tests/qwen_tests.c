/* Logica dei test del motore qwen (C puro; glue gtest in qwen_gtest.cc).
 * Ogni funzione qt_* ritorna 0=ok, 1=fail; dettagli su stderr.
 * Include il motore intero: accesso diretto a static e strutture. */
#define QWEN_TEST
#include "../qwen.c"
#include "tiny_st.h"

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

/* greedy 8 token dal prompt {1,2,3}; out deve avere spazio per 11.
 * Ritorna 0/1; verifica strutture (tied, head_dim, ibrido). */
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
static int qt_run8_budget(const char *dir, int64_t budget, int *out, int *resident_out) {
    Model m;
    model_init_ex(&m, dir, 0, budget, 16);
    if (resident_out) *resident_out = m.n_resident;
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

int qt_memknob_parity(void) {
    /* stessi id greedy con: tutto residente (budget 0), tutto streamato
     * (budget minuscolo -> R=0) e budget enorme (R=n_layers) — sia sul
     * modello denso che sull'ibrido (stati deltanet sempre residenti) */
    const char *dirs[2] = { tst_dir("qwen_tiny_model"), tst_dir("qwen_tiny_hybrid") };
    qt_write_dense_dir(dirs[0]);
    qt_write_hybrid_dir(dirs[1]);
    for (int d = 0; d < 2; d++) {
        int a[16], b[16], cc[16]; int r0, r1, r2;
        CHECK(qt_run8_budget(dirs[d], 0, a, &r0) == 0);                    /* classico */
        CHECK(qt_run8_budget(dirs[d], 1, b, &r1) == 0);                    /* R=0: tutto stream */
        CHECK(qt_run8_budget(dirs[d], (int64_t)1<<40, cc, &r2) == 0);      /* R=tutti */
        CHECK(r1 == 0 && r0 == 2 && r2 == 2);
        for (int i = 0; i < 11; i++) CHECK(a[i]==b[i] && a[i]==cc[i]);     /* stream f32 == residente f32 */
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
