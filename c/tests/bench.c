/* Micro-bench dei percorsi caldi (utility C pura, NON un test ctest):
 *   (a) matmul f32   O=4096 I=4096 S=1   (GEMV denso: lm_head, proiezioni)
 *   (b) matmul_q int8 stessa forma       (schema Q8_0 + dot_i8i8)
 *   (c) dot_f32 forma attention          (4096 dot di lunghezza 128)
 *   (d) deltanet_token dimensioni Qwen3.5-4B (D=2560, Hv=32, Hk=16, dk=dv=128, K=4)
 * Stampa GFLOP/s e GB/s (pesi) + i tier compilati. Compilare anche con
 * -DCOLIBRI_TEST_MARCH/-march per misurare i kernel reali. */
#define QWEN_TEST
#include "../qwen.c"

static uint64_t bn_rng = 88172645463325252ULL;
static float bn_frnd(void) {
    bn_rng ^= bn_rng<<13; bn_rng ^= bn_rng>>7; bn_rng ^= bn_rng<<17;
    return (float)((bn_rng>>11)*(1.0/9007199254740992.0)) - 0.5f;
}
static void bn_fill(float *p, int64_t n) { for (int64_t i = 0; i < n; i++) p[i] = bn_frnd(); }

int main(int argc, char **argv) {
    (void)argc;
    omp_hot_tune(argv);                             /* stesso tuning dei motori: numeri realistici */
    const char *th_ = getenv("THREADS");
    if (th_ && atoi(th_) > 0) omp_set_num_threads(atoi(th_));
    fprintf(stderr, "[bench] idot %s | f32 %s\n", IDOT_KERNEL, F32_KERNEL);
    fprintf(stderr, "%-28s %10s %10s %10s\n", "kernel", "s", "GFLOP/s", "GB/s");

    /* (a) matmul f32 4096x4096, S=1 */
    {
        int O = 4096, I = 4096, iters = 50;
        float *W = falloc((int64_t)O*I), *x = falloc(I), *y = falloc(O);
        bn_fill(W, (int64_t)O*I); bn_fill(x, I);
        matmul(y, x, W, 1, I, O);                       /* warm-up */
        double t0 = now_s();
        for (int it = 0; it < iters; it++) matmul(y, x, W, 1, I, O);
        double t = now_s() - t0;
        double fl = 2.0*O*I*iters;
        fprintf(stderr, "%-28s %10.4f %10.2f %10.2f\n", "matmul f32 4096x4096",
                t, fl/t/1e9, (double)O*I*4*iters/t/1e9);
        free(W); free(x); free(y);
    }
    /* (b) matmul_q int8 4096x4096 */
    {
        int O = 4096, I = 4096, iters = 50;
        float *W = falloc((int64_t)O*I), *x = falloc(I), *y = falloc(O);
        int8_t *q = malloc((int64_t)O*I); float *qs = falloc(O);
        bn_fill(W, (int64_t)O*I); bn_fill(x, I);
        quantize_rows(W, q, qs, O, I, 8);
        matmul_q(y, x, q, qs, I, O);                    /* warm-up */
        double t0 = now_s();
        for (int it = 0; it < iters; it++) matmul_q(y, x, q, qs, I, O);
        double t = now_s() - t0;
        double fl = 2.0*O*I*iters;
        fprintf(stderr, "%-28s %10.4f %10.2f %10.2f\n", "matmul_q int8 4096x4096",
                t, fl/t/1e9, (double)O*I*1*iters/t/1e9);
        free(W); free(x); free(y); free(q); free(qs);
    }
    /* (b2) matmul_q_s in batch (prefill): il peso viene letto una volta per
     * S righe di attivazione — GB/s pesi ~costante, GFLOP/s scala con S */
    {
        int O = 4096, I = 4096, iters = 50;
        static const int SS[2] = {8, 64};
        float *W = falloc((int64_t)O*I);
        int8_t *q = malloc((int64_t)O*I); float *qs = falloc(O);
        bn_fill(W, (int64_t)O*I);
        quantize_rows(W, q, qs, O, I, 8);
        for (int si = 0; si < 2; si++) {
            int S = SS[si];
            float *x = falloc((int64_t)S*I), *y = falloc((int64_t)S*O);
            bn_fill(x, (int64_t)S*I);
            matmul_q_s(y, x, q, qs, S, I, O);           /* warm-up */
            double t0 = now_s();
            for (int it = 0; it < iters; it++) matmul_q_s(y, x, q, qs, S, I, O);
            double t = now_s() - t0;
            double fl = 2.0*O*I*(double)S*iters;
            char nm[64]; snprintf(nm, sizeof nm, "matmul_q int8 4096x4096 S=%d", S);
            fprintf(stderr, "%-28s %10.4f %10.2f %10.2f\n", nm,
                    t, fl/t/1e9, (double)O*I*1*iters/t/1e9);
            free(x); free(y);
        }
        free(W); free(q); free(qs);
    }
    /* (b3) matmul_i4_grouped int4 (QBITS=4, gs=32): meta' dei byte di int8 */
    {
        int O = 4096, I = 4096, gs = 32, ng = I/gs, iters = 50;
        float *W = falloc((int64_t)O*I), *x = falloc(I), *y = falloc(O);
        uint8_t *q4 = malloc((int64_t)O*(I/2)); float *qs = falloc((int64_t)O*ng);
        bn_fill(W, (int64_t)O*I); bn_fill(x, I);
        pack_int4_grouped(W, q4, qs, O, I, gs);
        matmul_i4_grouped_s(y, x, q4, qs, 1, I, O, gs); /* warm-up */
        double t0 = now_s();
        for (int it = 0; it < iters; it++) matmul_i4_grouped_s(y, x, q4, qs, 1, I, O, gs);
        double t = now_s() - t0;
        double fl = 2.0*O*I*iters;
        fprintf(stderr, "%-28s %10.4f %10.2f %10.2f\n", "matmul_i4 gs32 4096x4096",
                t, fl/t/1e9, (double)O*(I/2)*iters/t/1e9);
        free(W); free(x); free(y); free(q4); free(qs);
    }
    /* (c2) dot_f32i8 forma KV int8: 4096 dot di lunghezza 128 */
    {
        int T = 4096, hd = 128, iters = 1000;
        float *qv = falloc(hd);
        int8_t *K8 = malloc((int64_t)T*hd);
        bn_fill(qv, hd);
        for (int64_t i = 0; i < (int64_t)T*hd; i++) K8[i] = (int8_t)(i*37);
        volatile float sink = 0;
        double t0 = now_s();
        for (int it = 0; it < iters; it++)
            for (int tt = 0; tt < T; tt++) sink += dot_f32i8(qv, K8 + (int64_t)tt*hd, hd);
        double t = now_s() - t0;
        double fl = 2.0*(double)T*hd*iters;
        fprintf(stderr, "%-28s %10.4f %10.2f %10.2f\n", "dot_f32i8 kv 4096x128",
                t, fl/t/1e9, (double)T*hd*1*iters/t/1e9);
        (void)sink;
        free(qv); free(K8);
    }
    /* (c) dot_f32 forma attention: 4096 dot di lunghezza 128 */
    {
        int T = 4096, hd = 128, iters = 1000;
        float *K = falloc((int64_t)T*hd), *qv = falloc(hd);
        bn_fill(K, (int64_t)T*hd); bn_fill(qv, hd);
        volatile float sink = 0;
        double t0 = now_s();
        for (int it = 0; it < iters; it++) {
            float acc = 0;
            for (int t = 0; t < T; t++) acc += dot_f32(qv, K + (int64_t)t*hd, hd);
            sink += acc;
        }
        double t = now_s() - t0;
        (void)sink;
        double fl = 2.0*T*hd*iters;
        fprintf(stderr, "%-28s %10.4f %10.2f %10.2f\n", "dot_f32 attn 4096x128",
                t, fl/t/1e9, (double)T*hd*4*iters/t/1e9);
        free(K); free(qv);
    }
    /* (d) deltanet_token, dimensioni Qwen3.5-4B */
    {
        int D = 2560, Hv = 32, Hk = 16, dk = 128, dv = 128, K = 4, T = 200;
        int kd = Hk*dk, vd = Hv*dv, cd = 2*kd + vd;
        Model m; memset(&m, 0, sizeof m);
        m.c.hidden = D; m.c.eps = 1e-6f;
        m.c.lin_hv = Hv; m.c.lin_hk = Hk; m.c.lin_dk = dk; m.c.lin_dv = dv; m.c.lin_conv = K;
        Layer l; memset(&l, 0, sizeof l);
        l.type = LT_LINEAR;
        #define MKB(mat, O_, I_) do { l.mat.O = O_; l.mat.I = I_; \
            l.mat.f = falloc((int64_t)(O_)*(I_)); bn_fill(l.mat.f, (int64_t)(O_)*(I_)); } while (0)
        MKB(aqkv, cd, D); MKB(az, vd, D); MKB(ab, Hv, D); MKB(aa, Hv, D); MKB(aout, D, vd);
        #undef MKB
        l.conv_w = falloc((int64_t)cd*K);  bn_fill(l.conv_w, (int64_t)cd*K);
        l.conv_b = falloc(cd);             bn_fill(l.conv_b, cd);
        l.dt_bias = falloc(Hv);            bn_fill(l.dt_bias, Hv);
        l.A_log = falloc(Hv);              bn_fill(l.A_log, Hv);
        l.dn_norm = falloc(dv);            bn_fill(l.dn_norm, dv);
        l.conv_state = calloc((size_t)cd*K, sizeof(float));
        l.Sstate = calloc((size_t)Hv*dk*dv, sizeof(float));
        float *x = falloc(D), *out = falloc(D);
        bn_fill(x, D);
        deltanet_token(&m, &l, x, out);                 /* warm-up */
        double t0 = now_s();
        for (int t = 0; t < T; t++) deltanet_token(&m, &l, x, out);
        double t = now_s() - t0;
        /* FLOP ricorrenza: 4 * Hv*dk*dv per token (decay+acc, update+dot) piu' i GEMV */
        double fl_rec = 4.0*Hv*dk*dv*T;
        double fl_mm = 2.0*((double)cd*D + (double)vd*D + 2.0*Hv*D + (double)D*vd)*T;
        fprintf(stderr, "%-28s %10.4f %10.2f %10s  (%.0f tok/s)\n", "deltanet_token 4B x200",
                t, (fl_rec+fl_mm)/t/1e9, "-", T/t);
        free(x); free(out);
    }
    /* (e) attention in decode a contesto lungo: 1 token nuovo, T=2048 in cache
     * (layer denso finto H=32 KV=8 hd=128 D=4096; K/V pre-riempite random) */
    {
        int D = 4096, H = 32, KV = 8, hd = 128, T = 2048, iters = 200;
        Model m; memset(&m, 0, sizeof m);
        m.c.hidden = D; m.c.n_heads = H; m.c.n_kv_heads = KV; m.c.head_dim = hd;
        m.c.rot = hd; m.c.theta = 1e6f; m.c.eps = 1e-6f; m.c.n_layers = 1;
        static int lt[1] = {LT_FULL}; m.c.ltype = lt;
        Layer l; memset(&l, 0, sizeof l);
        l.type = LT_FULL;
        #define MKB(mat, O_, I_) do { l.mat.O = O_; l.mat.I = I_; \
            l.mat.f = falloc((int64_t)(O_)*(I_)); bn_fill(l.mat.f, (int64_t)(O_)*(I_)); } while (0)
        MKB(q, H*hd, D); MKB(k, KV*hd, D); MKB(v, KV*hd, D); MKB(o, D, H*hd);
        #undef MKB
        l.qn = falloc(hd); bn_fill(l.qn, hd);
        l.kn = falloc(hd); bn_fill(l.kn, hd);
        kv_alloc(&m, T + 1);
        bn_fill(m.K[0], (int64_t)KV*(T+1)*hd);          /* cache pre-riempita direttamente */
        bn_fill(m.V[0], (int64_t)KV*(T+1)*hd);
        float *x = falloc(D), *out = falloc(D);
        bn_fill(x, D);
        attention(&m, &l, 0, x, 1, T, out);             /* warm-up */
        double t0 = now_s();
        for (int it = 0; it < iters; it++) attention(&m, &l, 0, x, 1, T, out);
        double t = now_s() - t0;
        double fl = 2.0*2.0*H*hd*(double)T*iters;       /* punteggi + contesto */
        double by = 2.0*KV*(double)T*hd*4*iters;        /* byte K+V letti dalla cache */
        fprintf(stderr, "%-28s %10.4f %10.2f %10.2f\n", "attention decode T=2048",
                t, fl/t/1e9, by/t/1e9);
        free(l.q.f); free(l.k.f); free(l.v.f); free(l.o.f);
        free(l.qn); free(l.kn); free(x); free(out);
    }
    return 0;
}
