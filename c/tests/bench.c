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
    return 0;
}
