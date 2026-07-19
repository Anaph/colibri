/* Test dei kernel di simd.h contro riferimenti scalari in int64/double.
 * simd.h e' autosufficiente: lo includiamo direttamente, senza motore.
 * Convenzione: 0 = pass, 1 = fail, 2 = skip. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "../simd.h"

/* LCG deterministico (stesso stile di qwen_tests.c) */
static uint64_t st_lcg = 0x1234567895555ULL;
static uint32_t st_rnd(void) {
    st_lcg = st_lcg*6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(st_lcg >> 33);
}
static float st_frnd(void) { return (float)st_rnd()/2147483648.0f - 1.0f; }

/* dot_i8i8 vs riferimento scalare int64: uguaglianza ESATTA, tutte le
 * lunghezze attorno ai passi vettoriali (16/32/64) e una rep con w=-128
 * ovunque (il caso limite del trucco del segno). */
int st_dot_i8i8(void) {
    static const int ns[] = {1,2,15,16,17,31,32,33,63,64,65,127,128,255,256,257,4096};
    int8_t w[4096], x[4096];
    for (size_t k = 0; k < sizeof ns/sizeof ns[0]; k++) {
        int n = ns[k];
        for (int rep = 0; rep < 32; rep++) {
            for (int i = 0; i < n; i++) {
                w[i] = rep == 0 ? -128 : (int8_t)(st_rnd() & 0xFF);   /* pieno range, incl. -128 */
                int v = (int)(st_rnd() % 255) - 127;                  /* |x|<=127 come qrow_i8 */
                x[i] = (int8_t)v;
            }
            int64_t ref = 0;
            for (int i = 0; i < n; i++) ref += (int64_t)w[i]*x[i];
            int32_t got = dot_i8i8(w, x, n);
            if ((int64_t)got != ref) {
                fprintf(stderr, "dot_i8i8 n=%d rep=%d: got %d expected %lld\n",
                        n, rep, got, (long long)ref);
                return 1;
            }
        }
    }
    return 0;
}

/* dot_i4i8 vs riferimento scalare int64 sul layout impacchettato (pari->nibble
 * basso, v+8): uguaglianza ESATTA, lunghezze attorno ai passi vettoriali
 * (32/64) e code dispari. */
int st_dot_i4i8(void) {
    static const int ns[] = {1,2,3,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,4097};
    uint8_t w4[(4097+1)/2]; int8_t x[4097];
    for (size_t k = 0; k < sizeof ns/sizeof ns[0]; k++) {
        int n = ns[k];
        for (int rep = 0; rep < 32; rep++) {
            for (int i = 0; i < (n+1)/2; i++) w4[i] = (uint8_t)(st_rnd() & 0xFF);
            for (int i = 0; i < n; i++) x[i] = (int8_t)((int)(st_rnd() % 255) - 127);
            int64_t ref = 0;
            for (int i = 0; i < n; i++) {
                uint8_t b = w4[i>>1];
                int v = (i & 1) ? (int)(b >> 4) - 8 : (int)(b & 0xF) - 8;
                ref += (int64_t)v * x[i];
            }
            int32_t got = dot_i4i8(w4, x, n);
            if ((int64_t)got != ref) {
                fprintf(stderr, "dot_i4i8 n=%d rep=%d: got %d expected %lld\n",
                        n, rep, got, (long long)ref);
                return 1;
            }
        }
    }
    return 0;
}

/* dot_f32i8 vs riferimento double, tolleranza relativa 1e-5 */
int st_dot_f32i8(void) {
    static const int ns[] = {1,7,8,9,15,16,17,100,4096};
    float x[4096]; int8_t w[4096];
    for (size_t k = 0; k < sizeof ns/sizeof ns[0]; k++) {
        int n = ns[k];
        for (int rep = 0; rep < 32; rep++) {
            for (int i = 0; i < n; i++) {
                x[i] = st_frnd();
                w[i] = (int8_t)(st_rnd() & 0xFF);       /* pieno range, incl. -128 */
            }
            double ref = 0, mag = 0;
            for (int i = 0; i < n; i++) {
                double p = (double)x[i]*(double)w[i];
                ref += p; mag += fabs(p);
            }
            float got = dot_f32i8(x, w, n);
            /* denominatore = somma dei |prodotti|: con w a pieno range la somma
             * cancella pesantemente e il bound relativo a |ref| non e' sensato */
            double den = mag > 1.0 ? mag : 1.0;
            if (fabs((double)got - ref)/den > 1e-5) {
                fprintf(stderr, "dot_f32i8 n=%d rep=%d: got %.8f expected %.8f\n",
                        n, rep, got, ref);
                return 1;
            }
        }
    }
    return 0;
}

/* dot_f32 vs riferimento double, tolleranza relativa 1e-5 */
int st_dot_f32(void) {
    static const int ns[] = {1,7,8,15,16,31,32,100,4096};
    float a[4096], b[4096];
    for (size_t k = 0; k < sizeof ns/sizeof ns[0]; k++) {
        int n = ns[k];
        for (int rep = 0; rep < 32; rep++) {
            for (int i = 0; i < n; i++) { a[i] = st_frnd(); b[i] = st_frnd(); }
            double ref = 0;
            for (int i = 0; i < n; i++) ref += (double)a[i]*b[i];
            float got = dot_f32(a, b, n);
            double den = fabs(ref) > 1.0 ? fabs(ref) : 1.0;
            if (fabs((double)got - ref)/den > 1e-5) {
                fprintf(stderr, "dot_f32 n=%d rep=%d: got %.8f expected %.8f\n",
                        n, rep, got, ref);
                return 1;
            }
        }
    }
    return 0;
}

/* qrow_i8: ricostruzione |x[i] - s*q[i]| <= s/2 (arrotondamento al piu' vicino) */
int st_qrow(void) {
    static const int ns[] = {1,3,16,100,1000};
    float x[1000]; int8_t q[1000];
    for (size_t k = 0; k < sizeof ns/sizeof ns[0]; k++) {
        int n = ns[k];
        for (int rep = 0; rep < 16; rep++) {
            for (int i = 0; i < n; i++) x[i] = st_frnd()*4.0f;
            float s = qrow_i8(x, q, n);
            for (int i = 0; i < n; i++) {
                if (fabsf(x[i] - s*(float)q[i]) > s*0.5f + 1e-7f) {
                    fprintf(stderr, "qrow_i8 n=%d i=%d: x=%.6f s=%.6g q=%d\n",
                            n, i, x[i], s, q[i]);
                    return 1;
                }
            }
        }
    }
    return 0;
}

/* kernel di riga del DeltaNet vs cicli scalari, stato/accumuli condivisi */
int st_dn_rows(void) {
    static const int dvs[] = {1,8,15,16,128};
    float S[128], Sr[128], kv[128], kvr[128], delta[128], oh[128], ohr[128];
    for (size_t k = 0; k < sizeof dvs/sizeof dvs[0]; k++) {
        int dv = dvs[k];
        for (int rep = 0; rep < 16; rep++) {
            float dec = 0.5f + 0.5f*fabsf(st_frnd());
            float ki = st_frnd(), qi = st_frnd();
            for (int j = 0; j < dv; j++) {
                S[j] = Sr[j] = st_frnd();
                kv[j] = kvr[j] = st_frnd();
                delta[j] = st_frnd();
                oh[j] = ohr[j] = st_frnd();
            }
            dn_row_decay_acc(S, dec, ki, kv, dv);
            for (int j = 0; j < dv; j++) { Sr[j] *= dec; kvr[j] += Sr[j]*ki; }
            for (int j = 0; j < dv; j++)
                if (fabsf(S[j]-Sr[j]) > 1e-6f || fabsf(kv[j]-kvr[j]) > 1e-6f) {
                    fprintf(stderr, "dn_row_decay_acc dv=%d j=%d: S %.7f/%.7f kv %.7f/%.7f\n",
                            dv, j, S[j], Sr[j], kv[j], kvr[j]);
                    return 1;
                }
            dn_row_update_dot(S, ki, delta, qi, oh, dv);
            for (int j = 0; j < dv; j++) { Sr[j] += ki*delta[j]; ohr[j] += Sr[j]*qi; }
            for (int j = 0; j < dv; j++)
                if (fabsf(S[j]-Sr[j]) > 1e-6f || fabsf(oh[j]-ohr[j]) > 1e-6f) {
                    fprintf(stderr, "dn_row_update_dot dv=%d j=%d: S %.7f/%.7f oh %.7f/%.7f\n",
                            dv, j, S[j], Sr[j], oh[j], ohr[j]);
                    return 1;
                }
        }
    }
    return 0;
}

/* logga i tier compilati (utile in CI per capire cosa si sta testando) */
int st_report(void) {
    fprintf(stderr, "idot=%s f32=%s\n", IDOT_KERNEL, F32_KERNEL);
    return 0;
}
