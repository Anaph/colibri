/* Trainer LoRA per il motore qwen: forward con stash + backward SCRITTO A MANO
 * (niente autograd) + AdamW. Incluso in fondo a qwen.c: vede tutti i tipi e i
 * kernel del motore. Si addestrano SOLO gli adattatori (A,B per slot), i pesi
 * base restano congelati: il backward propaga dx attraverso i pesi f32.
 *
 * Limiti della v1 (verificati da train_guard): modello denso Qwen3 (niente
 * ibridi/gated), pesi f32 (QBITS=0), tutti i layer residenti. */
#ifndef QWEN_TRAIN_H
#define QWEN_TRAIN_H

/* ---------- primitive ---------- */

/* dx[S,I] = dy[S,O] · W[O,I] (cioe' W^T applicata da sinistra per riga).
 * Parallelo su blocchi di colonne i: ogni thread scorre tutte le righe di W
 * sequenzialmente sul proprio range di i (segmento di riga contiguo). */
static void matmul_tn(float *dx, const float *dy, const float *W, int S, int O, int I) {
    memset(dx, 0, (size_t)S * I * sizeof(float));
    int nb = (I + 255) / 256;
    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < nb; ib++) {
        int i0 = ib * 256, i1 = i0 + 256 > I ? I : i0 + 256;
        for (int o = 0; o < O; o++) {
            const float *wrow = W + (int64_t)o * I;
            for (int s = 0; s < S; s++) {
                float dyv = dy[(int64_t)s * O + o];
                float *dxs = dx + (int64_t)s * I;
                for (int i = i0; i < i1; i++) dxs[i] += dyv * wrow[i];
            }
        }
    }
}

/* AdamW standard con bias-correction (beta 0.9/0.999, eps 1e-8) */
typedef struct { float *m, *v; } Adam;
static void adamw_step(float *w, const float *g, Adam *a, int64_t n, float lr, float wd, int t) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.f - powf(b1, (float)t), bc2 = 1.f - powf(b2, (float)t);
    for (int64_t i = 0; i < n; i++) {
        a->m[i] = b1 * a->m[i] + (1.f - b1) * g[i];
        a->v[i] = b2 * a->v[i] + (1.f - b2) * g[i] * g[i];
        float mh = a->m[i] / bc1, vh = a->v[i] / bc2;
        w[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * w[i]);
    }
}

/* come rmsnorm_row ma stasha il reciproco r (serve al backward) */
static void t_rmsnorm_row(float *out, const float *x, const float *w, int D, float eps, float *r_out) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    *r_out = r;
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

/* backward di y_i = x_i*r*w_i con r stashed: ACCUMULA in dx
 * dx_i = r*w_i*dy_i - x_i*(r^3/D)*sum_j dy_j*w_j*x_j */
static void bw_rmsnorm(float *dx, const float *dy, const float *x, const float *w, float r, int D) {
    double acc = 0;
    for (int i = 0; i < D; i++) acc += (double)dy[i] * w[i] * x[i];
    float k = (float)acc * r * r * r / D;
    for (int i = 0; i < D; i++) dx[i] += r * w[i] * dy[i] - x[i] * k;
}

/* RoPE inverso: la rotazione e' ortogonale, il backward e' R(-ang) = R(ang)^T */
static void rope_head_inv(float *x, int pos, float theta, int rot) {
    int h = rot / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(theta, -2.0f * j / rot);
        float ang = pos * inv, cs = cosf(ang), sn = -sinf(ang);
        float a = x[j], b = x[j + h];
        x[j]     = a * cs - b * sn;
        x[j + h] = b * cs + a * sn;
    }
}

/* backward di y = W*x + sc*B*(A*x): accumula dA/dB in g (se g != NULL) e
 * scrive dx = W^T*dy + sc*A^T*(B^T*dy) (dx sovrascritto; NULL = non serve).
 * Il termine LoRA su dx va incluso anche senza g: l'adattatore e' nel forward. */
static void bw_lora_mat(const Mat *W, const Lora *lo, Lora *g, const float *x,
                        const float *dy, float *dx, int S) {
    if (dx) matmul_tn(dx, dy, W->f, S, W->O, W->I);
    if (!lo || !lo->A || lo->r <= 0) return;
    int r = lo->r, I = W->I, O = W->O;
    float sc = lo->alpha / r;
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s * I, *dys = dy + (int64_t)s * O;
        float t[LORA_MAX_R], bt[LORA_MAX_R];
        for (int j = 0; j < r; j++) t[j] = dot_f32(lo->A + (int64_t)j * I, xs, I);
        for (int j = 0; j < r; j++) {
            double a = 0;
            for (int o = 0; o < O; o++) a += (double)lo->B[(int64_t)o * r + j] * dys[o];
            bt[j] = (float)a;
        }
        if (g) {
            for (int o = 0; o < O; o++)
                for (int j = 0; j < r; j++) g->B[(int64_t)o * r + j] += sc * dys[o] * t[j];
            for (int j = 0; j < r; j++) {
                float cj = sc * bt[j];
                for (int i = 0; i < I; i++) g->A[(int64_t)j * I + i] += cj * xs[i];
            }
        }
        if (dx) {
            float *dxs = dx + (int64_t)s * I;
            for (int j = 0; j < r; j++) {
                float cj = sc * bt[j];
                for (int i = 0; i < I; i++) dxs[i] += cj * lo->A[(int64_t)j * I + i];
            }
        }
    }
}

/* ---------- stash per-layer del forward di training ---------- */
typedef struct {
    float *x_in, *nrm_a;        /* [S,D] residuo in ingresso e post in_ln */
    float *q_pre, *q_rot, *ctx; /* [S,H*hd] pre qk-norm, post rope, contesto */
    float *k_pre;               /* [S,KV*hd] pre qk-norm (k post-rope in m->K) */
    float *probs;               /* [H,S,S] pesi attention (0 sopra la diagonale) */
    float *x_mid, *nrm_m;       /* [S,D] residuo dopo attention e post post_ln */
    float *g_pre, *u;           /* [S,inter] gate PRE-attivazione e up */
    float *r_in, *r_post;       /* [S] reciproci rmsnorm */
    float *rq, *rk;             /* [S*H], [S*KV] reciproci qk-norm */
} LStash;

static void lstash_alloc(LStash *st, const Cfg *c, int S) {
    int D = c->hidden, H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim, IN = c->inter;
    st->x_in  = falloc((int64_t)S * D); st->nrm_a = falloc((int64_t)S * D);
    st->x_mid = falloc((int64_t)S * D); st->nrm_m = falloc((int64_t)S * D);
    st->q_pre = falloc((int64_t)S * H * hd); st->q_rot = falloc((int64_t)S * H * hd);
    st->ctx   = falloc((int64_t)S * H * hd);
    st->k_pre = falloc((int64_t)S * KV * hd);
    st->probs = falloc((int64_t)H * S * S);
    st->g_pre = falloc((int64_t)S * IN); st->u = falloc((int64_t)S * IN);
    st->r_in = falloc(S); st->r_post = falloc(S);
    st->rq = falloc((int64_t)S * H); st->rk = falloc((int64_t)S * KV);
}

static void lstash_free(LStash *st) {
    free(st->x_in); free(st->nrm_a); free(st->x_mid); free(st->nrm_m);
    free(st->q_pre); free(st->q_rot); free(st->ctx); free(st->k_pre);
    free(st->probs); free(st->g_pre); free(st->u);
    free(st->r_in); free(st->r_post); free(st->rq); free(st->rk);
}

static float *fzalloc(int64_t n) { float *p = falloc(n); memset(p, 0, (size_t)n * sizeof(float)); return p; }

/* ---------- forward di training (denso non-gated, stash completo) ---------- */

/* attenzione con stash: input st->nrm_a gia' riempito, k/v in cache a pos 0..S-1.
 * Loop punteggi SERIALI (finestra di training piccola): probs stashate piene. */
static void t_attention(Model *m, Layer *l, int layer, int S, LStash *st, float *out) {
    Cfg *c = &m->c;
    int H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim, G = H / KV;
    int64_t qw = (int64_t)H * hd, kw = (int64_t)KV * hd;
    mat_apply(st->q_pre, st->nrm_a, &l->q, S);
    mat_apply(st->k_pre, st->nrm_a, &l->k, S);
    float *vv = falloc(S * kw);
    mat_apply(vv, st->nrm_a, &l->v, S);
    if (l->lo) {
        lora_apply(&l->lo->q, st->q_pre, st->nrm_a, S, l->q.I, l->q.O);
        lora_apply(&l->lo->k, st->k_pre, st->nrm_a, S, l->k.I, l->k.O);
        lora_apply(&l->lo->v, vv,        st->nrm_a, S, l->v.I, l->v.O);
    }
    for (int s = 0; s < S; s++) {
        for (int hh = 0; hh < H; hh++) {
            float *qr = st->q_rot + s * qw + (int64_t)hh * hd;
            t_rmsnorm_row(qr, st->q_pre + s * qw + (int64_t)hh * hd, l->qn, hd, c->eps, &st->rq[s * H + hh]);
            rope_head(qr, s, c->theta, c->rot);
        }
        for (int hh = 0; hh < KV; hh++) {
            float *kdst = m->K[layer] + ((int64_t)hh * m->max_t + s) * hd;
            t_rmsnorm_row(kdst, st->k_pre + s * kw + (int64_t)hh * hd, l->kn, hd, c->eps, &st->rk[s * KV + hh]);
            rope_head(kdst, s, c->theta, c->rot);
            memcpy(m->V[layer] + ((int64_t)hh * m->max_t + s) * hd, vv + s * kw + (int64_t)hh * hd, hd * sizeof(float));
        }
    }
    float scale = 1.f / sqrtf((float)hd);
    memset(st->probs, 0, (size_t)H * S * S * sizeof(float));
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / G;
        for (int s = 0; s < S; s++) {
            float *pr = st->probs + ((int64_t)hh * S + s) * S;
            const float *qv = st->q_rot + s * qw + (int64_t)hh * hd;
            for (int t = 0; t <= s; t++)
                pr[t] = dot_f32(qv, m->K[layer] + ((int64_t)kvh * m->max_t + t) * hd, hd) * scale;
            softmax_row(pr, s + 1);
            float *cx = st->ctx + s * qw + (int64_t)hh * hd;
            for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
            for (int t = 0; t <= s; t++) {
                const float *vr = m->V[layer] + ((int64_t)kvh * m->max_t + t) * hd;
                float a = pr[t];
                for (int dd = 0; dd < hd; dd++) cx[dd] += a * vr[dd];
            }
        }
    }
    mat_apply(out, st->ctx, &l->o, S);
    if (l->lo) lora_apply(&l->lo->o, out, st->ctx, S, l->o.I, l->o.O);
    free(vv);
}

/* forward di UN layer con stash; il residuo x [S,D] viene aggiornato in place */
static void t_layer_forward(Model *m, Layer *l, int layer, float *x, int S, LStash *st) {
    Cfg *c = &m->c; int D = c->hidden, IN = c->inter;
    memcpy(st->x_in, x, (size_t)S * D * sizeof(float));
    for (int s = 0; s < S; s++)
        t_rmsnorm_row(st->nrm_a + (int64_t)s * D, x + (int64_t)s * D, l->in_ln, D, c->eps, &st->r_in[s]);
    float *att = falloc((int64_t)S * D);
    t_attention(m, l, layer, S, st, att);
    for (int64_t i = 0; i < (int64_t)S * D; i++) x[i] += att[i];
    free(att);
    memcpy(st->x_mid, x, (size_t)S * D * sizeof(float));
    for (int s = 0; s < S; s++)
        t_rmsnorm_row(st->nrm_m + (int64_t)s * D, x + (int64_t)s * D, l->post_ln, D, c->eps, &st->r_post[s]);
    mat_apply(st->g_pre, st->nrm_m, &l->gate, S);
    mat_apply(st->u,     st->nrm_m, &l->up,   S);
    if (l->lo) {
        lora_apply(&l->lo->gate, st->g_pre, st->nrm_m, S, l->gate.I, l->gate.O);
        lora_apply(&l->lo->up,   st->u,     st->nrm_m, S, l->up.I,   l->up.O);
    }
    float *h = falloc((int64_t)S * IN), *d = falloc((int64_t)S * D);
    for (int64_t i = 0; i < (int64_t)S * IN; i++) {
        float gp = st->g_pre[i];
        h[i] = (gp / (1.f + expf(-gp))) * st->u[i];
    }
    mat_apply(d, h, &l->down, S);
    if (l->lo) lora_apply(&l->lo->down, d, h, S, l->down.I, l->down.O);
    for (int64_t i = 0; i < (int64_t)S * D; i++) x[i] += d[i];
    free(h); free(d);
}

/* forward completo: layer < L0 con i kernel del motore (stesso flusso di
 * step()), layer >= L0 con stash. x_out [S,D] = residuo finale (pre-norma). */
static void train_forward(Model *m, const int *ids, int S, LStash *st, int L0, float *x_out) {
    Cfg *c = &m->c; int D = c->hidden;
    m->kv_len = 0;
    float *x = x_out;
    for (int s = 0; s < S; s++)
        memcpy(x + (int64_t)s * D, m->embed + (int64_t)ids[s] * D, D * sizeof(float));
    float *nrm = falloc((int64_t)S * D), *tmp = falloc((int64_t)S * D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (i < L0) {
            for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s * D, x + (int64_t)s * D, l->in_ln, D, c->eps);
            attention(m, l, i, nrm, S, 0, tmp);
            for (int64_t j = 0; j < (int64_t)S * D; j++) x[j] += tmp[j];
            for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s * D, x + (int64_t)s * D, l->post_ln, D, c->eps);
            mlp(m, l, nrm, S, tmp);
            for (int64_t j = 0; j < (int64_t)S * D; j++) x[j] += tmp[j];
        } else {
            t_layer_forward(m, l, i, x, S, &st[i - L0]);
        }
    }
    m->kv_len = S;
    free(nrm); free(tmp);
}

/* ---------- backward di UN layer ----------
 * dx [S,D] in ingresso = dL/dx_out; in uscita = dL/dx_in. g = gradienti degli
 * adattatori del layer (NULL = niente accumulo, solo propagazione). */
static void t_layer_backward(Model *m, Layer *l, int layer, int S, LStash *st, LoraLayer *g, float *dx) {
    Cfg *c = &m->c; int D = c->hidden, IN = c->inter;
    int H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim, G = H / KV;
    int64_t qw = (int64_t)H * hd, kw = (int64_t)KV * hd;
    /* --- mlp: x_out = x_mid + down(silu(g_pre)*u) --- */
    float *h = falloc((int64_t)S * IN), *dh = falloc((int64_t)S * IN);
    for (int64_t i = 0; i < (int64_t)S * IN; i++) {
        float gp = st->g_pre[i];
        h[i] = (gp / (1.f + expf(-gp))) * st->u[i];
    }
    bw_lora_mat(&l->down, l->lo ? &l->lo->down : NULL, g ? &g->down : NULL, h, dx, dh, S);
    float *du = falloc((int64_t)S * IN), *dg = falloc((int64_t)S * IN);
    for (int64_t i = 0; i < (int64_t)S * IN; i++) {
        float gp = st->g_pre[i], sg = 1.f / (1.f + expf(-gp));
        du[i] = dh[i] * gp * sg;                           /* d/du: silu(g) */
        dg[i] = dh[i] * st->u[i] * sg * (1.f + gp * (1.f - sg)); /* silu'(g) */
    }
    float *dn = falloc((int64_t)S * D), *dn2 = falloc((int64_t)S * D);
    bw_lora_mat(&l->up,   l->lo ? &l->lo->up   : NULL, g ? &g->up   : NULL, st->nrm_m, du, dn,  S);
    bw_lora_mat(&l->gate, l->lo ? &l->lo->gate : NULL, g ? &g->gate : NULL, st->nrm_m, dg, dn2, S);
    for (int64_t i = 0; i < (int64_t)S * D; i++) dn[i] += dn2[i];
    /* post_ln: dx_mid = dx_out + bw_rmsnorm(dnrm_m); accumulo diretto in dx */
    for (int s = 0; s < S; s++)
        bw_rmsnorm(dx + (int64_t)s * D, dn + (int64_t)s * D, st->x_mid + (int64_t)s * D, l->post_ln, st->r_post[s], D);
    free(h); free(dh); free(du); free(dg);
    /* --- attention: x_mid = x_in + o_proj(ctx) --- */
    float *dctx = falloc(S * qw);
    bw_lora_mat(&l->o, l->lo ? &l->lo->o : NULL, g ? &g->o : NULL, st->ctx, dx, dctx, S);
    float *dqrot = fzalloc(S * qw), *dkrot = fzalloc(S * kw), *dv = fzalloc(S * kw);
    float *dp = falloc(S);
    float scale = 1.f / sqrtf((float)hd);
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / G;
        for (int s = 0; s < S; s++) {
            const float *dcx = dctx + s * qw + (int64_t)hh * hd;
            const float *pr = st->probs + ((int64_t)hh * S + s) * S;
            double psum = 0;
            for (int t = 0; t <= s; t++) {
                dp[t] = dot_f32(dcx, m->V[layer] + ((int64_t)kvh * m->max_t + t) * hd, hd);
                psum += (double)pr[t] * dp[t];
            }
            float *dq = dqrot + s * qw + (int64_t)hh * hd;
            const float *qv = st->q_rot + s * qw + (int64_t)hh * hd;
            for (int t = 0; t <= s; t++) {
                float dsv = pr[t] * (dp[t] - (float)psum);    /* softmax backward */
                const float *kr = m->K[layer] + ((int64_t)kvh * m->max_t + t) * hd;
                float *dk = dkrot + t * kw + (int64_t)kvh * hd;
                float *dvt = dv + t * kw + (int64_t)kvh * hd;
                for (int dd = 0; dd < hd; dd++) {
                    dq[dd]  += scale * dsv * kr[dd];
                    dk[dd]  += scale * dsv * qv[dd];          /* GQA: somma sulle G teste q */
                    dvt[dd] += pr[t] * dcx[dd];
                }
            }
        }
    }
    free(dp);
    /* rope inverso + qk-norm backward -> gradienti pre-proiezione */
    float *dqpre = fzalloc(S * qw), *dkpre = fzalloc(S * kw);
    for (int s = 0; s < S; s++) {
        for (int hh = 0; hh < H; hh++) {
            float *dq = dqrot + s * qw + (int64_t)hh * hd;
            rope_head_inv(dq, s, c->theta, c->rot);
            bw_rmsnorm(dqpre + s * qw + (int64_t)hh * hd, dq,
                       st->q_pre + s * qw + (int64_t)hh * hd, l->qn, st->rq[s * H + hh], hd);
        }
        for (int hh = 0; hh < KV; hh++) {
            float *dk = dkrot + s * kw + (int64_t)hh * hd;
            rope_head_inv(dk, s, c->theta, c->rot);
            bw_rmsnorm(dkpre + s * kw + (int64_t)hh * hd, dk,
                       st->k_pre + s * kw + (int64_t)hh * hd, l->kn, st->rk[s * KV + hh], hd);
        }
    }
    float *dna = falloc((int64_t)S * D), *tmp = falloc((int64_t)S * D);
    bw_lora_mat(&l->q, l->lo ? &l->lo->q : NULL, g ? &g->q : NULL, st->nrm_a, dqpre, dna, S);
    bw_lora_mat(&l->k, l->lo ? &l->lo->k : NULL, g ? &g->k : NULL, st->nrm_a, dkpre, tmp, S);
    for (int64_t i = 0; i < (int64_t)S * D; i++) dna[i] += tmp[i];
    bw_lora_mat(&l->v, l->lo ? &l->lo->v : NULL, g ? &g->v : NULL, st->nrm_a, dv, tmp, S);
    for (int64_t i = 0; i < (int64_t)S * D; i++) dna[i] += tmp[i];
    /* in_ln: dx_in = dx_mid + bw_rmsnorm(dnrm_a) */
    for (int s = 0; s < S; s++)
        bw_rmsnorm(dx + (int64_t)s * D, dna + (int64_t)s * D, st->x_in + (int64_t)s * D, l->in_ln, st->r_in[s], D);
    free(dctx); free(dqrot); free(dkrot); free(dv); free(dqpre); free(dkpre); free(dna); free(tmp);
}

/* ---------- loss + backward completo ----------
 * CE media sulle posizioni 0..S-2 (target ids[s+1]). gL = gradienti dei layer
 * addestrati [n_layers-L0] (NULL = solo loss, niente backward); gHead =
 * gradienti dell'adattatore lm_head (puo' essere NULL). Ritorna la loss. */
static int t_ce_chunk = 32;   /* posizioni per blocco di logits (TRAIN_CE_CHUNK) */

static double train_loss_and_backward(Model *m, const int *ids, int S, int L0,
                                      LStash *st, LoraLayer *gL, Lora *gHead) {
    Cfg *c = &m->c; int D = c->hidden, V = c->vocab;
    int bw = (gL != NULL) || (gHead != NULL);
    float *x = falloc((int64_t)S * D);
    train_forward(m, ids, S, st, L0, x);
    float *fx = falloc((int64_t)S * D), *rf = falloc(S);
    for (int s = 0; s < S; s++)
        t_rmsnorm_row(fx + (int64_t)s * D, x + (int64_t)s * D, m->final_norm, D, c->eps, &rf[s]);
    int Ntok = S - 1;
    int CH = t_ce_chunk < 1 ? 32 : t_ce_chunk;
    float *lg = falloc((int64_t)CH * V);
    float *dlg = bw ? falloc((int64_t)CH * V) : NULL;
    float *dfx = bw ? fzalloc((int64_t)S * D) : NULL;
    double loss = 0;
    for (int s0 = 0; s0 < Ntok; s0 += CH) {
        int n = Ntok - s0 < CH ? Ntok - s0 : CH;
        mat_apply(lg, fx + (int64_t)s0 * D, &m->lm_head, n);
        lora_apply(&m->lm_lora, lg, fx + (int64_t)s0 * D, n, m->lm_head.I, m->lm_head.O);
        for (int p = 0; p < n; p++) {
            float *row = lg + (int64_t)p * V;
            int y = ids[s0 + p + 1];
            float mx = row[0]; for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
            double sum = 0; for (int v = 0; v < V; v++) sum += exp((double)row[v] - mx);
            loss += -((double)row[y] - mx - log(sum));
            if (dlg) {   /* dlogit = (softmax - e_y)/Ntok */
                float *dr = dlg + (int64_t)p * V;
                for (int v = 0; v < V; v++) dr[v] = (float)(exp((double)row[v] - mx) / sum) / Ntok;
                dr[y] -= 1.f / Ntok;
            }
        }
        if (dlg)
            bw_lora_mat(&m->lm_head, m->lm_lora.r ? &m->lm_lora : NULL, gHead,
                        fx + (int64_t)s0 * D, dlg, dfx + (int64_t)s0 * D, n);
    }
    loss /= Ntok;
    free(lg);
    if (!bw) { free(fx); free(rf); free(x); return loss; }
    free(dlg);
    /* norma finale (la posizione S-1 non contribuisce: dfx li' resta 0) */
    float *dxr = fzalloc((int64_t)S * D);
    for (int s = 0; s < S; s++)
        bw_rmsnorm(dxr + (int64_t)s * D, dfx + (int64_t)s * D, x + (int64_t)s * D, m->final_norm, rf[s], D);
    free(dfx);
    for (int i = c->n_layers - 1; i >= L0; i--)
        t_layer_backward(m, &m->L[i], i, S, &st[i - L0], gL ? &gL[i - L0] : NULL, dxr);
    free(dxr); free(fx); free(rf); free(x);
    return loss;
}

/* ---------- setup del training ---------- */

/* la v1 del trainer copre solo il percorso su cui il backward e' scritto */
static void train_guard(Model *m, int L0) {
    if (m->qbits != 0) { fprintf(stderr, "[train] richiede pesi f32 (QBITS=0)\n"); exit(1); }
    if (m->c.hybrid)   { fprintf(stderr, "[train] training v1: solo modelli Qwen3 densi (niente layer deltanet)\n"); exit(1); }
    if (m->n_resident != m->c.n_layers) { fprintf(stderr, "[train] richiede tutti i layer residenti (niente MEM_GB/MEM_FRAC/MICRO)\n"); exit(1); }
    if (g_kv_bits) { fprintf(stderr, "[train] richiede KV f32 (niente KV_BITS: lo stash legge m->K/V)\n"); exit(1); }
    for (int i = L0; i < m->c.n_layers; i++)
        if (m->L[i].gated) { fprintf(stderr, "[train] layer %d gated: non supportato nel range addestrato\n", i); exit(1); }
}

/* itera i 7 slot LoRA di un layer con le dimensioni base: FN(campo, nome, I, O) */
#define LORA_SLOTS(m, l, FN) do { \
    int D_ = (m)->c.hidden, IN_ = (m)->c.inter; \
    FN(q,    "self_attn.q_proj", D_, (l)->q.O); \
    FN(k,    "self_attn.k_proj", D_, (l)->k.O); \
    FN(v,    "self_attn.v_proj", D_, (l)->v.O); \
    FN(o,    "self_attn.o_proj", (l)->o.I, D_); \
    FN(gate, "mlp.gate_proj", D_, IN_); \
    FN(up,   "mlp.up_proj",   D_, IN_); \
    FN(down, "mlp.down_proj", IN_, D_); \
} while (0)

/* xorshift deterministico per l'init degli adattatori */
static uint64_t t_rng_s;
static float t_frnd(void) {
    t_rng_s ^= t_rng_s << 13; t_rng_s ^= t_rng_s >> 7; t_rng_s ^= t_rng_s << 17;
    return (float)((t_rng_s >> 11) * (1.0 / 9007199254740992.0)) * 2.f - 1.f;
}

/* init di uno slot per il training: A ~ U(-1,1)/sqrt(I), B = 0 (adapter no-op) */
static void lora_train_slot(Lora *lo, int r, float alpha, int I, int O, uint64_t seed) {
    lo->r = r; lo->alpha = alpha;
    lo->A = falloc((int64_t)r * I);
    lo->B = fzalloc((int64_t)O * r);
    t_rng_s = (0x5EEDULL + seed) * 6364136223846793005ULL + 1442695040888963407ULL;
    float sc = 1.f / sqrtf((float)I);
    for (int64_t i = 0; i < (int64_t)r * I; i++) lo->A[i] = t_frnd() * sc;
}

/* alloca gli adattatori sui layer [L0, n_layers) (+ lm_head se head != 0) */
static void lora_train_alloc(Model *m, int L0, int r, float alpha, int head) {
    for (int i = L0; i < m->c.n_layers; i++) {
        Layer *l = &m->L[i];
        l->lo = calloc(1, sizeof(LoraLayer));
        if (!l->lo) { fprintf(stderr, "[train] OOM adattatori\n"); exit(1); }
        int slot = 0;
        #define TF(fld, sub, I_, O_) do { \
            lora_train_slot(&l->lo->fld, r, alpha, (I_), (O_), (uint64_t)i * 7 + slot); slot++; } while (0)
        LORA_SLOTS(m, l, TF);
        #undef TF
        (void)slot;
    }
    if (head)
        lora_train_slot(&m->lm_lora, r, alpha, m->c.hidden, m->c.vocab, (uint64_t)m->c.n_layers * 7);
}

/* gradienti: struttura speculare agli adattatori, azzerata */
static LoraLayer *lora_grad_alloc(Model *m, int L0, Lora *gHead) {
    /* L0 <= n_layers per costruzione (train_main clampa); il max tiene il
     * range non-negativo visibile al compilatore */
    int nl = m->c.n_layers - L0; if (nl < 1) nl = 1;
    LoraLayer *gL = calloc(nl, sizeof(LoraLayer));
    for (int i = L0; i < m->c.n_layers; i++) {
        Layer *l = &m->L[i]; LoraLayer *gg = &gL[i - L0];
        #define GF(fld, sub, I_, O_) do { if (l->lo && l->lo->fld.r) { \
            gg->fld.r = l->lo->fld.r; \
            gg->fld.A = fzalloc((int64_t)gg->fld.r * (I_)); \
            gg->fld.B = fzalloc((int64_t)(O_) * gg->fld.r); } } while (0)
        LORA_SLOTS(m, l, GF);
        #undef GF
    }
    if (gHead && m->lm_lora.r) {
        gHead->r = m->lm_lora.r;
        gHead->A = fzalloc((int64_t)gHead->r * m->c.hidden);
        gHead->B = fzalloc((int64_t)m->c.vocab * gHead->r);
    }
    return gL;
}

/* lista piatta dei tensori addestrati (peso, gradiente, stato Adam): comoda
 * per azzerare i gradienti, fare lo step AdamW e il finite-difference nei test */
typedef struct { float *w, *g; Adam a; int64_t n; } TParam;
#define T_MAX_PARAMS 4096
static TParam t_par[T_MAX_PARAMS]; static int t_npar;

static void t_par_add(float *w, float *g, int64_t n) {
    if (t_npar >= T_MAX_PARAMS) { fprintf(stderr, "[train] troppi tensori adattatore\n"); exit(1); }
    TParam *p = &t_par[t_npar++];
    p->w = w; p->g = g; p->n = n;
    p->a.m = fzalloc(n); p->a.v = fzalloc(n);
}

static void t_params_build(Model *m, int L0, LoraLayer *gL, Lora *gHead) {
    t_npar = 0;
    for (int i = L0; i < m->c.n_layers; i++) {
        Layer *l = &m->L[i]; LoraLayer *gg = &gL[i - L0];
        #define PF(fld, sub, I_, O_) do { if (l->lo && l->lo->fld.r) { \
            t_par_add(l->lo->fld.A, gg->fld.A, (int64_t)l->lo->fld.r * (I_)); \
            t_par_add(l->lo->fld.B, gg->fld.B, (int64_t)(O_) * l->lo->fld.r); } } while (0)
        LORA_SLOTS(m, l, PF);
        #undef PF
    }
    if (gHead && m->lm_lora.r) {
        t_par_add(m->lm_lora.A, gHead->A, (int64_t)m->lm_lora.r * m->c.hidden);
        t_par_add(m->lm_lora.B, gHead->B, (int64_t)m->c.vocab * m->lm_lora.r);
    }
}

static void t_grads_zero(void) {
    for (int p = 0; p < t_npar; p++) memset(t_par[p].g, 0, (size_t)t_par[p].n * sizeof(float));
}

static void t_adamw_all(float lr, float wd, int t) {
    for (int p = 0; p < t_npar; p++)
        adamw_step(t_par[p].w, t_par[p].g, &t_par[p].a, t_par[p].n, lr, wd, t);
}

/* ---------- salvataggio adattatori (nomi identici a quelli di lora_load) ---------- */
static void lora_save(Model *m, int L0, float alpha, const char *path) {
    char nm[110];   /* < sizeof(StwT.name): niente troncamento */
    for (int i = L0; i < m->c.n_layers; i++) {
        Layer *l = &m->L[i];
        if (!l->lo) continue;
        #define SF(fld, sub, I_, O_) do { if (l->lo->fld.r) { \
            int64_t shA[2] = {l->lo->fld.r, (I_)}, shB[2] = {(O_), l->lo->fld.r}; \
            snprintf(nm, sizeof(nm), "lora.layers.%d." sub ".A", i); stw_add(nm, 2, shA, l->lo->fld.A); \
            snprintf(nm, sizeof(nm), "lora.layers.%d." sub ".B", i); stw_add(nm, 2, shB, l->lo->fld.B); } } while (0)
        LORA_SLOTS(m, l, SF);
        #undef SF
    }
    if (m->lm_lora.r) {
        int64_t shA[2] = {m->lm_lora.r, m->c.hidden}, shB[2] = {m->c.vocab, m->lm_lora.r};
        stw_add("lora.lm_head.A", 2, shA, m->lm_lora.A);
        stw_add("lora.lm_head.B", 2, shB, m->lm_lora.B);
    }
    int64_t s1[1] = {1};
    stw_add("lora.alpha", 1, s1, &alpha);
    stw_write(path);
}

/* ---------- TRAIN mode: fine-tuning LoRA su un corpus di testo ----------
 * Attivato da TRAIN=<corpus.txt> (vedi main di qwen.c). Env:
 *   TRAIN_CTX=512 TRAIN_STRIDE=CTX TRAIN_EPOCHS=1 TRAIN_LR=1e-4 TRAIN_WD=0
 *   TRAIN_CE_CHUNK=32 LORA_RANK=8 LORA_ALPHA=2*rank LORA_LAYERS=4 LORA_HEAD=0
 *   LORA_OUT=lora.safetensors; LORA=<file> = warm start dagli adattatori. */
static int train_main(int argc, char **argv) {
    (void)argc;
    omp_hot_tune(argv);
    const char *th_ = getenv("THREADS");
    if (th_ && atoi(th_) > 0) omp_set_num_threads(atoi(th_));
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    const char *corpus = getenv("TRAIN");
    if (!corpus || !*corpus) { fprintf(stderr, "set TRAIN=<corpus.txt>\n"); return 1; }
    int ctx     = getenv("TRAIN_CTX")    ? atoi(getenv("TRAIN_CTX"))    : 512;
    int stride  = getenv("TRAIN_STRIDE") ? atoi(getenv("TRAIN_STRIDE")) : ctx;
    int epochs  = getenv("TRAIN_EPOCHS") ? atoi(getenv("TRAIN_EPOCHS")) : 1;
    float lr    = getenv("TRAIN_LR") ? (float)atof(getenv("TRAIN_LR")) : 1e-4f;
    float wd    = getenv("TRAIN_WD") ? (float)atof(getenv("TRAIN_WD")) : 0.f;
    if (getenv("TRAIN_CE_CHUNK")) t_ce_chunk = atoi(getenv("TRAIN_CE_CHUNK"));
    int rank    = getenv("LORA_RANK")   ? atoi(getenv("LORA_RANK"))   : 8;
    float alpha = getenv("LORA_ALPHA")  ? (float)atof(getenv("LORA_ALPHA")) : 2.f*rank;
    int nadapt  = getenv("LORA_LAYERS") ? atoi(getenv("LORA_LAYERS")) : 4;
    int head    = getenv("LORA_HEAD")   ? atoi(getenv("LORA_HEAD"))   : 0;
    const char *out = getenv("LORA_OUT") ? getenv("LORA_OUT") : "lora.safetensors";
    if (ctx < 8 || stride < 1 || epochs < 1) { fprintf(stderr, "[train] TRAIN_CTX/STRIDE/EPOCHS invalidi\n"); return 1; }
    if (rank < 1 || rank > LORA_MAX_R) { fprintf(stderr, "[train] LORA_RANK fuori range [1,%d]\n", LORA_MAX_R); return 1; }
    Model m;
    model_init(&m, snap, 0);
    banner(&m);
    int L0 = m.c.n_layers - nadapt; if (L0 < 0) L0 = 0;
    train_guard(&m, L0);
    /* tokenizza l'intero corpus */
    char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
    Tok T; tok_load(&T, tokpath);
    FILE *f = fopen(corpus, "rb"); if (!f) { perror(corpus); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    char *txt = malloc(fsz + 1);
    if (!txt || fread(txt, 1, fsz, f) != (size_t)fsz) { fprintf(stderr, "[train] lettura corpus fallita\n"); return 1; }
    txt[fsz] = 0; fclose(f);
    int *ids = malloc(((int64_t)fsz + 16) * sizeof(int));
    int nids = tok_encode(&T, txt, (int)fsz, ids, (int)fsz + 16);
    free(txt);
    fprintf(stderr, "[train] corpus %s: %ld byte, %d token; finestre da %d (stride %d), %d epoche\n",
            corpus, fsz, nids, ctx, stride, epochs);
    if (nids < 8) { fprintf(stderr, "[train] corpus troppo corto (<8 token)\n"); return 1; }
    kv_alloc(&m, ctx);
    /* adattatori: warm start da LORA= se impostato, altrimenti init nuovo */
    if (getenv("LORA") && *getenv("LORA")) {
        lora_load(&m);
        for (int i = L0; i < m.c.n_layers; i++)
            if (!m.L[i].lo) { fprintf(stderr, "[train] warm start: manca l'adattatore del layer %d\n", i); return 1; }
        if (head && !m.lm_lora.r) { fprintf(stderr, "[train] warm start: manca lora.lm_head\n"); return 1; }
        if (m.L[L0].lo->q.r) alpha = m.L[L0].lo->q.alpha;
    } else {
        lora_train_alloc(&m, L0, rank, alpha, head);
    }
    Lora gHead; memset(&gHead, 0, sizeof(gHead));
    LoraLayer *gL = lora_grad_alloc(&m, L0, head ? &gHead : NULL);
    t_params_build(&m, L0, gL, head ? &gHead : NULL);
    LStash *st = malloc((m.c.n_layers - L0) * sizeof(LStash));
    for (int i = 0; i < m.c.n_layers - L0; i++) lstash_alloc(&st[i], &m.c, ctx);
    int wtot = 0;
    for (int off = 0; off + 8 <= nids; off += stride) wtot++;
    int adam_t = 0;
    for (int ep = 0; ep < epochs; ep++) {
        int wi = 0;
        for (int off = 0; off + 8 <= nids; off += stride) {   /* finestre <8 token saltate */
            int S = nids - off < ctx ? nids - off : ctx;
            m.kv_len = 0;
            double t0 = now_s();
            t_grads_zero();
            double loss = train_loss_and_backward(&m, ids + off, S, L0, st, gL, head ? &gHead : NULL);
            t_adamw_all(lr, wd, ++adam_t);
            double dt = now_s() - t0;
            fprintf(stderr, "[train] epoch %d finestra %d/%d loss %.4f (%.1f tok/s)\n",
                    ep + 1, ++wi, wtot, loss, S / (dt > 1e-9 ? dt : 1e-9));
        }
    }
    lora_save(&m, L0, alpha, out);
    fprintf(stderr, "[train] adattatori salvati in %s\n", out);
    return 0;
}

#endif /* QWEN_TRAIN_H */
