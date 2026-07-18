/* Motore di inferenza Qwen3 in C puro (denso: 0.6B..32B).
 * Derivato da olmoe.c: GQA + qk-norm PER TESTA + RoPE (theta 1e6) + SwiGLU denso.
 * Tutti i pesi residenti in RAM (f32, oppure int8 con QBITS=8).
 *
 * Uso (variabili d'ambiente, stile glm/olmoe):
 *   SNAP=<dir snapshot HF>            obbligatoria (config.json + tokenizer.json + *.safetensors)
 *   PROMPT="..."                      one-shot; senza PROMPT ne' REF -> chat interattiva su stdin
 *   NGEN=256 CTX=4096                 limiti di generazione/contesto
 *   TEMP=0.7 NUCLEUS=0.95 SEED=n      sampling (TEMP=0 -> greedy)
 *   CHAT_TEMPLATE=1 THINK=0           template chat Qwen3 (<|im_start|>...); THINK=0 chiude il blocco think
 *   QBITS=8                           quantizza i pesi grandi a int8 al load (~4x meno RAM)
 *   REF=ref.json                      validazione: greedy sui prompt_ids, confronto con full_ids
 *   TOKENS=1                          dump degli id generati su stderr
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "nn.h"
#include "st.h"
#include "tok.h"

#define ENGINE_TAG "qwen"
#define ENGINE_EOT "<|im_end|>\n"

/* tipi di layer (valori del config: layer_types). Attenzione alla polarita':
 * in qwen 1 = linear_attention, in gemma 1 = full_attention. */
enum { LT_FULL = 0, LT_LINEAR = 1 };

/* tetto sui head_dim della parte lineare: dimensiona i buffer su stack di
 * deltanet_token, garantito dal check in load_cfg */
#define MAX_LIN_DV 1024

/* ---------- config (config.json HF di Qwen3 / Qwen3.5) ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, inter, vocab, max_pos;
    float theta, eps;
    int tie_emb;
    int eos[4], n_eos;
    /* ibrido Qwen3.5 (lignaggio Qwen3-Next): layer linear_attention (Gated
     * DeltaNet) intervallati da full_attention (Gated Attention). */
    int hybrid;
    int rot;                    /* dimensioni ruotate dal RoPE (partial_rotary_factor*head_dim) */
    int lin_hv, lin_hk;         /* teste value / key della parte lineare */
    int lin_dk, lin_dv;         /* head_dim key / value della parte lineare */
    int lin_conv;               /* kernel della conv1d causale */
    int *ltype;                 /* [n_layers] LT_FULL / LT_LINEAR */
} Cfg;

typedef struct {
    int type;                              /* LT_FULL / LT_LINEAR */
    float *in_ln, *post_ln;
    /* full attention (anche gated: q_proj raddoppiato con il gate per testa) */
    int gated;
    float *qn, *kn;                        /* qn/kn: lunghezza head_dim (per testa) */
    Mat q, k, v, o;
    /* linear attention (Gated DeltaNet, proiezioni separate stile Qwen3.5) */
    Mat aqkv, az, ab, aa, aout;
    float *conv_w, *conv_b;                /* [conv_dim][K] depthwise (+bias opzionale) */
    float *dt_bias, *A_log;                /* [lin_hv] */
    float *dn_norm;                        /* [lin_dv] rmsnorm gated per testa */
    float *conv_state;                     /* [conv_dim * K] persistente */
    float *Sstate;                         /* [lin_hv * lin_dk * lin_dv] persistente */
    /* mlp (comune) */
    Mat gate, up, down;
} Layer;

typedef struct {
    Cfg c;
    shards S;
    int qbits;
    float *embed, *final_norm;
    Mat lm_head; int lm_tied;
    Layer *L;
    /* kv-cache per-layer: K,V come [n_kv_heads * max_t * head_dim] */
    float **K, **V; int kv_len, max_t;
    /* streaming a budget (MEM_GB/MEM_FRAC): i primi n_resident layer stanno in
     * RAM, gli altri vengono riletti dal disco a ogni step in stream_buf */
    int n_resident;
    float *stream_buf;
    double load_s;
} Model;

#include "runtime.h"

/* ---------- caricamento config ---------- */
static void load_cfg(Cfg *c, const char *snap) {
    char *arena;
    jval *r = cfg_slurp(snap, &arena);
    cfg_common(r, c);
    jval *th = json_get(r,"rope_theta");   c->theta = th ? (float)th->num : 1000000.f;
    jval *te = json_get(r,"tie_word_embeddings"); c->tie_emb = (te && te->t==J_BOOL) ? te->boolean : 0;
    /* --- parte ibrida (Qwen3.5): presente solo se il config la dichiara --- */
    jval *prf = json_get(r,"partial_rotary_factor");
    c->rot = prf ? (int)(c->head_dim * prf->num + 0.5) : c->head_dim;
    if (c->rot < 2 || c->rot > c->head_dim || (c->rot & 1)) {
        fprintf(stderr,"config: partial_rotary_factor incoerente (rot=%d, head_dim=%d)\n", c->rot, c->head_dim); exit(1);
    }
    jval *lv = json_get(r,"linear_num_value_heads");
    c->lin_hv  = lv ? (int)lv->num : 0;
    jval *lk = json_get(r,"linear_num_key_heads");   c->lin_hk  = lk ? (int)lk->num : 0;
    jval *ld = json_get(r,"linear_key_head_dim");    c->lin_dk  = ld ? (int)ld->num : 0;
    jval *le = json_get(r,"linear_value_head_dim");  c->lin_dv  = le ? (int)le->num : 0;
    jval *lc = json_get(r,"linear_conv_kernel_dim"); c->lin_conv= lc ? (int)lc->num : 4;
    c->ltype = calloc(c->n_layers, sizeof(int));
    jval *lt = json_get(r,"layer_types");
    if (lt && lt->t==J_ARR) {
        for (int i = 0; i < c->n_layers && i < lt->len; i++)
            c->ltype[i] = (lt->kids[i]->t==J_STR && !strcmp(lt->kids[i]->str,"linear_attention")) ? LT_LINEAR : LT_FULL;
    } else if (c->lin_hv > 0) {
        jval *fi = json_get(r,"full_attention_interval");
        int interval = fi ? (int)fi->num : 4;
        for (int i = 0; i < c->n_layers; i++) c->ltype[i] = ((i+1) % interval) ? LT_LINEAR : LT_FULL;
    }
    c->hybrid = 0;
    for (int i = 0; i < c->n_layers; i++) if (c->ltype[i] == LT_LINEAR) c->hybrid = 1;
    if (c->hybrid && (c->lin_hv<=0 || c->lin_hk<=0 || c->lin_dk<=0 || c->lin_dv<=0 ||
                      c->lin_dk>MAX_LIN_DV || c->lin_dv>MAX_LIN_DV ||
                      c->lin_hv % c->lin_hk || c->lin_conv<1 || c->lin_conv>8)) {
        fprintf(stderr,"config: parametri linear_attention mancanti o incoerenti\n"); exit(1);
    }
}

/* elenco delle MATRICI di un layer (unica fonte per loader, streamer e
 * prefetcher). Richiede l->type/l->gated gia' impostati. Ritorna il numero. */
static int layer_matrefs(Model *m, int li, MatRef *r) {
    Cfg *c = &m->c; Layer *l = &m->L[li];
    int n = 0, D = c->hidden, hd = c->head_dim, H = c->n_heads, KV = c->n_kv_heads;
    #define MR(field, fmt, O_, I_) do { r[n].mat=&l->field; \
        snprintf(r[n].name,sizeof(r[n].name),"model.layers.%d." fmt,li); \
        r[n].O=(O_); r[n].I=(I_); n++; } while(0)
    if (l->type == LT_FULL) {
        MR(q, "self_attn.q_proj.weight", l->gated ? 2*H*hd : H*hd, D);
        MR(k, "self_attn.k_proj.weight", KV*hd, D);
        MR(v, "self_attn.v_proj.weight", KV*hd, D);
        MR(o, "self_attn.o_proj.weight", D, H*hd);
    } else {
        int kd = c->lin_hk*c->lin_dk, vd = c->lin_hv*c->lin_dv, cd = 2*kd + vd;
        MR(aqkv, "linear_attn.in_proj_qkv.weight", cd, D);
        MR(az,   "linear_attn.in_proj_z.weight",   vd, D);
        MR(ab,   "linear_attn.in_proj_b.weight",   c->lin_hv, D);
        MR(aa,   "linear_attn.in_proj_a.weight",   c->lin_hv, D);
        MR(aout, "linear_attn.out_proj.weight",    D, vd);
    }
    MR(gate, "mlp.gate_proj.weight", c->inter, D);
    MR(up,   "mlp.up_proj.weight",   c->inter, D);
    MR(down, "mlp.down_proj.weight", D, c->inter);
    #undef MR
    return n;
}

/* parte piccola SEMPRE residente: norme, vettori e stati deltanet */
static void load_small(Model *m) {
    Cfg *c = &m->c;
    int D = c->hidden, hd = c->head_dim, H = c->n_heads;
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        l->type = c->ltype[i];
        #define LDT(field, suffix, n_) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm,n_)
        LDT(in_ln,  "input_layernorm.weight", D);
        LDT(post_ln,"post_attention_layernorm.weight", D);
        if (l->type == LT_FULL) {
            LDT(qn, "self_attn.q_norm.weight", hd);  /* per testa, NON per hidden */
            LDT(kn, "self_attn.k_norm.weight", hd);
            /* Qwen3.5: q_proj raddoppiato = [query|gate]. Rilevato dalla forma. */
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_proj.weight",i);
            l->gated = (st_numel(&m->S, nm) == (int64_t)2*H*hd*D);
        } else {
            int kd = c->lin_hk*c->lin_dk, vd = c->lin_hv*c->lin_dv;
            int cd = 2*kd + vd, K = c->lin_conv;
            LDT(conv_w,  "linear_attn.conv1d.weight", (int64_t)cd*K);   /* [cd,1,K] depthwise */
            snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn.conv1d.bias",i);
            l->conv_b = st_has(&m->S, nm) ? load_t(m, nm, cd) : NULL;
            LDT(dt_bias, "linear_attn.dt_bias", c->lin_hv);
            LDT(A_log,   "linear_attn.A_log",   c->lin_hv);
            LDT(dn_norm, "linear_attn.norm.weight", c->lin_dv);
            l->conv_state = falloc((int64_t)cd*K);
            l->Sstate = falloc((int64_t)c->lin_hv*c->lin_dk*c->lin_dv);
        }
        #undef LDT
    }
}

/* parte fissa del budget specifica del motore: KV-cache dei layer full */
static int64_t fixed_bytes(Model *m, int ctx) {
    Cfg *c = &m->c;
    int nfull = 0; for (int i = 0; i < c->n_layers; i++) if (c->ltype[i] == LT_FULL) nfull++;
    return (int64_t)nfull * 2 * c->n_kv_heads * ctx * c->head_dim * 4;
}

/* azzera gli stati ricorrenti dei layer lineari (inizio generazione / reset contesto) */
static void state_reset(Model *m) {
    Cfg *c = &m->c;
    if (!m->L || !c->hybrid) return;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (l->type != LT_LINEAR) continue;
        int cd = 2*c->lin_hk*c->lin_dk + c->lin_hv*c->lin_dv;
        memset(l->conv_state, 0, (int64_t)cd*c->lin_conv*sizeof(float));
        memset(l->Sstate, 0, (int64_t)c->lin_hv*c->lin_dk*c->lin_dv*sizeof(float));
    }
}

/* ---------- RoPE neox (half-split) su una testa a posizione assoluta pos.
 * rot < hd (partial rotary, Qwen3.5): ruotate solo le prime rot dimensioni,
 * accoppiate (j, j+rot/2), inv_freq calcolata su rot; il resto passa invariato. */
static void rope_head(float *x, int pos, float theta, int rot) {
    int h = rot / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(theta, -2.0f * j / rot);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* attenzione GQA sui token nuovi x[S,hidden]; pos_base = posizione del primo token nuovo.
 * Con l->gated (Qwen3.5): q_proj emette [query|gate] per testa; il contesto viene
 * moltiplicato per sigmoid(gate) prima di o_proj. */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c;
    int H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim;
    int G = H / KV;                       /* teste query per testa kv */
    int64_t qw = (int64_t)H*hd, kw = (int64_t)KV*hd;
    float *q, *gate = NULL;
    if (l->gated) {
        float *qg = falloc(S*2*qw);
        mat_apply(qg, x, &l->q, S);
        q = falloc(S*qw); gate = falloc(S*qw);
        for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
            memcpy(q    + s*qw + (int64_t)hh*hd, qg + s*2*qw + (int64_t)hh*2*hd,      hd*sizeof(float));
            memcpy(gate + s*qw + (int64_t)hh*hd, qg + s*2*qw + (int64_t)hh*2*hd + hd, hd*sizeof(float));
        }
        free(qg);
    } else {
        q = falloc(S*qw);
        mat_apply(q, x, &l->q, S);
    }
    float *k = falloc(S*kw), *vv = falloc(S*kw);
    mat_apply(k,  x, &l->k, S);
    mat_apply(vv, x, &l->v, S);
    /* qk-norm PER TESTA (Qwen3), poi RoPE per testa */
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) {
            float *qh = q + s*qw + (int64_t)hh*hd;
            rmsnorm_row(qh, qh, l->qn, hd, c->eps);
            rope_head(qh, pos, c->theta, c->rot);
        }
        for (int hh = 0; hh < KV; hh++) {
            float *kh = k + s*kw + (int64_t)hh*hd;
            rmsnorm_row(kh, kh, l->kn, hd, c->eps);
            rope_head(kh, pos, c->theta, c->rot);
        }
    }
    /* scrive k,v nella kv-cache alle posizioni pos_base..pos_base+S-1 */
    for (int s = 0; s < S; s++) for (int hh = 0; hh < KV; hh++) {
        int t = pos_base + s;
        memcpy(m->K[layer] + ((int64_t)hh*m->max_t + t)*hd, k + s*kw + (int64_t)hh*hd, hd*sizeof(float));
        memcpy(m->V[layer] + ((int64_t)hh*m->max_t + t)*hd, vv + s*kw + (int64_t)hh*hd, hd*sizeof(float));
    }
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc(S*qw);
    int Tk = pos_base + S;
    #pragma omp parallel
    {
        float *sc = falloc(Tk);
        #pragma omp for collapse(2) schedule(static)
        for (int hh = 0; hh < H; hh++) {
            for (int s = 0; s < S; s++) {
                int kvh = hh / G;                 /* GQA: testa kv condivisa */
                int qpos = pos_base + s;
                const float *qv = q + s*qw + (int64_t)hh*hd;
                for (int t = 0; t <= qpos; t++) {
                    const float *kr = m->K[layer] + ((int64_t)kvh*m->max_t + t)*hd;
                    float acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd]*kr[dd];
                    sc[t] = acc * scale;
                }
                softmax_row(sc, qpos+1);
                float *cx = ctx + s*qw + (int64_t)hh*hd;
                for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
                for (int t = 0; t <= qpos; t++) {
                    const float *vr = m->V[layer] + ((int64_t)kvh*m->max_t + t)*hd;
                    float a = sc[t];
                    for (int dd = 0; dd < hd; dd++) cx[dd] += a * vr[dd];
                }
            }
        }
        free(sc);
    }
    if (gate) {                            /* Qwen3.5: gating per-elemento sull'output */
        for (int64_t i = 0; i < S*qw; i++) ctx[i] *= 1.f/(1.f + expf(-gate[i]));
        free(gate);
    }
    mat_apply(out, ctx, &l->o, S);
    free(q); free(k); free(vv); free(ctx);
}

/* ---------- Gated DeltaNet (Qwen3.5, layer linear_attention) ----------
 * Ricorrenza per token (fp32), stato per layer:
 *   conv_state [cd][K]  finestra scorrevole della conv1d causale depthwise
 *   S [hv][dk][dv]      stato ricorrente (sostituisce la KV-cache: NON cresce)
 * Formule (transformers, torch_recurrent_gated_delta_rule + modular_qwen3_5):
 *   q,k l2-normalizzate per testa; q *= dk^-0.5
 *   g = -exp(A_log)*softplus(a+dt_bias); beta = sigmoid(b)
 *   S *= exp(g); kv = k.S; delta = (v-kv)*beta; S += k(x)delta; o = q.S
 *   out = rmsnorm(o)*w * silu(z) per testa -> out_proj */
static inline float softplusf(float x){ return x > 20.f ? x : log1pf(expf(x)); }
static inline float sigmoidf(float x){ return 1.f/(1.f + expf(-x)); }
static void l2norm_head(float *x, int n){
    double s = 0; for (int i = 0; i < n; i++) s += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)s + 1e-6f);
    for (int i = 0; i < n; i++) x[i] *= r;
}

/* un token: x[D] (gia' normato) -> out[D]; aggiorna gli stati del layer */
static void deltanet_token(Model *m, Layer *l, const float *x, float *out) {
    Cfg *c = &m->c;
    int Hv = c->lin_hv, Hk = c->lin_hk, dk = c->lin_dk, dv = c->lin_dv, K = c->lin_conv;
    int kd = Hk*dk, vd = Hv*dv, cd = 2*kd + vd, R = Hv/Hk;
    float *qkv = falloc(cd), *z = falloc(vd), *b = falloc(Hv), *a = falloc(Hv);
    mat_apply(qkv, x, &l->aqkv, 1);
    mat_apply(z,   x, &l->az,   1);
    mat_apply(b,   x, &l->ab,   1);
    mat_apply(a,   x, &l->aa,   1);
    /* conv1d causale depthwise sul solo qkv (z passa fuori), poi silu */
    #pragma omp parallel for schedule(static)
    for (int ch = 0; ch < cd; ch++) {
        float *cs = l->conv_state + (int64_t)ch*K;
        memmove(cs, cs+1, (K-1)*sizeof(float));
        cs[K-1] = qkv[ch];
        const float *w = l->conv_w + (int64_t)ch*K;
        float v = 0; for (int t = 0; t < K; t++) v += cs[t]*w[t];
        if (l->conv_b) v += l->conv_b[ch];
        qkv[ch] = v * sigmoidf(v);          /* silu */
    }
    float *q = qkv, *k = qkv + kd, *v = qkv + 2*kd;
    float qscale = 1.f / sqrtf((float)dk);
    for (int h = 0; h < Hk; h++) {
        l2norm_head(q + (int64_t)h*dk, dk);
        l2norm_head(k + (int64_t)h*dk, dk);
        for (int i = 0; i < dk; i++) q[(int64_t)h*dk + i] *= qscale;
    }
    float *o = falloc(vd);
    #pragma omp parallel for schedule(static)
    for (int hv = 0; hv < Hv; hv++) {
        int hk = hv / R;                    /* testa k condivisa (Hv/Hk teste v per testa k) */
        const float *qh = q + (int64_t)hk*dk, *kh = k + (int64_t)hk*dk, *vh = v + (int64_t)hv*dv;
        float *S = l->Sstate + (int64_t)hv*dk*dv;
        float g    = -expf(l->A_log[hv]) * softplusf(a[hv] + l->dt_bias[hv]);
        float beta = sigmoidf(b[hv]);
        float dec  = expf(g);
        float kv[MAX_LIN_DV], delta[MAX_LIN_DV]; /* dv <= MAX_LIN_DV garantito dal check in load_cfg */
        for (int j = 0; j < dv; j++) kv[j] = 0;
        for (int i = 0; i < dk; i++) {
            float *Si = S + (int64_t)i*dv;
            float ki = kh[i];
            for (int j = 0; j < dv; j++) { Si[j] *= dec; kv[j] += Si[j]*ki; }
        }
        for (int j = 0; j < dv; j++) delta[j] = (vh[j] - kv[j]) * beta;
        float *oh = o + (int64_t)hv*dv;
        for (int j = 0; j < dv; j++) oh[j] = 0;
        for (int i = 0; i < dk; i++) {
            float *Si = S + (int64_t)i*dv;
            float ki = kh[i], qi = qh[i];
            for (int j = 0; j < dv; j++) { Si[j] += ki*delta[j]; oh[j] += Si[j]*qi; }
        }
        /* rmsnorm gated per testa: norm(o)*w * silu(z) */
        double ms = 0; for (int j = 0; j < dv; j++) ms += (double)oh[j]*oh[j];
        float r = 1.f / sqrtf((float)(ms/dv) + c->eps);
        const float *zh = z + (int64_t)hv*dv;
        for (int j = 0; j < dv; j++) oh[j] = oh[j]*r*l->dn_norm[j] * (zh[j]*sigmoidf(zh[j]));
    }
    mat_apply(out, o, &l->aout, 1);
    free(qkv); free(z); free(b); free(a); free(o);
}

/* deltanet su S token in sequenza (prefill = ricorrenza per token) */
static void deltanet(Model *m, Layer *l, float *x, int S, float *out) {
    int D = m->c.hidden;
    for (int s = 0; s < S; s++)
        deltanet_token(m, l, x + (int64_t)s*D, out + (int64_t)s*D);
}

/* SwiGLU denso: out[S,D] = down( silu(gate(x)) * up(x) ), per-riga per limitare i buffer */
static void mlp(Model *m, Layer *l, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, I = c->inter;
    float *g = falloc(I), *u = falloc(I);
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s*D;
        mat_apply(g, xs, &l->gate, 1);
        mat_apply(u, xs, &l->up,   1);
        for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
        mat_apply(out + (int64_t)s*D, g, &l->down, 1);
    }
    free(g); free(u);
}

/* un passo: token nuovi ids[S] a posizione pos_base. Ritorna logits dell'ultimo token (malloc'd). */
static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    if (m->n_resident < c->n_layers) layer_prefetch(m, m->n_resident);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (i >= m->n_resident) {
            layer_stream_in(m, i);                  /* rilegge il layer dal disco (f32) */
            if (i + 1 < c->n_layers && i + 1 >= m->n_resident) layer_prefetch(m, i + 1);
        }
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        if (l->type == LT_LINEAR) deltanet(m, l, nrm, S, tmp);
        else attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        mlp(m, l, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->kv_len = pos_base + S;
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    mat_apply(logit, last, &m->lm_head, 1);
    free(x); free(nrm); free(tmp); free(last);
    return logit;
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    m->max_t = max_t; m->kv_len = 0;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        if (c->ltype[i] == LT_LINEAR) continue;  /* i layer lineari usano lo stato, non la KV */
        m->K[i] = falloc((int64_t)c->n_kv_heads * max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_kv_heads * max_t * c->head_dim);
    }
    state_reset(m);
}

/* costruisce il turno chat Qwen3 (ChatML). THINK=0 pre-chiude il blocco think. */
static int build_turn(char *buf, int cap, const char *user) {
    int think = getenv("THINK") ? atoi(getenv("THINK")) : 0;
    int bl = snprintf(buf, cap, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
    if (!think) bl += snprintf(buf+bl, cap-bl, "<think>\n\n</think>\n\n");
    return bl;
}

/* semina gli stop token del template ChatML */
static void stops_seed(Model *m, Tok *T) {
    (void)m;
    stop_add(tok_id_of(T, "<|im_end|>"));
    stop_add(tok_id_of(T, "<|endoftext|>"));
}

static void banner(Model *m) {
    int nlin = 0; for (int i = 0; i < m->c.n_layers; i++) nlin += (m->c.ltype[i] == LT_LINEAR);
    fprintf(stderr, "[qwen] %d layer (%d deltanet), hidden %d, %d/%d teste (hd %d, rot %d), inter %d, vocab %d%s | load %.1fs | RSS %.2f GB\n",
            m->c.n_layers, nlin, m->c.hidden, m->c.n_heads, m->c.n_kv_heads, m->c.head_dim, m->c.rot, m->c.inter, m->c.vocab,
            m->lm_tied ? " | lm_head=embed" : "", m->load_s, rss_gb());
}

#ifndef QWEN_TEST
int main(void) { return engine_main(); }
#endif /* QWEN_TEST */
