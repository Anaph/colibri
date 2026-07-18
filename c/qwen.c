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
    int *ltype;                 /* [n_layers] 0=full_attention 1=linear_attention */
} Cfg;

typedef struct {
    int type;                              /* 0=full_attention 1=linear_attention */
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
    double load_s;
} Model;

/* ---------- caricamento config ---------- */
#define CKR(name, v, lo, hi) do { long _v=(long)(v); if(_v<(lo)||_v>(hi)){ \
    fprintf(stderr,"config.json: %s=%ld fuori range [%ld,%ld]\n",name,_v,(long)(lo),(long)(hi)); exit(1);} } while(0)

static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    /* i rilasci multimodali annidano il config testo sotto text_config */
    jval *tc = json_get(r,"text_config"); if (tc && tc->t==J_OBJ) r = tc;
    c->hidden    = (int)json_get(r,"hidden_size")->num;
    c->n_layers  = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads   = (int)json_get(r,"num_attention_heads")->num;
    c->n_kv_heads= (int)json_get(r,"num_key_value_heads")->num;
    c->inter     = (int)json_get(r,"intermediate_size")->num;
    c->vocab     = (int)json_get(r,"vocab_size")->num;
    jval *hd = json_get(r,"head_dim");
    c->head_dim  = hd ? (int)hd->num : c->hidden / c->n_heads;   /* Qwen3 lo dichiara esplicito */
    jval *mp = json_get(r,"max_position_embeddings"); c->max_pos = mp ? (int)mp->num : 32768;
    jval *th = json_get(r,"rope_theta");   c->theta = th ? (float)th->num : 1000000.f;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-6f;
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
            c->ltype[i] = (lt->kids[i]->t==J_STR && !strcmp(lt->kids[i]->str,"linear_attention")) ? 1 : 0;
    } else if (c->lin_hv > 0) {
        jval *fi = json_get(r,"full_attention_interval");
        int interval = fi ? (int)fi->num : 4;
        for (int i = 0; i < c->n_layers; i++) c->ltype[i] = ((i+1) % interval) ? 1 : 0;
    }
    c->hybrid = 0;
    for (int i = 0; i < c->n_layers; i++) if (c->ltype[i]) c->hybrid = 1;
    if (c->hybrid && (c->lin_hv<=0 || c->lin_hk<=0 || c->lin_dk<=0 || c->lin_dv<=0 ||
                      c->lin_dk>1024 || c->lin_dv>1024 ||
                      c->lin_hv % c->lin_hk || c->lin_conv<1 || c->lin_conv>8)) {
        fprintf(stderr,"config: parametri linear_attention mancanti o incoerenti\n"); exit(1);
    }
    c->n_eos = 0;
    jval *eo = json_get(r,"eos_token_id");
    if (eo) {
        if (eo->t==J_ARR) { for (int i=0;i<eo->len && c->n_eos<4;i++) c->eos[c->n_eos++]=(int)eo->kids[i]->num; }
        else c->eos[c->n_eos++]=(int)eo->num;
    }
    CKR("hidden_size",          c->hidden,     8, 65536);
    CKR("num_hidden_layers",    c->n_layers,   1, 256);
    CKR("num_attention_heads",  c->n_heads,    1, 256);
    CKR("num_key_value_heads",  c->n_kv_heads, 1, c->n_heads);
    CKR("head_dim",             c->head_dim,   2, 1024);
    CKR("intermediate_size",    c->inter,      8, 262144);
    CKR("vocab_size",           c->vocab,     16, 2000000);
    if (c->n_heads % c->n_kv_heads) { fprintf(stderr,"config: n_heads %% n_kv_heads != 0\n"); exit(1); }
    free(buf); free(arena);
}

/* ---------- caricamento pesi ---------- */
static float *load_t(Model *m, const char *name, int64_t expect) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    if (expect > 0 && n != expect) {
        fprintf(stderr, "tensor %s: numel %lld != atteso %lld (layout diverso?)\n",
                name, (long long)n, (long long)expect);
        exit(1);
    }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

/* carica [O,I]; con QBITS=8 tiene solo int8+scala e libera l'f32 */
static void load_mat(Model *m, Mat *w, const char *name, int O, int I) {
    w->O = O; w->I = I; w->q = NULL; w->qs = NULL;
    w->f = load_t(m, name, (int64_t)O*I);
    if (m->qbits == 8) {
        w->q = malloc((int64_t)O*I); w->qs = falloc(O);
        if (!w->q) { fprintf(stderr,"OOM quant %s\n",name); exit(1); }
        quantize_rows(w->f, w->q, w->qs, O, I, 8);
        free(w->f); w->f = NULL;
    }
}

static void model_init(Model *m, const char *snap, int qbits) {
    memset(m, 0, sizeof(*m));
    m->qbits = qbits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    m->embed      = load_t(m, "model.embed_tokens.weight", (int64_t)c->vocab*c->hidden);
    m->final_norm = load_t(m, "model.norm.weight", c->hidden);
    if (c->tie_emb || !st_has(&m->S, "lm_head.weight")) {
        m->lm_tied = 1;
        m->lm_head.f = m->embed; m->lm_head.q=NULL; m->lm_head.qs=NULL;
        m->lm_head.O = c->vocab; m->lm_head.I = c->hidden;
    } else {
        load_mat(m, &m->lm_head, "lm_head.weight", c->vocab, c->hidden);
    }
    int D = c->hidden, hd = c->head_dim, H = c->n_heads, KV = c->n_kv_heads;
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        l->type = c->ltype[i];
        #define LDT(field, suffix, n) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm,n)
        #define LDM(field, suffix, O_, I_) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); load_mat(m,&l->field,nm,O_,I_)
        LDT(in_ln,  "input_layernorm.weight", D);
        LDT(post_ln,"post_attention_layernorm.weight", D);
        if (l->type == 0) {                          /* full attention (Qwen3 o gated Qwen3.5) */
            LDT(qn, "self_attn.q_norm.weight", hd);  /* per testa, NON per hidden */
            LDT(kn, "self_attn.k_norm.weight", hd);
            /* Qwen3.5: q_proj raddoppiato = [query|gate] per testa. Rilevato dalla forma. */
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_proj.weight",i);
            int64_t qn_ = st_numel(&m->S, nm);
            l->gated = (qn_ == (int64_t)2*H*hd*D);
            load_mat(m, &l->q, nm, l->gated ? 2*H*hd : H*hd, D);
            LDM(k, "self_attn.k_proj.weight", KV*hd, D);
            LDM(v, "self_attn.v_proj.weight", KV*hd, D);
            LDM(o, "self_attn.o_proj.weight", D, H*hd);
        } else {                                     /* linear attention (Gated DeltaNet) */
            int kd = c->lin_hk*c->lin_dk, vd = c->lin_hv*c->lin_dv;
            int cd = 2*kd + vd, K = c->lin_conv;
            LDM(aqkv, "linear_attn.in_proj_qkv.weight", cd, D);
            LDM(az,   "linear_attn.in_proj_z.weight",   vd, D);
            LDM(ab,   "linear_attn.in_proj_b.weight",   c->lin_hv, D);
            LDM(aa,   "linear_attn.in_proj_a.weight",   c->lin_hv, D);
            LDM(aout, "linear_attn.out_proj.weight",    D, vd);
            LDT(conv_w,  "linear_attn.conv1d.weight", (int64_t)cd*K);   /* [cd,1,K] depthwise */
            snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn.conv1d.bias",i);
            l->conv_b = st_has(&m->S, nm) ? load_t(m, nm, cd) : NULL;
            LDT(dt_bias, "linear_attn.dt_bias", c->lin_hv);
            LDT(A_log,   "linear_attn.A_log",   c->lin_hv);
            LDT(dn_norm, "linear_attn.norm.weight", c->lin_dv);
            l->conv_state = falloc((int64_t)cd*K);
            l->Sstate = falloc((int64_t)c->lin_hv*c->lin_dk*c->lin_dv);
        }
        LDM(gate, "mlp.gate_proj.weight", c->inter, D);
        LDM(up,   "mlp.up_proj.weight",   c->inter, D);
        LDM(down, "mlp.down_proj.weight", D, c->inter);
        #undef LDT
        #undef LDM
    }
    m->load_s = now_s() - t0;
}

/* azzera gli stati ricorrenti dei layer lineari (inizio generazione / reset contesto) */
static void state_reset(Model *m) {
    Cfg *c = &m->c;
    if (!m->L || !c->hybrid) return;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (l->type != 1) continue;
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
        float kv[1024], delta[1024];        /* dv <= 1024 garantito dal CKR */
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
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        if (l->type == 1) deltanet(m, l, nrm, S, tmp);
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
        if (c->ltype[i]) continue;         /* i layer lineari usano lo stato, non la KV */
        m->K[i] = falloc((int64_t)c->n_kv_heads * max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_kv_heads * max_t * c->head_dim);
    }
    state_reset(m);
}

/* ---------- generazione di un turno ----------
 * hist[len..len+k) = token nuovi da prefillare; genera fino a n_new token o stop.
 * Stampa il testo su stdout se echo!=0. Ritorna il numero di token generati
 * (stop incluso se emesso); *stopped=1 se l'ultimo token e' uno stop. */
static int gen_turn(Model *m, Tok *T, int *hist, int len, int k, int n_new, int echo, int *stopped) {
    int dump = getenv("TOKENS") && atoi(getenv("TOKENS"));
    double t0 = now_s();
    float *logit = step(m, hist + len, k, len);      /* PREFILL */
    double tpre = now_s() - t0;
    int base = len + k, ng = 0; *stopped = 0;
    t0 = now_s();
    for (int s = 0; s < n_new; s++) {
        int t = pick_tok(logit, m->c.vocab);
        free(logit); logit = NULL;
        hist[base + ng++] = t;
        if (dump) fprintf(stderr, "%d ", t);
        if (is_stop(t)) { *stopped = 1; break; }
        if (echo) {
            char buf[64]; int bn = tok_decode(T, &t, 1, buf, 63);
            fwrite(buf, 1, bn, stdout); fflush(stdout);
        }
        if (s == n_new - 1) break;
        logit = step(m, &hist[base + ng - 1], 1, base + ng - 1);
    }
    if (logit) free(logit);
    if (dump) fprintf(stderr, "\n");
    double tgen = now_s() - t0;
    fprintf(stderr, "\n[qwen] prefill %d tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.2f tok/s) | RSS %.2f GB\n",
            k, tpre, k/(tpre>1e-9?tpre:1e-9), ng, tgen, ng/(tgen>1e-9?tgen:1e-9), rss_gb());
    return ng;
}

/* costruisce il turno chat Qwen3 (ChatML). think=0 pre-chiude il blocco think. */
static int build_turn(char *buf, int cap, const char *user, int think) {
    int bl = snprintf(buf, cap, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
    if (!think) bl += snprintf(buf+bl, cap-bl, "<think>\n\n</think>\n\n");
    return bl;
}

/* ---------- ref.json (validazione) ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a) { fprintf(stderr, "ref.json: manca %s\n", key); exit(1); }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

static int run_ref(Model *m, const char *refpath) {
    FILE *f = fopen(refpath, "rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull;
    int *prompt = read_int_array(ref,"prompt_ids",&np);
    int *full   = read_int_array(ref,"full_ids",&nfull);
    int n_new = nfull - np;
    if (n_new <= 0) { fprintf(stderr,"ref.json: full_ids non estende prompt_ids\n"); return 1; }
    kv_alloc(m, nfull + 1);
    int *out = malloc(nfull * sizeof(int));
    memcpy(out, prompt, np*sizeof(int));
    g_temp = 0;                                    /* la validazione e' greedy */
    double t0 = now_s();
    float *logit = step(m, prompt, np, 0);
    int len = np;
    for (int s = 0; s < n_new; s++) {
        int best = argmax_v(logit, m->c.vocab);
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        logit = step(m, &out[len-1], 1, len-1);
    }
    double dt = now_s() - t0;
    int match = 0;
    printf("Reference: "); for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : "); for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    printf("Speed: %.2f tok/s | PEAK RSS %.2f GB\n", n_new/dt, rss_gb());
    free(buf); free(arena); free(prompt); free(full); free(out);
    return match == n_new ? 0 : 2;
}

/* ---------- main ---------- */
#ifndef QWEN_TEST
int main(void) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    int qbits = getenv("QBITS") ? atoi(getenv("QBITS")) : 0;
    if (qbits != 0 && qbits != 8) { fprintf(stderr, "QBITS deve essere 0 (f32) o 8 (int8)\n"); return 1; }
    int ngen  = getenv("NGEN") ? atoi(getenv("NGEN")) : 256;
    int maxctx= getenv("CTX")  ? atoi(getenv("CTX"))  : 4096;
    if (getenv("TEMP"))    g_temp = (float)atof(getenv("TEMP"));
    if (getenv("NUCLEUS")) g_nuc  = (float)atof(getenv("NUCLEUS"));
    if (getenv("SEED"))    g_rng  = (uint64_t)strtoull(getenv("SEED"),NULL,10) | 1u;
    int templ = getenv("CHAT_TEMPLATE") ? atoi(getenv("CHAT_TEMPLATE")) : 1;
    int think = getenv("THINK") ? atoi(getenv("THINK")) : 0;

    Model m;
    model_init(&m, snap, qbits);
    int nlin = 0; for (int i = 0; i < m.c.n_layers; i++) nlin += m.c.ltype[i];
    fprintf(stderr, "[qwen] %d layer (%d deltanet), hidden %d, %d/%d teste (hd %d, rot %d), inter %d, vocab %d%s | load %.1fs | RSS %.2f GB\n",
            m.c.n_layers, nlin, m.c.hidden, m.c.n_heads, m.c.n_kv_heads, m.c.head_dim, m.c.rot, m.c.inter, m.c.vocab,
            m.lm_tied ? " | lm_head=embed" : "", m.load_s, rss_gb());
    if (m.c.max_pos > 0 && maxctx > m.c.max_pos) maxctx = m.c.max_pos;

    const char *refpath = getenv("REF");
    if (refpath) return run_ref(&m, refpath);

    char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
    Tok T; tok_load(&T, tokpath);
    stop_add(tok_id_of(&T, "<|im_end|>"));
    stop_add(tok_id_of(&T, "<|endoftext|>"));
    for (int i = 0; i < m.c.n_eos; i++) stop_add(m.c.eos[i]);
    fprintf(stderr, "[qwen] stop tokens:"); for (int i=0;i<g_nstop;i++) fprintf(stderr," %d",g_stop[i]); fprintf(stderr,"\n");

    kv_alloc(&m, maxctx);
    int *hist = malloc(maxctx * sizeof(int));
    char *buf = malloc(1<<16);

    const char *prompt = getenv("PROMPT");
    if (prompt) {                                   /* one-shot */
        int bl = templ ? build_turn(buf, 1<<16, prompt, think)
                       : snprintf(buf, 1<<16, "%s", prompt);
        int k = tok_encode(&T, buf, bl, hist, maxctx - 2);
        int cur = ngen; if (k + cur + 1 > maxctx) cur = maxctx - k - 1;
        int stopped;
        gen_turn(&m, &T, hist, 0, k, cur, 1, &stopped);
        printf("\n");
        return 0;
    }

    /* chat interattiva: KV persistente, storia append-only */
    fprintf(stderr, "[qwen] chat interattiva: scrivi e premi invio (Ctrl-D per uscire)\n");
    int len = 0;                                    /* token gia' in KV */
    char *line = NULL; size_t lcap = 0;
    for (;;) {
        fprintf(stderr, "\n> "); fflush(stderr);
        ssize_t nr = getline(&line, &lcap, stdin);
        if (nr < 0) break;
        while (nr > 0 && (line[nr-1]=='\n' || line[nr-1]=='\r')) line[--nr]=0;
        if (!nr) continue;
        int bl = templ ? build_turn(buf, 1<<16, line, think)
                       : snprintf(buf, 1<<16, "%s", line);
        int k = tok_encode(&T, buf, bl, hist + len, maxctx - len - 2);
        if (len + k + 8 >= maxctx) {                /* contesto pieno: reset conversazione */
            fprintf(stderr, "[qwen] contesto pieno, reset della conversazione\n");
            len = 0; m.kv_len = 0; state_reset(&m);  /* lo stato ricorrente non e' troncabile */
            k = tok_encode(&T, buf, bl, hist, maxctx - 2);
        }
        int cur = ngen; if (len + k + cur + 1 > maxctx) cur = maxctx - len - k - 1;
        int stopped;
        int ng = gen_turn(&m, &T, hist, len, k, cur, 1, &stopped);
        len += k + ng;
        /* chiude il blocco assistant nel transcript: i token del suffisso entrano
         * in KV col prefill del turno successivo */
        const char *suffix = stopped ? "\n" : "<|im_end|>\n";
        len += tok_encode(&T, suffix, (int)strlen(suffix), hist + len, maxctx - len);
    }
    return 0;
}
#endif /* QWEN_TEST */
