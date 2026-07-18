/* Motore di inferenza Gemma 4 in C puro (testo; il 12B-it multimodale
 * encoder-free viene eseguito ignorando le proiezioni di visione/audio).
 * Architettura (lignaggio Gemma 3/3n, verificata sui sorgenti transformers):
 *   - attenzione ibrida: sliding_attention (finestra locale, theta 10k, RoPE
 *     pieno) alternata a full_attention 5:1 (ultimo layer full; p-RoPE:
 *     inv_freq su TUTTO head_dim ma frequenze nulle oltre rope_angles =
 *     int(partial_rotary_factor*head_dim/2); theta 1e6; head_dim globale
 *     eventualmente diverso da quello locale)
 *   - q/k/v-norm RMS per testa; norme "sandwich" (input / post_attention /
 *     pre_feedforward / post_feedforward)
 *   - GeGLU (gelu_pytorch_tanh); embedding scalato per sqrt(hidden)
 *   - KV-sharing opzionale (num_kv_shared_layers: gli ultimi layer riusano il
 *     K/V dell'ultimo layer non condiviso dello stesso tipo)
 *   - K=V opzionale (attention_k_eq_v: v_proj assente, V = K pre-RoPE)
 *   - PLE opzionale (per-layer embeddings, hidden_size_per_layer_input>0)
 * I dettagli marcati VERIFY vanno confermati sul checkpoint reale: ogni
 * tensore incerto passa da un probe che stampa i candidati prima di uscire.
 *
 * Uso (variabili d'ambiente, stile qwen):
 *   SNAP=<dir snapshot HF>          obbligatoria
 *   PROMPT="..."                    one-shot; senza PROMPT ne' REF -> chat su stdin
 *   NGEN=256 CTX=4096 TEMP=0.7 NUCLEUS=0.95 SEED=n
 *   CHAT_TEMPLATE=1                 <start_of_turn>user ... <end_of_turn>
 *   QBITS=8                        int8 al load
 *   MEM_GB=f / MEM_FRAC=f          budget di residenza (vedi README)
 *   REF=ref.json TOKENS=1          validazione / dump id
 *   GEMMA_NORM_PLAIN=1             RMSNorm con peso "w" invece di "(1+w)" (VERIFY)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include "nn.h"
#include "st.h"
#include "tok.h"

/* ---------- config (config.json HF, text_config) ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, inter, vocab, max_pos;
    int ghd, n_gkv;             /* head_dim / teste kv dei layer full (global) */
    float eps;
    float theta_g, theta_l;     /* rope_theta (full) / rope_local_base_freq (sliding) */
    int window;                 /* sliding_window */
    int rot_angles;             /* p-RoPE: coppie ruotate sui layer full */
    float qscalar;              /* query_pre_attn_scalar (0 -> 1/sqrt(hd)) */
    float softcap;              /* final_logit_softcapping (0 -> off) */
    int tie_emb;
    int zc_norm;                /* 1: peso (1+w) stile Gemma3 (VERIFY) */
    int k_eq_v;                 /* attention_k_eq_v */
    int n_kv_shared;            /* num_kv_shared_layers */
    int ple_dim, ple_vocab;     /* hidden_size_per_layer_input / vocab per-layer */
    int eos[4], n_eos;
    int *ltype;                 /* [n_layers] 1=full_attention 0=sliding_attention */
    int *kv_src;                /* [n_layers] layer sorgente del K/V (se' stesso se non condiviso) */
} GCfg;

typedef struct {
    int type;                   /* 1=full 0=sliding */
    int shared_kv;              /* riusa K/V del layer kv_src (proiezioni k/v assenti) */
    float *in_ln, *post_attn_ln, *pre_ff_ln, *post_ff_ln;
    float *qn, *kn, *vn;        /* per testa, lunghezza hd del layer; vn opzionale */
    Mat q, k, v, o;
    Mat gate, up, down;
    /* PLE */
    Mat ple_gate, ple_proj;     /* per_layer_input_gate [ple,D], per_layer_projection [D,ple] */
    float *ple_norm;            /* post_per_layer_input_norm [D] */
} GLayer;

typedef struct {
    GCfg c;
    shards S;
    int qbits;
    float *embed, *final_norm;
    Mat lm_head; int lm_tied;
    GLayer *L;
    /* PLE globali */
    float *ple_embed;           /* [ple_vocab, n_layers*ple_dim] */
    Mat ple_model_proj;         /* [n_layers*ple_dim, D] */
    float *ple_proj_norm;       /* [n_layers*ple_dim]? VERIFY: norm su ple_dim */
    float **K, **V; int kv_len, max_t;
    double load_s;
} GModel;

/* ---------- config ---------- */
#define CKR(name, v, lo, hi) do { long _v=(long)(v); if(_v<(lo)||_v>(hi)){ \
    fprintf(stderr,"config.json: %s=%ld fuori range [%ld,%ld]\n",name,_v,(long)(lo),(long)(hi)); exit(1);} } while(0)

static void load_cfg(GCfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    jval *tc = json_get(r,"text_config"); if (tc && tc->t==J_OBJ) r = tc;
    if (json_get(r,"enable_moe_block") && json_get(r,"enable_moe_block")->boolean) {
        fprintf(stderr,"gemma: config MoE (enable_moe_block) non supportato da questo motore\n"); exit(1);
    }
    c->hidden    = (int)json_get(r,"hidden_size")->num;
    c->n_layers  = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads   = (int)json_get(r,"num_attention_heads")->num;
    c->n_kv_heads= (int)json_get(r,"num_key_value_heads")->num;
    c->inter     = (int)json_get(r,"intermediate_size")->num;
    c->vocab     = (int)json_get(r,"vocab_size")->num;
    jval *hd = json_get(r,"head_dim");
    c->head_dim  = hd ? (int)hd->num : c->hidden / c->n_heads;
    jval *ghd = json_get(r,"global_head_dim"); c->ghd = ghd ? (int)ghd->num : c->head_dim;
    jval *gkv = json_get(r,"num_global_key_value_heads"); c->n_gkv = gkv ? (int)gkv->num : c->n_kv_heads;
    jval *mp = json_get(r,"max_position_embeddings"); c->max_pos = mp ? (int)mp->num : 32768;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps = ep ? (float)ep->num : 1e-6f;
    jval *th = json_get(r,"rope_theta");   c->theta_g = th ? (float)th->num : 1000000.f;
    jval *tl = json_get(r,"rope_local_base_freq"); c->theta_l = tl ? (float)tl->num : 10000.f;
    jval *sw = json_get(r,"sliding_window"); c->window = sw ? (int)sw->num : 1024;
    /* p-RoPE: rope_angles = int(prf * ghd / 2); inv_freq calcolata su ghd intero.
     * VERIFY: rope_parameters puo' annidare partial_rotary_factor per layer_type. */
    float prf = 0.25f;
    jval *pf = json_get(r,"partial_rotary_factor"); if (pf) prf = (float)pf->num;
    jval *rp = json_get(r,"rope_parameters");
    if (rp && rp->t==J_OBJ) {
        jval *fa = json_get(rp,"full_attention");
        if (fa && fa->t==J_OBJ) {
            jval *p2 = json_get(fa,"partial_rotary_factor"); if (p2) prf = (float)p2->num;
            jval *t2 = json_get(fa,"rope_theta"); if (t2) c->theta_g = (float)t2->num;
        }
        jval *sa = json_get(rp,"sliding_attention");
        if (sa && sa->t==J_OBJ) { jval *t3 = json_get(sa,"rope_theta"); if (t3) c->theta_l = (float)t3->num; }
    }
    c->rot_angles = (int)(prf * c->ghd) / 2;
    jval *qs = json_get(r,"query_pre_attn_scalar"); c->qscalar = qs ? (float)qs->num : 0.f;
    jval *sc = json_get(r,"final_logit_softcapping"); c->softcap = (sc && sc->t==J_NUM) ? (float)sc->num : 0.f;
    jval *te = json_get(r,"tie_word_embeddings"); c->tie_emb = (te && te->t==J_BOOL) ? te->boolean : 1;
    jval *kv = json_get(r,"attention_k_eq_v"); c->k_eq_v = (kv && kv->t==J_BOOL) ? kv->boolean : 0;
    jval *ks = json_get(r,"num_kv_shared_layers"); c->n_kv_shared = ks ? (int)ks->num : 0;
    jval *pl = json_get(r,"hidden_size_per_layer_input"); c->ple_dim = pl ? (int)pl->num : 0;
    jval *pv = json_get(r,"vocab_size_per_layer_input"); c->ple_vocab = pv ? (int)pv->num : c->vocab;
    /* VERIFY: convenzione RMSNorm. Gemma3 usa (1+w); i sorgenti gemma4 citati
     * suggeriscono "w" puro. Default (1+w), override GEMMA_NORM_PLAIN=1. */
    c->zc_norm = !(getenv("GEMMA_NORM_PLAIN") && atoi(getenv("GEMMA_NORM_PLAIN")));
    c->n_eos = 0;
    jval *eo = json_get(r,"eos_token_id");
    if (eo) {
        if (eo->t==J_ARR) { for (int i=0;i<eo->len && c->n_eos<4;i++) c->eos[c->n_eos++]=(int)eo->kids[i]->num; }
        else c->eos[c->n_eos++]=(int)eo->num;
    }
    /* layer_types: espliciti, altrimenti pattern 5:1 (ogni 6o full) con ultimo full */
    c->ltype = calloc(c->n_layers, sizeof(int));
    jval *lt = json_get(r,"layer_types");
    if (lt && lt->t==J_ARR) {
        for (int i = 0; i < c->n_layers && i < lt->len; i++)
            c->ltype[i] = (lt->kids[i]->t==J_STR && !strcmp(lt->kids[i]->str,"full_attention")) ? 1 : 0;
    } else {
        jval *pat = json_get(r,"sliding_window_pattern");
        int per = pat ? (int)pat->num : 6;
        for (int i = 0; i < c->n_layers; i++) c->ltype[i] = ((i+1) % per) ? 0 : 1;
        c->ltype[c->n_layers-1] = 1;
    }
    /* KV-sharing: gli ultimi n_kv_shared layer prendono il K/V dell'ultimo layer
     * NON condiviso dello stesso tipo */
    c->kv_src = calloc(c->n_layers, sizeof(int));
    int first_shared = c->n_layers - c->n_kv_shared;
    for (int i = 0; i < c->n_layers; i++) {
        c->kv_src[i] = i;
        if (i >= first_shared) {
            for (int j = first_shared - 1; j >= 0; j--)
                if (c->ltype[j] == c->ltype[i]) { c->kv_src[i] = j; break; }
            if (c->kv_src[i] == i) { fprintf(stderr,"config: kv-shared layer %d senza sorgente\n", i); exit(1); }
        }
    }
    CKR("hidden_size",          c->hidden,     8, 65536);
    CKR("num_hidden_layers",    c->n_layers,   1, 256);
    CKR("num_attention_heads",  c->n_heads,    1, 256);
    CKR("num_key_value_heads",  c->n_kv_heads, 1, c->n_heads);
    CKR("num_global_key_value_heads", c->n_gkv, 1, c->n_heads);
    CKR("head_dim",             c->head_dim,   2, 1024);
    CKR("global_head_dim",      c->ghd,        2, 1024);
    CKR("intermediate_size",    c->inter,      8, 262144);
    CKR("vocab_size",           c->vocab,     16, 2000000);
    CKR("sliding_window",       c->window,     1, 1<<20);
    CKR("num_kv_shared_layers", c->n_kv_shared, 0, c->n_layers-1);
    if (c->ple_dim) CKR("hidden_size_per_layer_input", c->ple_dim, 1, 1024);
    if (c->n_heads % c->n_kv_heads || c->n_heads % c->n_gkv) {
        fprintf(stderr,"config: n_heads non divisibile per le teste kv\n"); exit(1);
    }
    if (c->rot_angles < 1 || c->rot_angles > c->ghd/2) {
        fprintf(stderr,"config: partial_rotary_factor incoerente (rot_angles=%d, ghd=%d)\n", c->rot_angles, c->ghd); exit(1);
    }
    free(buf); free(arena);
}

/* ---------- caricamento pesi ---------- */
static float *load_t(GModel *m, const char *name, int64_t expect) {
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

static void load_mat(GModel *m, Mat *w, const char *name, int O, int I) {
    w->O = O; w->I = I; w->q = NULL; w->qs = NULL;
    w->f = load_t(m, name, (int64_t)O*I);
    if (m->qbits == 8) {
        w->q = malloc((int64_t)O*I); w->qs = falloc(O);
        if (!w->q) { fprintf(stderr,"OOM quant %s\n",name); exit(1); }
        quantize_rows(w->f, w->q, w->qs, O, I, 8);
        free(w->f); w->f = NULL;
    }
}

/* probe: primo nome esistente tra i candidati; se nessuno, li stampa ed esce.
 * Serve al bring-up sul checkpoint reale: i nomi VERIFY falliscono parlando. */
static const char *probe_name(GModel *m, char *buf, int cap, int required, int n, ...) {
    va_list ap; va_start(ap, n);
    const char *cands[8]; int nc = 0;
    for (int i = 0; i < n && i < 8; i++) {
        const char *fmt = va_arg(ap, const char *);
        cands[nc++] = fmt;
        snprintf(buf, cap, "%s", fmt);
        if (st_has(&m->S, buf)) { va_end(ap); return buf; }
    }
    va_end(ap);
    if (!required) return NULL;
    fprintf(stderr, "gemma: nessuno dei tensori candidati esiste nel checkpoint:\n");
    for (int i = 0; i < nc; i++) fprintf(stderr, "  %s\n", cands[i]);
    exit(1);
}

static void model_init(GModel *m, const char *snap, int qbits) {
    memset(m, 0, sizeof(*m));
    m->qbits = qbits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    GCfg *c = &m->c;
    double t0 = now_s();
    int D = c->hidden;
    m->embed      = load_t(m, "model.embed_tokens.weight", (int64_t)c->vocab*D);
    m->final_norm = load_t(m, "model.norm.weight", D);
    if (c->tie_emb || !st_has(&m->S, "lm_head.weight")) {
        m->lm_tied = 1;
        m->lm_head.f = m->embed; m->lm_head.q=NULL; m->lm_head.qs=NULL;
        m->lm_head.O = c->vocab; m->lm_head.I = D;
    } else {
        load_mat(m, &m->lm_head, "lm_head.weight", c->vocab, D);
    }
    /* PLE globale (VERIFY nomi: lignaggio gemma-3n) */
    if (c->ple_dim > 0) {
        char nm[256];
        probe_name(m, nm, sizeof(nm), 1, 2,
                   "model.embed_tokens_per_layer.weight", "model.per_layer_embed_tokens.weight");
        m->ple_embed = load_t(m, nm, (int64_t)c->ple_vocab*c->n_layers*c->ple_dim);
        probe_name(m, nm, sizeof(nm), 1, 1, "model.per_layer_model_projection.weight");
        load_mat(m, &m->ple_model_proj, nm, c->n_layers*c->ple_dim, D);
        if (probe_name(m, nm, sizeof(nm), 0, 1, "model.per_layer_projection_norm.weight"))
            m->ple_proj_norm = load_t(m, nm, c->ple_dim);
        fprintf(stderr, "[gemma] PLE attivo: dim %d, vocab %d\n", c->ple_dim, c->ple_vocab);
    }
    m->L = calloc(c->n_layers, sizeof(GLayer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        GLayer *l = &m->L[i];
        l->type = c->ltype[i];
        int hd = l->type ? c->ghd : c->head_dim;
        int KV = l->type ? c->n_gkv : c->n_kv_heads;
        int H  = c->n_heads;
        #define LDT(field, suffix, n_) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm,n_)
        #define LDM(field, suffix, O_, I_) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); load_mat(m,&l->field,nm,O_,I_)
        LDT(in_ln,        "input_layernorm.weight", D);
        LDT(post_attn_ln, "post_attention_layernorm.weight", D);
        LDT(pre_ff_ln,    "pre_feedforward_layernorm.weight", D);
        LDT(post_ff_ln,   "post_feedforward_layernorm.weight", D);
        LDT(qn, "self_attn.q_norm.weight", hd);
        LDT(kn, "self_attn.k_norm.weight", hd);
        snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.v_norm.weight",i);   /* nuovo in gemma4; opzionale */
        l->vn = st_has(&m->S, nm) ? load_t(m, nm, hd) : NULL;
        LDM(q, "self_attn.q_proj.weight", H*hd,  D);
        LDM(o, "self_attn.o_proj.weight", D, H*hd);
        snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.k_proj.weight",i);
        l->shared_kv = (c->kv_src[i] != i) && !st_has(&m->S, nm);
        if (!l->shared_kv) {
            LDM(k, "self_attn.k_proj.weight", KV*hd, D);
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.v_proj.weight",i);
            if (st_has(&m->S, nm)) {
                load_mat(m, &l->v, nm, KV*hd, D);
            } else if (c->k_eq_v) {
                l->v.f = NULL; l->v.q = NULL;      /* V = K (pre-RoPE), nessuna proiezione */
            } else {
                fprintf(stderr, "gemma: %s assente e attention_k_eq_v=false\n", nm); exit(1);
            }
        }
        LDM(gate, "mlp.gate_proj.weight", c->inter, D);
        LDM(up,   "mlp.up_proj.weight",   c->inter, D);
        LDM(down, "mlp.down_proj.weight", D, c->inter);
        if (c->ple_dim > 0) {                       /* VERIFY nomi per-layer PLE */
            LDM(ple_gate, "per_layer_input_gate.weight", c->ple_dim, D);
            LDM(ple_proj, "per_layer_projection.weight", D, c->ple_dim);
            LDT(ple_norm, "post_per_layer_input_norm.weight", D);
        }
        #undef LDT
        #undef LDM
    }
    m->load_s = now_s() - t0;
}

/* ---------- RMSNorm Gemma: peso (1+w) (zc) oppure w ---------- */
static void gnorm_row(const GCfg *c, float *out, const float *x, const float *w, int D) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + c->eps);
    if (c->zc_norm) for (int i = 0; i < D; i++) out[i] = x[i] * r * (1.f + w[i]);
    else            for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

/* ---------- p-RoPE: coppie (j, j+hd/2); freq theta^(-2j/hd) per j<rot, oltre identita' ---------- */
static void gemma_rope_head(float *x, int pos, float theta, int hd, int rot_angles) {
    int h = hd / 2;
    for (int j = 0; j < rot_angles && j < h; j++) {
        float inv = powf(theta, -2.0f * j / hd);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* ---------- attenzione ibrida (GQA; sliding o full con p-RoPE) ---------- */
static void attention(GModel *m, GLayer *l, int layer, float *x, int S, int pos_base, float *out) {
    GCfg *c = &m->c;
    int H = c->n_heads;
    int hd = l->type ? c->ghd : c->head_dim;
    int KV = l->type ? c->n_gkv : c->n_kv_heads;
    int G = H / KV;
    float theta = l->type ? c->theta_g : c->theta_l;
    int rot = l->type ? c->rot_angles : hd/2;        /* sliding: RoPE pieno */
    int src = c->kv_src[layer];
    int64_t qw = (int64_t)H*hd, kw = (int64_t)KV*hd;
    float *q = falloc(S*qw);
    mat_apply(q, x, &l->q, S);
    if (!l->shared_kv) {
        float *k = falloc(S*kw), *vv = falloc(S*kw);
        mat_apply(k, x, &l->k, S);
        for (int s = 0; s < S; s++) for (int hh = 0; hh < KV; hh++) {
            float *kh = k + s*kw + (int64_t)hh*hd;
            gnorm_row(c, kh, kh, l->kn, hd);
        }
        if (l->v.f || l->v.q) {
            mat_apply(vv, x, &l->v, S);
            for (int s = 0; s < S; s++) for (int hh = 0; hh < KV; hh++) {
                float *vh = vv + s*kw + (int64_t)hh*hd;
                if (l->vn) gnorm_row(c, vh, vh, l->vn, hd);
            }
        } else {
            memcpy(vv, k, S*kw*sizeof(float));       /* k_eq_v: V = K normato, PRE-RoPE */
            if (l->vn) for (int s = 0; s < S; s++) for (int hh = 0; hh < KV; hh++) {
                float *vh = vv + s*kw + (int64_t)hh*hd;
                gnorm_row(c, vh, vh, l->vn, hd);
            }
        }
        for (int s = 0; s < S; s++) {
            int pos = pos_base + s;
            for (int hh = 0; hh < KV; hh++)
                gemma_rope_head(k + s*kw + (int64_t)hh*hd, pos, theta, hd, rot);
        }
        for (int s = 0; s < S; s++) for (int hh = 0; hh < KV; hh++) {
            int t = pos_base + s;
            memcpy(m->K[layer] + ((int64_t)hh*m->max_t + t)*hd, k + s*kw + (int64_t)hh*hd, hd*sizeof(float));
            memcpy(m->V[layer] + ((int64_t)hh*m->max_t + t)*hd, vv + s*kw + (int64_t)hh*hd, hd*sizeof(float));
        }
        free(k); free(vv);
    }
    /* q: norma per testa + rope */
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) {
            float *qh = q + s*qw + (int64_t)hh*hd;
            gnorm_row(c, qh, qh, l->qn, hd);
            gemma_rope_head(qh, pos, theta, hd, rot);
        }
    }
    float *Kc = m->K[src], *Vc = m->V[src];
    float scale = c->qscalar > 0 ? 1.f/sqrtf(c->qscalar) : 1.f/sqrtf((float)hd);
    float *ctx = falloc(S*qw);
    int Tk = pos_base + S;
    #pragma omp parallel
    {
        float *sc = falloc(Tk);
        #pragma omp for collapse(2) schedule(static)
        for (int hh = 0; hh < H; hh++) {
            for (int s = 0; s < S; s++) {
                int kvh = hh / G;
                int qpos = pos_base + s;
                int t0 = l->type ? 0 : (qpos - c->window + 1 > 0 ? qpos - c->window + 1 : 0);
                const float *qv = q + s*qw + (int64_t)hh*hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *kr = Kc + ((int64_t)kvh*m->max_t + t)*hd;
                    float acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd]*kr[dd];
                    sc[t-t0] = acc * scale;
                }
                softmax_row(sc, qpos-t0+1);
                float *cx = ctx + s*qw + (int64_t)hh*hd;
                for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
                for (int t = t0; t <= qpos; t++) {
                    const float *vr = Vc + ((int64_t)kvh*m->max_t + t)*hd;
                    float a = sc[t-t0];
                    for (int dd = 0; dd < hd; dd++) cx[dd] += a * vr[dd];
                }
            }
        }
        free(sc);
    }
    mat_apply(out, ctx, &l->o, S);
    free(q); free(ctx);
}

/* ---------- GeGLU: down( gelu_tanh(gate(x)) * up(x) ) ---------- */
static inline float gelu_tanh(float x) {
    return 0.5f*x*(1.f + tanhf(0.7978845608028654f*(x + 0.044715f*x*x*x)));
}
static void mlp(GModel *m, GLayer *l, float *x, int S, float *out) {
    GCfg *c = &m->c; int D = c->hidden, I = c->inter;
    float *g = falloc(I), *u = falloc(I);
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s*D;
        mat_apply(g, xs, &l->gate, 1);
        mat_apply(u, xs, &l->up,   1);
        for (int i = 0; i < I; i++) g[i] = gelu_tanh(g[i]) * u[i];
        mat_apply(out + (int64_t)s*D, g, &l->down, 1);
    }
    free(g); free(u);
}

/* ---------- PLE: contributo per-layer nel residuo dopo il MLP ----------
 * combined[i] = (proj_ctx[i] + tok_embed[i]) / sqrt(2)   (per layer i)
 * poi in ogni layer: x += ple_proj( gelu(ple_gate(x)) * combined_i ), norm. */
static void ple_inputs(GModel *m, const int *ids, int S, float *out /*[S, n_layers, ple]*/) {
    GCfg *c = &m->c; int P = c->ple_dim, NL = c->n_layers, D = c->hidden;
    float tok_scale = sqrtf((float)P);
    float inv_sqrt_d = 1.f/sqrtf((float)D), inv_sqrt2 = 1.f/sqrtf(2.f);
    float emb_scale = sqrtf((float)D);
    float *xemb = falloc(D), *proj = falloc((int64_t)NL*P);
    for (int s = 0; s < S; s++) {
        int id = ids[s] < c->ple_vocab ? ids[s] : 0;
        /* contesto: proiezione dell'embedding principale scalato */
        for (int i = 0; i < D; i++) xemb[i] = m->embed[(int64_t)id*D + i] * emb_scale;
        mat_apply(proj, xemb, &m->ple_model_proj, 1);
        const float *pe = m->ple_embed + (int64_t)id*NL*P;
        float *os = out + (int64_t)s*NL*P;
        for (int li = 0; li < NL; li++) {
            float *op = os + (int64_t)li*P;
            const float *pp = proj + (int64_t)li*P;
            /* norma del contesto (VERIFY: norm su ple_dim per layer) */
            float tmp[1024];
            for (int j = 0; j < P; j++) tmp[j] = pp[j]*inv_sqrt_d;
            if (m->ple_proj_norm) {
                double ms=0; for (int j=0;j<P;j++) ms += (double)tmp[j]*tmp[j];
                float r = 1.f/sqrtf((float)(ms/P) + c->eps);
                for (int j=0;j<P;j++) tmp[j] = tmp[j]*r*(c->zc_norm ? 1.f+m->ple_proj_norm[j] : m->ple_proj_norm[j]);
            }
            for (int j = 0; j < P; j++)
                op[j] = (tmp[j] + pe[(int64_t)li*P + j]*tok_scale) * inv_sqrt2;
        }
    }
    free(xemb); free(proj);
}

static void ple_apply(GModel *m, GLayer *l, int li, const float *ple /*[S,NL,P]*/, float *x, int S) {
    GCfg *c = &m->c; int D = c->hidden, P = c->ple_dim, NL = c->n_layers;
    float *g = falloc(P), *d = falloc(D), *nrm = falloc(D);
    for (int s = 0; s < S; s++) {
        float *xs = x + (int64_t)s*D;
        const float *pl_in = ple + ((int64_t)s*NL + li)*P;
        mat_apply(g, xs, &l->ple_gate, 1);
        for (int j = 0; j < P; j++) g[j] = gelu_tanh(g[j]) * pl_in[j];
        mat_apply(d, g, &l->ple_proj, 1);
        for (int i = 0; i < D; i++) d[i] += xs[i];               /* + residuo */
        gnorm_row(c, nrm, d, l->ple_norm, D);
        memcpy(xs, nrm, D*sizeof(float));
    }
    free(g); free(d); free(nrm);
}

/* ---------- un passo ---------- */
static float *step(GModel *m, const int *ids, int S, int pos_base) {
    GCfg *c = &m->c; int D = c->hidden;
    float emb_scale = sqrtf((float)D);
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++)
        for (int i = 0; i < D; i++)
            x[(int64_t)s*D + i] = m->embed[(int64_t)ids[s]*D + i] * emb_scale;
    float *ple = NULL;
    if (c->ple_dim > 0) {
        ple = falloc((int64_t)S*c->n_layers*c->ple_dim);
        ple_inputs(m, ids, S, ple);
    }
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        GLayer *l = &m->L[i];
        /* sandwich: x += post_attn_norm(attn(in_norm(x))) */
        for (int s = 0; s < S; s++) gnorm_row(c, nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D);
        attention(m, l, i, nrm, S, pos_base, tmp);
        for (int s = 0; s < S; s++) {
            gnorm_row(c, tmp + (int64_t)s*D, tmp + (int64_t)s*D, l->post_attn_ln, D);
            float *xs = x + (int64_t)s*D, *ts = tmp + (int64_t)s*D;
            for (int j = 0; j < D; j++) xs[j] += ts[j];
        }
        /* x += post_ff_norm(mlp(pre_ff_norm(x))) */
        for (int s = 0; s < S; s++) gnorm_row(c, nrm + (int64_t)s*D, x + (int64_t)s*D, l->pre_ff_ln, D);
        mlp(m, l, nrm, S, tmp);
        for (int s = 0; s < S; s++) {
            gnorm_row(c, tmp + (int64_t)s*D, tmp + (int64_t)s*D, l->post_ff_ln, D);
            float *xs = x + (int64_t)s*D, *ts = tmp + (int64_t)s*D;
            for (int j = 0; j < D; j++) xs[j] += ts[j];
        }
        if (c->ple_dim > 0) ple_apply(m, l, i, ple, x, S);
    }
    m->kv_len = pos_base + S;
    float *last = falloc(D);
    gnorm_row(c, last, x + (int64_t)(S-1)*D, m->final_norm, D);
    float *logit = falloc(c->vocab);
    mat_apply(logit, last, &m->lm_head, 1);
    if (c->softcap > 0)
        for (int i = 0; i < c->vocab; i++) logit[i] = c->softcap * tanhf(logit[i]/c->softcap);
    free(x); free(nrm); free(tmp); free(last);
    if (ple) free(ple);
    return logit;
}

static void kv_alloc(GModel *m, int max_t) {
    GCfg *c = &m->c;
    m->max_t = max_t; m->kv_len = 0;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        if (c->kv_src[i] != i) continue;           /* i layer kv-shared leggono dal sorgente */
        int hd = c->ltype[i] ? c->ghd : c->head_dim;
        int KV = c->ltype[i] ? c->n_gkv : c->n_kv_heads;
        m->K[i] = falloc((int64_t)KV * max_t * hd);
        m->V[i] = falloc((int64_t)KV * max_t * hd);
    }
    for (int i = 0; i < c->n_layers; i++)
        if (c->kv_src[i] != i) { m->K[i] = m->K[c->kv_src[i]]; m->V[i] = m->V[c->kv_src[i]]; }
}

/* ---------- generazione di un turno (identica al protocollo qwen) ---------- */
static int gen_turn(GModel *m, Tok *T, int *hist, int len, int k, int n_new, int echo, int *stopped) {
    int dump = getenv("TOKENS") && atoi(getenv("TOKENS"));
    double t0 = now_s();
    float *logit = step(m, hist + len, k, len);
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
    fprintf(stderr, "\n[gemma] prefill %d tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.2f tok/s) | RSS %.2f GB\n",
            k, tpre, k/(tpre>1e-9?tpre:1e-9), ng, tgen, ng/(tgen>1e-9?tgen:1e-9), rss_gb());
    return ng;
}

static int build_turn(char *buf, int cap, const char *user) {
    return snprintf(buf, cap, "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user);
}

/* ---------- ref.json (protocollo identico a qwen) ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a) { fprintf(stderr, "ref.json: manca %s\n", key); exit(1); }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

static int run_ref(GModel *m, const char *refpath) {
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
    g_temp = 0;
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
#ifndef GEMMA_TEST
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

    GModel m;
    model_init(&m, snap, qbits);
    int nfull = 0; for (int i = 0; i < m.c.n_layers; i++) nfull += m.c.ltype[i];
    fprintf(stderr, "[gemma] %d layer (%d full/%d sliding, finestra %d), hidden %d, %d teste (hd %d/%d, rot %d), vocab %d%s%s%s | load %.1fs | RSS %.2f GB\n",
            m.c.n_layers, nfull, m.c.n_layers-nfull, m.c.window, m.c.hidden, m.c.n_heads,
            m.c.head_dim, m.c.ghd, m.c.rot_angles, m.c.vocab,
            m.lm_tied ? " | lm_head=embed" : "",
            m.c.n_kv_shared ? " | kv-shared" : "",
            m.c.ple_dim ? " | PLE" : "",
            m.load_s, rss_gb());
    if (m.c.max_pos > 0 && maxctx > m.c.max_pos) maxctx = m.c.max_pos;

    const char *refpath = getenv("REF");
    if (refpath) return run_ref(&m, refpath);

    char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
    Tok T; tok_load(&T, tokpath);
    stop_add(tok_id_of(&T, "<end_of_turn>"));
    for (int i = 0; i < m.c.n_eos; i++) stop_add(m.c.eos[i]);
    fprintf(stderr, "[gemma] stop tokens:"); for (int i=0;i<g_nstop;i++) fprintf(stderr," %d",g_stop[i]); fprintf(stderr,"\n");

    kv_alloc(&m, maxctx);
    int *hist = malloc(maxctx * sizeof(int));
    char *buf = malloc(1<<16);

    const char *prompt = getenv("PROMPT");
    if (prompt) {
        int bl = templ ? build_turn(buf, 1<<16, prompt)
                       : snprintf(buf, 1<<16, "%s", prompt);
        int k = tok_encode(&T, buf, bl, hist, maxctx - 2);
        int cur = ngen; if (k + cur + 1 > maxctx) cur = maxctx - k - 1;
        int stopped;
        gen_turn(&m, &T, hist, 0, k, cur, 1, &stopped);
        printf("\n");
        return 0;
    }

    fprintf(stderr, "[gemma] chat interattiva: scrivi e premi invio (Ctrl-D per uscire)\n");
    int len = 0;
    char *line = NULL; size_t lcap = 0;
    for (;;) {
        fprintf(stderr, "\n> "); fflush(stderr);
        ssize_t nr = getline(&line, &lcap, stdin);
        if (nr < 0) break;
        while (nr > 0 && (line[nr-1]=='\n' || line[nr-1]=='\r')) line[--nr]=0;
        if (!nr) continue;
        int bl = templ ? build_turn(buf, 1<<16, line)
                       : snprintf(buf, 1<<16, "%s", line);
        int k = tok_encode(&T, buf, bl, hist + len, maxctx - len - 2);
        if (len + k + 8 >= maxctx) {
            fprintf(stderr, "[gemma] contesto pieno, reset della conversazione\n");
            len = 0; m.kv_len = 0;
            k = tok_encode(&T, buf, bl, hist, maxctx - 2);
        }
        int cur = ngen; if (len + k + cur + 1 > maxctx) cur = maxctx - len - k - 1;
        int stopped;
        int ng = gen_turn(&m, &T, hist, len, k, cur, 1, &stopped);
        len += k + ng;
        const char *suffix = stopped ? "\n" : "<end_of_turn>\n";
        len += tok_encode(&T, suffix, (int)strlen(suffix), hist + len, maxctx - len);
    }
    return 0;
}
#endif /* GEMMA_TEST */
