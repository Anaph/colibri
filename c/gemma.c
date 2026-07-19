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
 *   QBITS=8|4                      int8/int4 al load (embed/lm_head restano int8 con QBITS=4)
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
#include "gguf.h"
#include "tok.h"

#define ENGINE_TAG "gemma"
#define ENGINE_EOT "<end_of_turn>\n"

/* tipi di layer (valori del config: layer_types). Attenzione alla polarita':
 * in gemma 1 = full_attention, in qwen 1 = linear_attention. */
enum { LT_SLIDING = 0, LT_FULL = 1 };

/* tetto su hidden_size_per_layer_input: dimensiona il buffer su stack di
 * ple_inputs, garantito dal check in load_cfg */
#define MAX_PLE_DIM 1024

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
    int *ltype;                 /* [n_layers] LT_FULL / LT_SLIDING */
    int *kv_src;                /* [n_layers] layer sorgente del K/V (se' stesso se non condiviso) */
} Cfg;

typedef struct {
    int type;                   /* LT_FULL / LT_SLIDING */
    int shared_kv;              /* riusa K/V del layer kv_src (proiezioni k/v assenti) */
    float *in_ln, *post_attn_ln, *pre_ff_ln, *post_ff_ln;
    float *qn, *kn, *vn;        /* per testa, lunghezza hd del layer; vn opzionale */
    Mat q, k, v, o;
    Mat gate, up, down;
    /* PLE */
    Mat ple_gate, ple_proj;     /* per_layer_input_gate [ple,D], per_layer_projection [D,ple] */
    float *ple_norm;            /* post_per_layer_input_norm [D] */
} Layer;

typedef struct {
    MODEL_COMMON_FIELDS;                   /* contratto con runtime.h (nn.h) */
    /* PLE globali (i layer kv-shared ALIASANO anche K8/V8/Ks/Vs come K/V) */
    float *ple_embed;           /* [ple_vocab, n_layers*ple_dim] */
    Mat ple_model_proj;         /* [n_layers*ple_dim, D] */
    float *ple_proj_norm;       /* [n_layers*ple_dim]? VERIFY: norm su ple_dim */
} Model;

#include "runtime.h"

/* ---------- config ---------- */
static void load_cfg(Cfg *c, const char *snap) {
    jval *root; char *buf;
    jval *r = cfg_slurp(snap, &root, &buf);
    if (json_get(r,"enable_moe_block") && json_get(r,"enable_moe_block")->boolean) {
        fprintf(stderr,"gemma: config MoE (enable_moe_block) non supportato da questo motore\n"); exit(1);
    }
    cfg_common(r, c);
    jval *ghd = json_get(r,"global_head_dim"); c->ghd = ghd ? (int)ghd->num : c->head_dim;
    jval *gkv = json_get(r,"num_global_key_value_heads"); c->n_gkv = gkv ? (int)gkv->num : c->n_kv_heads;
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
    /* layer_types: espliciti, altrimenti pattern 5:1 (ogni 6o full) con ultimo full */
    c->ltype = calloc(c->n_layers, sizeof(int));
    jval *lt = json_get(r,"layer_types");
    if (lt && lt->t==J_ARR) {
        for (int i = 0; i < c->n_layers && i < lt->len; i++)
            c->ltype[i] = (lt->kids[i]->t==J_STR && !strcmp(lt->kids[i]->str,"full_attention")) ? LT_FULL : LT_SLIDING;
    } else {
        jval *pat = json_get(r,"sliding_window_pattern");
        int per = pat ? (int)pat->num : 6;
        for (int i = 0; i < c->n_layers; i++) c->ltype[i] = ((i+1) % per) ? LT_SLIDING : LT_FULL;
        c->ltype[c->n_layers-1] = LT_FULL;
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
    CKR("num_global_key_value_heads", c->n_gkv, 1, c->n_heads);
    CKR("global_head_dim",      c->ghd,        2, 1024);
    CKR("sliding_window",       c->window,     1, 1<<20);
    CKR("num_kv_shared_layers", c->n_kv_shared, 0, c->n_layers-1);
    if (c->ple_dim) CKR("hidden_size_per_layer_input", c->ple_dim, 1, MAX_PLE_DIM);
    if (c->n_heads % c->n_gkv) {
        fprintf(stderr,"config: n_heads non divisibile per le teste kv\n"); exit(1);
    }
    if (c->rot_angles < 1 || c->rot_angles > c->ghd/2) {
        fprintf(stderr,"config: partial_rotary_factor incoerente (rot_angles=%d, ghd=%d)\n", c->rot_angles, c->ghd); exit(1);
    }
    json_free(root); free(buf);       /* Cfg non trattiene puntatori nel JSON */
}

/* probe: primo nome esistente tra i candidati; se nessuno, li stampa ed esce.
 * Serve al bring-up sul checkpoint reale: i nomi VERIFY falliscono parlando. */
static const char *probe_name(Model *m, char *buf, int cap, int required, int n, ...) {
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

/* elenco delle MATRICI streamabili di un layer (richiede type/shared_kv gia'
 * impostati; k/v assenti sui layer kv-shared, v assente con k_eq_v). */
static int layer_matrefs(Model *m, int li, MatRef *r) {
    Cfg *c = &m->c; Layer *l = &m->L[li];
    int n = 0, D = c->hidden;
    int hd = l->type == LT_FULL ? c->ghd : c->head_dim;
    int KV = l->type == LT_FULL ? c->n_gkv : c->n_kv_heads;
    int H = c->n_heads;
    #define MR(field, fmt, O_, I_) do { r[n].mat=&l->field; \
        snprintf(r[n].name,sizeof(r[n].name),"model.layers.%d." fmt,li); \
        r[n].O=(O_); r[n].I=(I_); n++; } while(0)
    MR(q, "self_attn.q_proj.weight", H*hd, D);
    MR(o, "self_attn.o_proj.weight", D, H*hd);
    if (!l->shared_kv) {
        MR(k, "self_attn.k_proj.weight", KV*hd, D);
        char nm[128]; snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.v_proj.weight",li);
        if (st_has(&m->S, nm)) MR(v, "self_attn.v_proj.weight", KV*hd, D);
    }
    MR(gate, "mlp.gate_proj.weight", c->inter, D);
    MR(up,   "mlp.up_proj.weight",   c->inter, D);
    MR(down, "mlp.down_proj.weight", D, c->inter);
    if (c->ple_dim > 0) {
        MR(ple_gate, "per_layer_input_gate.weight", c->ple_dim, D);
        MR(ple_proj, "per_layer_projection.weight", D, c->ple_dim);
    }
    #undef MR
    return n;
}

/* parte piccola SEMPRE residente: PLE globale, norme + flag di struttura */
static void load_small(Model *m) {
    Cfg *c = &m->c;
    int D = c->hidden;
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
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        l->type = c->ltype[i];
        int hd = l->type == LT_FULL ? c->ghd : c->head_dim;
        #define LDT(field, suffix, n_) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm,n_)
        LDT(in_ln,        "input_layernorm.weight", D);
        LDT(post_attn_ln, "post_attention_layernorm.weight", D);
        LDT(pre_ff_ln,    "pre_feedforward_layernorm.weight", D);
        LDT(post_ff_ln,   "post_feedforward_layernorm.weight", D);
        LDT(qn, "self_attn.q_norm.weight", hd);
        LDT(kn, "self_attn.k_norm.weight", hd);
        snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.v_norm.weight",i);   /* nuovo in gemma4; opzionale */
        l->vn = st_has(&m->S, nm) ? load_t(m, nm, hd) : NULL;
        snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.k_proj.weight",i);
        l->shared_kv = (c->kv_src[i] != i) && !st_has(&m->S, nm);
        snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.v_proj.weight",i);
        if (!l->shared_kv && !st_has(&m->S, nm) && !c->k_eq_v) {
            fprintf(stderr, "gemma: %s assente e attention_k_eq_v=false\n", nm); exit(1);
        }
        if (c->ple_dim > 0) { LDT(ple_norm, "post_per_layer_input_norm.weight", D); }
        #undef LDT
    }
}

/* parte fissa del budget specifica del motore: KV per layer proprietario
 * (hd/KV per tipo) + tabelle PLE globali */
static int64_t fixed_bytes(Model *m, int ctx) {
    Cfg *c = &m->c;
    int64_t b = 0;
    if (c->ple_dim > 0) b += (int64_t)c->ple_vocab*c->n_layers*c->ple_dim*4
                           + (int64_t)c->n_layers*c->ple_dim*c->hidden*4;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->kv_src[i] != i) continue;
        int hd = c->ltype[i] == LT_FULL ? c->ghd : c->head_dim;
        int KV = c->ltype[i] == LT_FULL ? c->n_gkv : c->n_kv_heads;
        int64_t rows = (int64_t)2 * KV * ctx;
        b += g_kv_bits == 8 ? rows*hd + rows*4 : rows*hd*4;
    }
    return b;
}

/* nessuno stato ricorrente: reset contesto = solo kv_len (gestito dal chiamante) */
static void state_reset(Model *m) { (void)m; }

/* ---------- RMSNorm Gemma: peso (1+w) (zc) oppure w ---------- */
static void gnorm_row(const Cfg *c, float *out, const float *x, const float *w, int D) {
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
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c;
    int H = c->n_heads;
    int hd = l->type == LT_FULL ? c->ghd : c->head_dim;
    int KV = l->type == LT_FULL ? c->n_gkv : c->n_kv_heads;
    int G = H / KV;
    float theta = l->type == LT_FULL ? c->theta_g : c->theta_l;
    int rot = l->type == LT_FULL ? c->rot_angles : hd/2;        /* sliding: RoPE pieno */
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
            int64_t slot = (int64_t)hh*m->max_t + t;
            if (m->K8[layer]) {                  /* KV_BITS=8: K post-RoPE e V (anche la
                                                  * copia k_eq_v pre-RoPE) quantizzati
                                                  * indipendentemente, ognuno con la sua scala */
                kv_store_row(m->K8[layer] + slot*hd, &m->Ks[layer][slot], k + s*kw + (int64_t)hh*hd, hd);
                kv_store_row(m->V8[layer] + slot*hd, &m->Vs[layer][slot], vv + s*kw + (int64_t)hh*hd, hd);
            } else {
                memcpy(m->K[layer] + slot*hd, k + s*kw + (int64_t)hh*hd, hd*sizeof(float));
                memcpy(m->V[layer] + slot*hd, vv + s*kw + (int64_t)hh*hd, hd*sizeof(float));
            }
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
    const int8_t *K8c = m->K8[src], *V8c = m->V8[src];
    const float *Ksc = m->Ks[src], *Vsc = m->Vs[src];
    float scale = c->qscalar > 0 ? 1.f/sqrtf(c->qscalar) : 1.f/sqrtf((float)hd);
    float *ctx = falloc(S*qw);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            /* scratch pre-allocato per thread (att_sc, vedi kv_alloc): niente
             * malloc/free dentro la regione calda */
            float *sc = m->att_sc + (int64_t)omp_get_thread_num() * m->max_t;
            int kvh = hh / G;
            int qpos = pos_base + s;
            int t0 = l->type == LT_FULL ? 0 : (qpos - c->window + 1 > 0 ? qpos - c->window + 1 : 0);
            const float *qv = q + s*qw + (int64_t)hh*hd;
            int64_t kvbase = (int64_t)kvh * m->max_t;
            if (K8c) att_scores_i8(sc, qv, K8c, Ksc, kvbase, t0, qpos, hd, scale);
            else     att_scores_f32(sc, qv, Kc, kvbase, t0, qpos, hd, scale);
            softmax_row(sc, qpos-t0+1);
            float *cx = ctx + s*qw + (int64_t)hh*hd;
            if (V8c) att_accum_i8(cx, sc, V8c, Vsc, kvbase, t0, qpos, hd);
            else     att_accum_f32(cx, sc, Vc, kvbase, t0, qpos, hd);
        }
    }
    mat_apply(out, ctx, &l->o, S);
    free(q); free(ctx);
}

/* ---------- GeGLU: down( gelu_tanh(gate(x)) * up(x) ) ---------- */
static inline float gelu_tanh(float x) {
    return 0.5f*x*(1.f + tanhf(0.7978845608028654f*(x + 0.044715f*x*x*x)));
}
static void mlp(Model *m, Layer *l, float *x, int S, float *out) {
    /* in batch sugli S token: ogni riga di peso letta una volta per il batch */
    Cfg *c = &m->c; int I = c->inter;
    float *g = falloc((int64_t)S*I), *u = falloc((int64_t)S*I);
    mat_apply(g, x, &l->gate, S);
    mat_apply(u, x, &l->up,   S);
    for (int64_t i = 0; i < (int64_t)S*I; i++) g[i] = gelu_tanh(g[i]) * u[i];
    mat_apply(out, g, &l->down, S);
    free(g); free(u);
}

/* ---------- PLE: contributo per-layer nel residuo dopo il MLP ----------
 * combined[i] = (proj_ctx[i] + tok_embed[i]) / sqrt(2)   (per layer i)
 * poi in ogni layer: x += ple_proj( gelu(ple_gate(x)) * combined_i ), norm. */
static void ple_inputs(Model *m, const int *ids, int S, float *out /*[S, n_layers, ple]*/) {
    Cfg *c = &m->c; int P = c->ple_dim, NL = c->n_layers, D = c->hidden;
    float tok_scale = sqrtf((float)P);
    float inv_sqrt_d = 1.f/sqrtf((float)D), inv_sqrt2 = 1.f/sqrtf(2.f);
    float emb_scale = sqrtf((float)D);
    float *xemb = falloc(D), *proj = falloc((int64_t)NL*P);
    for (int s = 0; s < S; s++) {
        int id = ids[s] < c->ple_vocab ? ids[s] : 0;
        /* contesto: proiezione dell'embedding principale scalato */
        embed_row(m, id, emb_scale, xemb);
        mat_apply(proj, xemb, &m->ple_model_proj, 1);
        const float *pe = m->ple_embed + (int64_t)id*NL*P;
        float *os = out + (int64_t)s*NL*P;
        for (int li = 0; li < NL; li++) {
            float *op = os + (int64_t)li*P;
            const float *pp = proj + (int64_t)li*P;
            /* norma del contesto (VERIFY: norm su ple_dim per layer) */
            float tmp[MAX_PLE_DIM];
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

static void ple_apply(Model *m, Layer *l, int li, const float *ple /*[S,NL,P]*/, float *x, int S) {
    Cfg *c = &m->c; int D = c->hidden, P = c->ple_dim, NL = c->n_layers;
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
static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    float emb_scale = sqrtf((float)D);
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) embed_row(m, ids[s], emb_scale, x + (int64_t)s*D);
    float *ple = NULL;
    if (c->ple_dim > 0) {
        ple = falloc((int64_t)S*c->n_layers*c->ple_dim);
        ple_inputs(m, ids, S, ple);
    }
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    if (m->n_resident < c->n_layers) layer_prefetch(m, m->n_resident);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (i >= m->n_resident) {
            layer_stream_in(m, i);                  /* rilegge il layer dal disco (f32) */
            if (i + 1 < c->n_layers && i + 1 >= m->n_resident) layer_prefetch(m, i + 1);
        }
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
    /* blocco intermedio di un prefill a blocchi: niente final-norm/lm_head */
    if (g_skip_logits) { free(x); free(nrm); free(tmp); if (ple) free(ple); return NULL; }
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

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    kv_arrays_alloc(m, max_t);
    for (int i = 0; i < c->n_layers; i++) {
        if (c->kv_src[i] != i) continue;           /* i layer kv-shared leggono dal sorgente */
        int hd = c->ltype[i] == LT_FULL ? c->ghd : c->head_dim;
        int KV = c->ltype[i] == LT_FULL ? c->n_gkv : c->n_kv_heads;
        kv_layer_alloc(m, i, KV, hd, max_t);
    }
    for (int i = 0; i < c->n_layers; i++)
        if (c->kv_src[i] != i) {                   /* alias del sorgente, scale comprese */
            m->K[i] = m->K[c->kv_src[i]];   m->V[i] = m->V[c->kv_src[i]];
            m->K8[i] = m->K8[c->kv_src[i]]; m->V8[i] = m->V8[c->kv_src[i]];
            m->Ks[i] = m->Ks[c->kv_src[i]]; m->Vs[i] = m->Vs[c->kv_src[i]];
        }
    state_reset(m);   /* no-op per gemma: simmetria col gemello qwen */
}

/* costruisce il turno chat Gemma */
static int build_turn(char *buf, int cap, const char *user) {
    return snprintf(buf, cap, "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user);
}

/* semina gli stop token del template Gemma */
static void stops_seed(Model *m, Tok *T) {
    (void)m;
    stop_add(tok_id_of(T, "<end_of_turn>"));
}

static void banner(Model *m) {
    int nfull = 0; for (int i = 0; i < m->c.n_layers; i++) nfull += (m->c.ltype[i] == LT_FULL);
    fprintf(stderr, "[gemma] %d layer (%d full/%d sliding, finestra %d), hidden %d, %d teste (hd %d/%d, rot %d), vocab %d%s%s%s | load %.1fs | RSS %.2f GB | idot %s | f32 %s\n",
            m->c.n_layers, nfull, m->c.n_layers-nfull, m->c.window, m->c.hidden, m->c.n_heads,
            m->c.head_dim, m->c.ghd, m->c.rot_angles, m->c.vocab,
            m->lm_tied ? " | lm_head=embed" : "",
            m->c.n_kv_shared ? " | kv-shared" : "",
            m->c.ple_dim ? " | PLE" : "",
            m->load_s, rss_gb(), IDOT_KERNEL, F32_KERNEL);
}

#ifndef GEMMA_TEST
int main(int argc, char **argv) { return engine_main(argc, argv); }
#endif /* GEMMA_TEST */
