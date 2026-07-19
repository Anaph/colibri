/* Impalcatura condivisa dei motori densi (qwen.c, gemma.c): caricamento
 * config/pesi, streaming dei layer a budget (MEM_GB/MEM_FRAC), validazione
 * REF, loop di generazione e main.
 *
 * Meccanismo a hook: ogni motore, PRIMA di includere questo header, definisce
 * ENGINE_TAG / ENGINE_EOT e i typedef completi di Cfg/Layer/Model; DOPO
 * l'include implementa gli hook statici prototipati qui sotto (load_cfg,
 * load_small, layer_matrefs, fixed_bytes, step, kv_alloc, state_reset,
 * build_turn, stops_seed, banner). Un hook mancante fa fallire la build
 * rumorosamente. Tutto static, un'istanza per translation unit. */
#ifndef RUNTIME_H
#define RUNTIME_H

/* elenco delle MATRICI streamabili di un layer (unica fonte per loader,
 * streamer e prefetcher); riempito dall'hook layer_matrefs del motore. */
typedef struct { Mat *mat; char name[96]; int O, I; } MatRef;
#define MAX_LAYER_MATS 16

/* ---------- hook del motore (implementati dopo l'include) ---------- */
static void load_cfg(Cfg *c, const char *snap);
static void load_small(Model *m);
static int layer_matrefs(Model *m, int li, MatRef *r);
static int64_t fixed_bytes(Model *m, int ctx);
static float *step(Model *m, const int *ids, int S, int pos_base);
static void kv_alloc(Model *m, int max_t);
static void state_reset(Model *m);
static int build_turn(char *buf, int cap, const char *user);
static void stops_seed(Model *m, Tok *T);
static void banner(Model *m);

/* hook opzionali di strumentazione: no-op se il motore non li definisce */
#ifndef ENGINE_LOGITS_HOOK
#define ENGINE_LOGITS_HOOK(m, lo) ((void)0)
#endif
#ifndef ENGINE_OBSERVE
#define ENGINE_OBSERVE(m, tok) ((void)0)
#endif
/* chiamato una volta dopo model_init+banner (sia REF che generazione):
 * il motore puo' caricare stato extra (es. adattatori LoRA) */
#ifndef ENGINE_POST_INIT
#define ENGINE_POST_INIT(m) ((void)0)
#endif
/* il motore dichiara ENGINE_MICRO 1 se il suo step() sa girare senza embed
 * residente (gather per riga dal disco). Senza dichiarazione MICRO=1 fallisce
 * rumorosamente invece di crashare su embed NULL. */
#ifndef ENGINE_MICRO
#define ENGINE_MICRO 0
#endif

/* ---------- sorgente GGUF (GGUF=<file> al posto di SNAP=<dir>) ----------
 * gguf_index riempie lo stesso indice shards con nomi HF: da qui in poi il
 * runtime non distingue le due sorgenti, salvo config (sintetico dai
 * metadati) e tokenizer (tokenizer.ggml.*). */
static const char *g_gguf = NULL;
static GgufMeta g_gguf_meta;

/* ---------- config: range check ---------- */
#define CKR(name, v, lo, hi) do { long _v=(long)(v); if(_v<(lo)||_v>(hi)){ \
    fprintf(stderr,"config.json: %s=%ld fuori range [%ld,%ld]\n",name,_v,(long)(lo),(long)(hi)); exit(1);} } while(0)

/* legge e parsa config.json; i rilasci multimodali annidano il config testo
 * sotto text_config. Ritorna l'oggetto config; *root_out e' la RADICE parsata
 * (puo' differire per il reparent text_config) e *buf_out il testo: il
 * chiamante li libera con json_free/free a parsing dei campi concluso. */
static jval *cfg_slurp(const char *snap, jval **root_out, char **buf_out) {
    char *buf;
    if (g_gguf) {
        buf = gguf_synth_config(&g_gguf_meta);      /* metadati -> JSON con chiavi HF */
    } else {
        char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
        buf = slurp_file(path, NULL);
    }
    jval *root = json_parse(buf);
    jval *r = root;
    jval *tc = json_get(root,"text_config"); if (tc && tc->t==J_OBJ) r = tc;
    *root_out = root; *buf_out = buf;
    return r;
}

/* campi del config parsati in modo identico da tutti i motori */
static void cfg_common(jval *r, Cfg *c) {
    c->hidden    = (int)json_get(r,"hidden_size")->num;
    c->n_layers  = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads   = (int)json_get(r,"num_attention_heads")->num;
    c->n_kv_heads= (int)json_get(r,"num_key_value_heads")->num;
    c->inter     = (int)json_get(r,"intermediate_size")->num;
    c->vocab     = (int)json_get(r,"vocab_size")->num;
    jval *hd = json_get(r,"head_dim");
    c->head_dim  = hd ? (int)hd->num : c->hidden / c->n_heads;
    jval *mp = json_get(r,"max_position_embeddings"); c->max_pos = mp ? (int)mp->num : 32768;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps = ep ? (float)ep->num : 1e-6f;
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
}

/* ---------- caricamento pesi ---------- */
static float *load_t(Model *m, const char *name, int64_t expect) {
    st_tensor *t = st_expect(&m->S, name, expect);
    float *p = falloc(t->numel);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

/* gruppo delle scale int4 (QGROUP): 32 = blocco Q4_0 di GGUF; 0 = per riga */
static int g_qgroup = 32;

/* carica [O,I] e quantizza secondo bits: 0=f32, 8=int8+scala per riga,
 * 4=int4 impacchettato con scale per gruppo (g_qgroup; 0 -> per riga) */
static void load_mat_bits(Model *m, Mat *w, const char *name, int O, int I, int bits) {
    mat_reset_storage(w);
    w->O = O; w->I = I;
    /* GGUF Q4_0 + QBITS=4 con gruppo 32: repack LOSSLESS (pura permutazione
     * di nibble, gguf.h) invece di dequant+requant — stessi bit del file */
    if (bits == 4 && g_qgroup == 32 && I % 32 == 0 && st_dtype(&m->S, name) == ST_Q4_0) {
        st_tensor *t = st_expect(&m->S, name, (int64_t)O*I);
        void *raw = balloc(t->nbytes, name);
        st_read_raw(&m->S, name, raw, 0);
        w->q4 = balloc((int64_t)O*(I/2), name); w->qs = falloc((int64_t)O*(I/32));
        gguf_repack_q4_0(raw, w->q4, w->qs, O, I);
        free(raw);
        w->gs = 32;
        return;
    }
    w->f = load_t(m, name, (int64_t)O*I);
    if (bits == 8) {
        w->q = balloc((int64_t)O*I, name); w->qs = falloc(O);
        quantize_rows(w->f, w->q, w->qs, O, I, 8);
        free(w->f); w->f = NULL;
    } else if (bits == 4) {
        int gs = g_qgroup;
        int64_t rb = ((int64_t)I+1)/2, ng = gs > 0 ? ((int64_t)I+gs-1)/gs : 1;
        w->q4 = balloc((int64_t)O*rb, name); w->qs = falloc((int64_t)O*ng);
        if (gs > 0) pack_int4_grouped(w->f, w->q4, w->qs, O, I, gs);
        else pack_int4(w->f, w->q4, w->qs, O, I);
        w->gs = gs;
        free(w->f); w->f = NULL;
    }
}

static void load_mat(Model *m, Mat *w, const char *name, int O, int I) {
    load_mat_bits(m, w, name, O, I, m->qbits);
}

/* embed int8 per riga (QBITS=8): letto e quantizzato A BLOCCHI di righe, il
 * picco transiente e' un blocco f32 e non l'intera tabella (622 MB gia' a
 * 0.6B). quantize_rows lavora per riga, quindi il risultato e' bit-identico
 * alla quantizzazione one-shot della tabella intera. Il blocco e' una
 * variabile solo perche' i test lo stringono per esercitare il loop. */
static int g_embed_chunk_rows = 8192;
static void load_embed_q8(Model *m) {
    Cfg *c = &m->c; int D = c->hidden; int64_t V = c->vocab;
    st_expect(&m->S, "model.embed_tokens.weight", V*D);
    m->embed_q  = balloc(V*D, "embed int8");
    m->embed_qs = falloc(V);
    int rows = g_embed_chunk_rows; if (rows < 1) rows = 1;
    float *scratch = falloc((int64_t)rows*D);
    for (int64_t v = 0; v < V; v += rows) {
        int64_t r = V - v < rows ? V - v : rows;
        st_read_slice_f32(&m->S, "model.embed_tokens.weight", v*D, r*D, scratch, 0);
        quantize_rows(scratch, m->embed_q + v*D, m->embed_qs + v, (int)r, D, 8);
    }
    free(scratch);
}

static int64_t layer_f32_bytes(Model *m, int li) {
    MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, li, r);
    int64_t b = 0;
    for (int j = 0; j < n; j++) b += (int64_t)r[j].O*r[j].I*4;
    return b;
}

/* rilettura di un layer streamato. QBITS=0/4: f32 in stream_buf come sempre
 * (per int4 l'impacchettamento a OGNI step costerebbe piu' del risparmio).
 * QBITS=8: lettura a blocchi di righe + quantize_rows nello scratch int8 —
 * il transiente f32 e' un blocco, lo scratch e' 4x piu' piccolo e le matrici
 * streamate girano sullo stesso kernel int8 di quelle residenti (la
 * quantizzazione per riga rende il risultato bit-identico al load residente). */
static void layer_stream_in(Model *m, int li) {
    MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, li, r);
    if (m->qbits == 8) {
        static float *chunk = NULL; static int64_t ccap = 0;
        int64_t qoff = 0, soff = 0;
        for (int j = 0; j < n; j++) {
            int O = r[j].O, I = r[j].I;
            int rows = (int)((4 << 20) / ((int64_t)I * 4)); if (rows < 1) rows = 1;
            grow((void **)&chunk, &ccap, (int64_t)rows*I, sizeof(float), "chunk streaming");
            for (int64_t o = 0; o < O; o += rows) {
                int64_t rr = O - o < rows ? O - o : rows;
                st_read_slice_f32(&m->S, r[j].name, o*I, rr*I, chunk, 0);
                quantize_rows(chunk, m->stream_q + qoff + o*I, m->stream_qs + soff + o, (int)rr, I, 8);
            }
            mat_reset_storage(r[j].mat);
            r[j].mat->q = m->stream_q + qoff; r[j].mat->qs = m->stream_qs + soff;
            r[j].mat->O = O; r[j].mat->I = I;
            qoff += (int64_t)O*I; soff += O;
        }
        return;
    }
    int64_t off = 0;
    for (int j = 0; j < n; j++) {
        st_read_f32(&m->S, r[j].name, m->stream_buf + off, 0);  /* drop=0: la page cache aiuta */
        mat_reset_storage(r[j].mat);
        r[j].mat->f = m->stream_buf + off;
        r[j].mat->O = r[j].O; r[j].mat->I = r[j].I;
        off += (int64_t)r[j].O*r[j].I;
    }
}

static void layer_prefetch(Model *m, int li) {
#ifndef _WIN32                       /* su Windows WILLNEED e' sincrono: niente overlap */
    MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, li, r);
    for (int j = 0; j < n; j++) st_prefetch(&m->S, r[j].name);
#else
    (void)m; (void)li;
#endif
}

/* ---------- micro-RSS (MICRO=1): consumo di RAM minimo assoluto ----------
 * NESSUN peso resta residente: l'embedding si legge per riga (gather nello
 * step del motore), ogni matmul rilegge la propria matrice dal disco a blocchi
 * di g_micro_chunk byte in uno scratch costante. Con g_micro_drop=1 (default)
 * ogni blocco viene anche scartato dalla page cache subito dopo l'uso: il
 * footprint e' davvero solo attivazioni + KV + tokenizer, pensato per limiti
 * HARD (cgroup/embedded). Prezzo: l'intero modello transita dal disco a OGNI
 * token — la velocita' e' bandwidth-del-disco, non della RAM. */
static int     g_micro = 0;             /* attivato da MICRO=1 (engine_main) o dai test */
static int     g_micro_drop = 1;        /* MICRO_DROP=0 -> lascia vivere la page cache */
static int64_t g_micro_chunk = 4 << 20; /* byte f32 dello scratch di streaming */

/* y[S,O] = x[S,I] @ W^T leggendo W dal disco a blocchi di righe; installata in
 * g_mat_stream_fn cosi' mat_apply (nn.h) la usa per le Mat con sh!=NULL.
 * Scratch statico che cresce e basta: contratto di chiamata SERIALE, come
 * matmul_q_s. Bit-identica al percorso f32 residente (stesse righe, stesso
 * dot_f32). */
static void mat_stream(float *y, const float *x, const Mat *w, int S) {
    shards *Sh = (shards *)w->sh;
    int I = w->I, O = w->O;
    int rows = (int)(g_micro_chunk / ((int64_t)I * 4));
    if (rows < 1) rows = 1;
    if (rows > O) rows = O;
    static float *buf = NULL; static int64_t cap = 0;
    grow((void **)&buf, &cap, (int64_t)rows * I, sizeof(float), "scratch micro");
    for (int o0 = 0; o0 < O; o0 += rows) {
        int r = O - o0 < rows ? O - o0 : rows;
        st_read_slice_f32(Sh, w->sname, (int64_t)o0 * I, (int64_t)r * I, buf, g_micro_drop);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < r; o++)
            for (int s = 0; s < S; s++)
                y[(int64_t)s*O + o0 + o] = dot_f32(x + (int64_t)s*I, buf + (int64_t)o*I, I);
    }
}

/* prepara una Mat streamata: dims validate contro il file, nessun dato letto */
static void mat_stream_init(Model *m, Mat *w, const char *name, int O, int I) {
    st_expect(&m->S, name, (int64_t)O*I);
    mat_reset_storage(w);
    w->O = O; w->I = I;
    w->sh = &m->S; w->sname = strdup(name);
}

/* budget_bytes==0 -> tutto residente (comportamento classico) */
static void model_init_ex(Model *m, const char *snap, int qbits, int64_t budget_bytes, int ctx_hint) {
    memset(m, 0, sizeof(*m));
    m->qbits = qbits;
    /* GGUF: l'indice va costruito PRIMA del config (i metadati SONO il config) */
    if (g_gguf) gguf_index(&m->S, &g_gguf_meta, g_gguf);
    load_cfg(&m->c, snap);
    if (!g_gguf) st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    int D = c->hidden;
    m->final_norm = load_t(m, "model.norm.weight", D);
    m->lm_tied = c->tie_emb || !st_has(&m->S, "lm_head.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    /* 1) parte piccola SEMPRE residente (hook: norme, vettori, stati, PLE...) */
    load_small(m);
    int64_t max_lb = 0;
    for (int i = 0; i < c->n_layers; i++) { int64_t b = layer_f32_bytes(m, i); if (b > max_lb) max_lb = b; }
    /* micro-RSS: nessun peso residente, embed compreso (gather per riga nello
     * step del motore); ogni Mat diventa un descrittore streamato. */
    if (g_micro) {
#if ENGINE_MICRO
        g_mat_stream_fn = mat_stream;
        mat_stream_init(m, &m->lm_head,
                        m->lm_tied ? "model.embed_tokens.weight" : "lm_head.weight", c->vocab, D);
        for (int i = 0; i < c->n_layers; i++) {
            MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
            for (int j = 0; j < n; j++) mat_stream_init(m, r[j].mat, r[j].name, r[j].O, r[j].I);
        }
        m->n_resident = 0;                    /* verita': zero layer residenti */
        m->load_s = now_s() - t0;
        fprintf(stderr, "[" ENGINE_TAG "] micro-RSS: 0 pesi residenti, matmul streamato a blocchi da %lld MB, page cache %s\n",
                (long long)(g_micro_chunk >> 20), g_micro_drop ? "scartata (MICRO_DROP=0 per tenerla)" : "attiva");
        return;
#else
        fprintf(stderr, "[" ENGINE_TAG "] MICRO=1 non supportato da questo motore\n");
        exit(1);
#endif
    }
    /* QBITS!=0 copre anche l'embed, ma SEMPRE a int8 (anche con QBITS=4):
     * l'lm_head e' il GEMV piu' sensibile alla quantizzazione e l'int4 li'
     * risparmierebbe poco rispetto alle matrici dei layer */
    if (m->qbits) load_embed_q8(m);
    else m->embed = load_t(m, "model.embed_tokens.weight", (int64_t)c->vocab*D);
    if (m->lm_tied) {
        mat_reset_storage(&m->lm_head);
        m->lm_head.f = m->embed; m->lm_head.q = m->embed_q; m->lm_head.qs = m->embed_qs;
        m->lm_head.O = c->vocab; m->lm_head.I = D;
    } else {
        /* testa non condivisa: int8 anche con QBITS=4 (vedi sopra) */
        load_mat_bits(m, &m->lm_head, "lm_head.weight", c->vocab, D, m->qbits == 4 ? 8 : m->qbits);
    }
    /* 2) budget -> quanti layer di matrici stanno residenti */
    m->n_resident = c->n_layers;
    if (budget_bytes > 0) {
        /* embed (e l'eventuale lm_head separato): f32 oppure int8+scala
         * (con QBITS=4 embed e testa restano comunque int8) */
        int64_t vd = m->qbits ? (int64_t)c->vocab*D + (int64_t)c->vocab*4
                              : (int64_t)c->vocab*D*4;
        int64_t fixed = vd + (int64_t)D*4;                          /* + final_norm */
        if (!m->lm_tied) fixed += vd;
        fixed += (int64_t)c->n_layers * 8 * D * 4;                  /* norme/vettori: stima larga */
        fixed += fixed_bytes(m, ctx_hint > 0 ? ctx_hint : 4096);    /* hook: KV, PLE... */
        /* scratch di streaming: int8 con QBITS=8 (layer_stream_in quantizza), f32 altrimenti */
        int64_t scratch = (m->qbits == 8) ? max_lb/4 + max_lb/64 : max_lb;
        int64_t used = fixed + scratch;
        int R = 0;
        for (; R < c->n_layers; R++) {
            int64_t lb = layer_f32_bytes(m, R);
            if (m->qbits == 8) lb = lb/4 + lb/64;                   /* int8 + scale */
            else if (m->qbits == 4) lb = lb/8 + lb/32;              /* int4 + scale di gruppo (gs=32) */
            if (used + lb > budget_bytes) break;
            used += lb;
        }
        m->n_resident = R;
        fprintf(stderr, "[" ENGINE_TAG "] budget %.2f GB -> %d/%d layer residenti (fisso %.2f GB, scratch %.2f GB)\n",
                budget_bytes/1073741824.0, R, c->n_layers, fixed/1073741824.0, scratch/1073741824.0);
#if ENGINE_MICRO
        /* il classico non scende sotto embed + scratch: budget irrealizzabile */
        if (budget_bytes < fixed + scratch)
            fprintf(stderr, "[" ENGINE_TAG "] budget sotto il pavimento residente (%.2f GB): per la RSS minima usa MICRO=1\n",
                    (fixed + scratch)/1073741824.0);
#endif
    }
    /* 3) matrici: residenti (QBITS onorato) o streamate (dims impostate, f=NULL) */
    for (int i = 0; i < c->n_layers; i++) {
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        for (int j = 0; j < n; j++) {
            if (i < m->n_resident) load_mat(m, r[j].mat, r[j].name, r[j].O, r[j].I);
            else {
                st_expect(&m->S, r[j].name, (int64_t)r[j].O*r[j].I);
                mat_reset_storage(r[j].mat);
                r[j].mat->O = r[j].O; r[j].mat->I = r[j].I;
            }
        }
    }
    if (m->n_resident < c->n_layers) {
        /* scratch dimensionato sul massimo dei layer EFFETTIVAMENTE streamati
         * (i >= n_resident), non sul massimo globale */
        int64_t smax = 0, rmax = 0;
        for (int i = m->n_resident; i < c->n_layers; i++) {
            int64_t b = layer_f32_bytes(m, i); if (b > smax) smax = b;
            MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
            int64_t rows = 0; for (int j = 0; j < n; j++) rows += r[j].O;
            if (rows > rmax) rmax = rows;
        }
        if (m->qbits == 8) {
            m->stream_q = balloc(smax/4, "scratch stream int8");  /* 1 byte per elemento f32 */
            m->stream_qs = falloc(rmax);
        } else {
            m->stream_buf = falloc(smax/4);
        }
    }
    m->load_s = now_s() - t0;
}

static void model_init(Model *m, const char *snap, int qbits) {
    model_init_ex(m, snap, qbits, 0, 0);
}

/* ---------- prefill a blocchi (PREFILL_CHUNK) ----------
 * Le attivazioni del prefill crescono con S (mlp: 2*S*inter f32 — 1.6 GB a
 * S=4096 su un 4B) e lo scratch statico di matmul_q_s resta a S*I per sempre:
 * spezzare il prompt in blocchi <= C limita entrambi a una costante. E' un
 * opt-in (default 0 = spento) perche' con MEM_GB ogni step() rilegge i layer
 * streamati dal disco: N blocchi = N riletture del prompt.
 * Bit-esattezza: ogni operazione per-token dipende solo dalla posizione
 * ASSOLUTA (RoPE, scrittura KV, ricorrenza deltanet, PLE) e l'attention legge
 * la KV scritta dalle posizioni precedenti, identica comunque si spezzi;
 * matmul_q_s quantizza le attivazioni PER RIGA, quindi e' batch-invariante.
 * g_skip_logits: sui blocchi intermedi il motore esce da step() PRIMA di
 * final-norm/lm_head (e dello stash TTA) e ritorna NULL — i logits (e lo
 * stash) esistono solo per l'ultimo token del prompt, come non-chunked. */
static int g_prefill_chunk = 0;
static int g_skip_logits = 0;

/* KV_BITS=8: KV-cache int8 con scala per (testa_kv, posizione). Default 0
 * (f32): la numerica di REF non cambia mai in silenzio. */
static int g_kv_bits = 0;

static float *step_chunked(Model *m, const int *ids, int S, int pos_base) {
    int C = g_prefill_chunk;
    if (C <= 0 || S <= C) return step(m, ids, S, pos_base);
    int done = 0;
    for (; S - done > C; done += C) {
        g_skip_logits = 1;
        float *lo = step(m, ids + done, C, pos_base + done);
        g_skip_logits = 0;
        if (lo) free(lo);              /* i motori ritornano NULL quando saltano */
    }
    return step(m, ids + done, S - done, pos_base + done);
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
    char *buf = slurp_file(refpath, NULL);
    jval *ref = json_parse(buf);
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
    float *logit = step_chunked(m, prompt, np, 0);
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
    json_free(ref); free(buf); free(prompt); free(full); free(out);
    return match == n_new ? 0 : 2;
}

/* ---------- generazione di un turno ----------
 * hist[len..len+k) = token nuovi da prefillare; genera fino a n_new token o stop.
 * Stampa il testo su stdout se echo!=0. Ritorna il numero di token generati
 * (stop incluso se emesso); *stopped=1 se l'ultimo token e' uno stop. */
static int gen_turn(Model *m, Tok *T, int *hist, int len, int k, int n_new, int echo, int *stopped) {
    int dump = getenv("TOKENS") && atoi(getenv("TOKENS"));
    double t0 = now_s();
    float *logit = step_chunked(m, hist + len, k, len);  /* PREFILL (a blocchi se PREFILL_CHUNK) */
    ENGINE_LOGITS_HOOK(m, logit);
    double tpre = now_s() - t0;
    int base = len + k, ng = 0; *stopped = 0;
    t0 = now_s();
    for (int s = 0; s < n_new; s++) {
        int t = pick_tok(logit, m->c.vocab);
        free(logit); logit = NULL;
        hist[base + ng++] = t;
        ENGINE_OBSERVE(m, t);
        if (dump) fprintf(stderr, "%d ", t);
        if (is_stop(t)) { *stopped = 1; break; }
        if (echo) {
            char buf[64]; int bn = tok_decode(T, &t, 1, buf, 63);
            fwrite(buf, 1, bn, stdout); fflush(stdout);
        }
        if (s == n_new - 1) break;
        logit = step(m, &hist[base + ng - 1], 1, base + ng - 1);
        ENGINE_LOGITS_HOOK(m, logit);
    }
    if (logit) free(logit);
    if (dump) fprintf(stderr, "\n");
    double tgen = now_s() - t0;
    fprintf(stderr, "\n[" ENGINE_TAG "] prefill %d tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.2f tok/s) | RSS %.2f GB\n",
            k, tpre, k/(tpre>1e-9?tpre:1e-9), ng, tgen, ng/(tgen>1e-9?tgen:1e-9), rss_gb());
    return ng;
}

/* ---------- budget di memoria: MEM_GB (GiB) batte MEM_FRAC (frazione della
 * RAM fisica TOTALE, deterministico). 0 = tutto residente. ---------- */
static int64_t budget_from_env(const char *gb, const char *frac, int64_t total_ram) {
    if (gb && *gb) { double g = atof(gb); if (g > 0) return (int64_t)(g * 1073741824.0); }
    if (frac && *frac) { double f = atof(frac); if (f > 0 && f <= 1 && total_ram > 0) return (int64_t)(f * total_ram); }
    return 0;
}

/* ---------- tuning permanente dei thread OpenMP (portato da glm.c) ----------
 * Le regioni parallele dei motori densi sono piccole e back-to-back (centinaia
 * di fork/join per token); con la wait policy passiva di default libgomp
 * parcheggia il team tra una regione e l'altra e la latenza di risveglio
 * domina. Tenere i thread caldi (spin attivo) collassa quell'overhead: su glm
 * il tempo matmul e' passato da 66.9s a 20.9s sulla build Zen5, senza alcuna
 * variazione dell'output numerico.
 *
 * libgomp legge le variabili OMP_/GOMP_ in un COSTRUTTORE che gira prima di
 * main(): un setenv() qui seguito dall'esecuzione normale arriverebbe troppo
 * tardi. Quindi al primo ingresso si seminano i default vincenti — rispettando
 * qualunque valore l'utente abbia gia' impostato (overwrite=0) — e ci si
 * re-esegue una volta sola cosi' un costruttore libgomp fresco li raccoglie.
 * La sentinella COLI_OMP_TUNED garantisce al massimo un re-exec; COLI_NO_OMP_TUNE=1
 * e' il kill-switch documentato che disattiva tutto il percorso. Su piattaforme
 * senza /proc/self/exe (o se execv fallisce) si prosegue senza tuning. */
static void omp_hot_tune(char **argv) {
    if (!getenv("COLI_OMP_TUNED") && !getenv("COLI_NO_OMP_TUNE")) {
        setenv("OMP_WAIT_POLICY", "active", 0);  /* team caldo tra le regioni piccole e fitte */
        setenv("GOMP_SPINCOUNT", "200000", 0);   /* spin breve, poi yield: le attese lunghe non bruciano un core */
        setenv("OMP_PROC_BIND", "close", 0);     /* team impacchettato su core adiacenti (localita' di cache) */
        setenv("OMP_DYNAMIC", "FALSE", 0);       /* team a taglia fissa: niente churn per-regione */
        setenv("COLI_OMP_TUNED", "1", 1);
#if defined(__linux__)
        fprintf(stderr, "[OMP] hot-thread tuning: re-exec once (COLI_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/self/exe", argv);           /* ritorna solo in caso di errore -> si prosegue senza tuning */
        perror("[OMP] execv self-reexec failed, running untuned");
#elif defined(__FreeBSD__)
        fprintf(stderr, "[OMP] hot-thread tuning: re-exec once (COLI_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/curproc/file", argv);       /* ritorna solo in caso di errore -> si prosegue senza tuning */
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
}

/* ---------- main condiviso ---------- */
static int engine_main(int argc, char **argv) {
    (void)argc;
    omp_hot_tune(argv);
    /* THREADS: tetto sul team OpenMP (batte OMP_NUM_THREADS), applicato PRIMA
     * di qualunque allocazione dipendente dal numero di thread. */
    const char *th_ = getenv("THREADS");
    if (th_ && atoi(th_) > 0) omp_set_num_threads(atoi(th_));
    const char *snap = getenv("SNAP");
    g_gguf = getenv("GGUF");
    if (g_gguf && !*g_gguf) g_gguf = NULL;
    if (!snap && !g_gguf) { fprintf(stderr, "set SNAP=<snapshot directory> oppure GGUF=<file.gguf>\n"); return 1; }
    int qbits = getenv("QBITS") ? atoi(getenv("QBITS")) : 0;
    if (qbits != 0 && qbits != 4 && qbits != 8) { fprintf(stderr, "QBITS deve essere 0 (f32), 4 (int4) o 8 (int8)\n"); return 1; }
    if (getenv("QGROUP")) {
        g_qgroup = atoi(getenv("QGROUP"));
        if (g_qgroup < 0 || (g_qgroup > 0 && g_qgroup % 16)) {
            fprintf(stderr, "QGROUP deve essere 0 (scala per riga) o un multiplo di 16\n"); return 1; }
    }
    int ngen  = getenv("NGEN") ? atoi(getenv("NGEN")) : 256;
    if (getenv("PREFILL_CHUNK")) g_prefill_chunk = atoi(getenv("PREFILL_CHUNK"));
    if (getenv("KV_BITS")) {
        g_kv_bits = atoi(getenv("KV_BITS"));
        if (g_kv_bits != 0 && g_kv_bits != 8) { fprintf(stderr, "KV_BITS deve essere 0 (f32) o 8 (int8)\n"); return 1; }
    }
    /* MICRO=1: micro-RSS. La KV-cache resta l'unica voce grande -> il default
     * di contesto scende a 256 (CTX esplicito vince sempre). */
    const char *mi_ = getenv("MICRO");
    g_micro = mi_ && atoi(mi_) > 0;
    const char *md_ = getenv("MICRO_DROP");
    if (md_ && *md_) g_micro_drop = atoi(md_) != 0;
    int maxctx= getenv("CTX")  ? atoi(getenv("CTX"))  : (g_micro ? 256 : 4096);
    if (getenv("TEMP"))    g_temp = (float)atof(getenv("TEMP"));
    if (getenv("NUCLEUS")) g_nuc  = (float)atof(getenv("NUCLEUS"));
    if (getenv("SEED"))    g_rng  = (uint64_t)strtoull(getenv("SEED"),NULL,10) | 1u;
    int templ = getenv("CHAT_TEMPLATE") ? atoi(getenv("CHAT_TEMPLATE")) : 1;
    int64_t budget = budget_from_env(getenv("MEM_GB"), getenv("MEM_FRAC"), compat_total_ram_bytes());

    Model m;
    model_init_ex(&m, snap, qbits, budget, maxctx);
    banner(&m);
    /* banner precede il ramo REF: l'hook gira UNA volta per entrambi i percorsi */
    ENGINE_POST_INIT(&m);
    if (m.c.max_pos > 0 && maxctx > m.c.max_pos) maxctx = m.c.max_pos;

    const char *refpath = getenv("REF");
    if (refpath) return run_ref(&m, refpath);

    Tok T;
    if (g_gguf) tok_load_gguf(&T, &g_gguf_meta);   /* single-file: vocab/merges dai metadati */
    else {
        char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
        tok_load(&T, tokpath);
    }
    stops_seed(&m, &T);
    for (int i = 0; i < m.c.n_eos; i++) stop_add(m.c.eos[i]);
    fprintf(stderr, "[" ENGINE_TAG "] stop tokens:"); for (int i=0;i<g_nstop;i++) fprintf(stderr," %d",g_stop[i]); fprintf(stderr,"\n");

    kv_alloc(&m, maxctx);
    int *hist = malloc(maxctx * sizeof(int));
    char *buf = malloc(1<<16);

    const char *prompt = getenv("PROMPT");
    if (prompt) {                                   /* one-shot */
        int bl = templ ? build_turn(buf, 1<<16, prompt)
                       : snprintf(buf, 1<<16, "%s", prompt);
        int k = tok_encode(&T, buf, bl, hist, maxctx - 2);
        int cur = ngen; if (k + cur + 1 > maxctx) cur = maxctx - k - 1;
        int stopped;
        gen_turn(&m, &T, hist, 0, k, cur, 1, &stopped);
        printf("\n");
        return 0;
    }

    /* chat interattiva: KV persistente, storia append-only */
    fprintf(stderr, "[" ENGINE_TAG "] chat interattiva: scrivi e premi invio (Ctrl-D per uscire)\n");
    int len = 0;                                    /* token gia' in KV */
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
        if (len + k + 8 >= maxctx) {                /* contesto pieno: reset conversazione */
            fprintf(stderr, "[" ENGINE_TAG "] contesto pieno, reset della conversazione\n");
            len = 0; m.kv_len = 0; state_reset(&m);  /* lo stato ricorrente non e' troncabile */
            k = tok_encode(&T, buf, bl, hist, maxctx - 2);
        }
        int cur = ngen; if (len + k + cur + 1 > maxctx) cur = maxctx - len - k - 1;
        int stopped;
        int ng = gen_turn(&m, &T, hist, len, k, cur, 1, &stopped);
        len += k + ng;
        /* chiude il blocco assistant nel transcript: i token del suffisso entrano
         * in KV col prefill del turno successivo */
        const char *suffix = stopped ? "\n" : ENGINE_EOT;
        len += tok_encode(&T, suffix, (int)strlen(suffix), hist + len, maxctx - len);
    }
    return 0;
}

#endif /* RUNTIME_H */
