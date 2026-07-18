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

/* ---------- config: range check ---------- */
#define CKR(name, v, lo, hi) do { long _v=(long)(v); if(_v<(lo)||_v>(hi)){ \
    fprintf(stderr,"config.json: %s=%ld fuori range [%ld,%ld]\n",name,_v,(long)(lo),(long)(hi)); exit(1);} } while(0)

/* legge e parsa config.json; i rilasci multimodali annidano il config testo
 * sotto text_config. buf/arena restano vivi (one-shot allo startup). */
static jval *cfg_slurp(const char *snap, char **arena_out) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    jval *r = json_parse(buf, arena_out);
    jval *tc = json_get(r,"text_config"); if (tc && tc->t==J_OBJ) r = tc;
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

static int64_t layer_f32_bytes(Model *m, int li) {
    MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, li, r);
    int64_t b = 0;
    for (int j = 0; j < n; j++) b += (int64_t)r[j].O*r[j].I*4;
    return b;
}

/* rilettura di un layer streamato: tutte le matrici in stream_buf (f32) */
static void layer_stream_in(Model *m, int li) {
    MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, li, r);
    int64_t off = 0;
    for (int j = 0; j < n; j++) {
        st_read_f32(&m->S, r[j].name, m->stream_buf + off, 0);  /* drop=0: la page cache aiuta */
        r[j].mat->f = m->stream_buf + off;
        r[j].mat->q = NULL; r[j].mat->qs = NULL;
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

/* budget_bytes==0 -> tutto residente (comportamento classico) */
static void model_init_ex(Model *m, const char *snap, int qbits, int64_t budget_bytes, int ctx_hint) {
    memset(m, 0, sizeof(*m));
    m->qbits = qbits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
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
    m->L = calloc(c->n_layers, sizeof(Layer));
    /* 1) parte piccola SEMPRE residente (hook: norme, vettori, stati, PLE...) */
    load_small(m);
    /* 2) budget -> quanti layer di matrici stanno residenti */
    m->n_resident = c->n_layers;
    int64_t max_lb = 0;
    for (int i = 0; i < c->n_layers; i++) { int64_t b = layer_f32_bytes(m, i); if (b > max_lb) max_lb = b; }
    if (budget_bytes > 0) {
        int64_t fixed = ((int64_t)c->vocab*D + D)*4;                /* embed + final_norm */
        if (!m->lm_tied) fixed += (int64_t)c->vocab*D*4;
        fixed += (int64_t)c->n_layers * 8 * D * 4;                  /* norme/vettori: stima larga */
        fixed += fixed_bytes(m, ctx_hint > 0 ? ctx_hint : 4096);    /* hook: KV, PLE... */
        int64_t used = fixed + max_lb;                              /* scratch di streaming */
        int R = 0;
        for (; R < c->n_layers; R++) {
            int64_t lb = layer_f32_bytes(m, R);
            if (m->qbits == 8) lb = lb/4 + lb/64;                   /* int8 + scale */
            if (used + lb > budget_bytes) break;
            used += lb;
        }
        m->n_resident = R;
        fprintf(stderr, "[" ENGINE_TAG "] budget %.2f GB -> %d/%d layer residenti (fisso %.2f GB, scratch %.2f GB)\n",
                budget_bytes/1073741824.0, R, c->n_layers, fixed/1073741824.0, max_lb/1073741824.0);
    }
    /* 3) matrici: residenti (QBITS onorato) o streamate (dims impostate, f=NULL) */
    for (int i = 0; i < c->n_layers; i++) {
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        for (int j = 0; j < n; j++) {
            if (i < m->n_resident) load_mat(m, r[j].mat, r[j].name, r[j].O, r[j].I);
            else {
                int64_t have = st_numel(&m->S, r[j].name);
                if (have != (int64_t)r[j].O*r[j].I) {
                    fprintf(stderr, "tensor %s: numel %lld != atteso %lld\n",
                            r[j].name, (long long)have, (long long)((int64_t)r[j].O*r[j].I));
                    exit(1);
                }
                r[j].mat->f = NULL; r[j].mat->q = NULL; r[j].mat->qs = NULL;
                r[j].mat->O = r[j].O; r[j].mat->I = r[j].I;
            }
        }
    }
    if (m->n_resident < c->n_layers)
        m->stream_buf = falloc(max_lb/4);
    m->load_s = now_s() - t0;
}

static void model_init(Model *m, const char *snap, int qbits) {
    model_init_ex(m, snap, qbits, 0, 0);
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

/* ---------- generazione di un turno ----------
 * hist[len..len+k) = token nuovi da prefillare; genera fino a n_new token o stop.
 * Stampa il testo su stdout se echo!=0. Ritorna il numero di token generati
 * (stop incluso se emesso); *stopped=1 se l'ultimo token e' uno stop. */
static int gen_turn(Model *m, Tok *T, int *hist, int len, int k, int n_new, int echo, int *stopped) {
    int dump = getenv("TOKENS") && atoi(getenv("TOKENS"));
    double t0 = now_s();
    float *logit = step(m, hist + len, k, len);      /* PREFILL */
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
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    int qbits = getenv("QBITS") ? atoi(getenv("QBITS")) : 0;
    if (qbits != 0 && qbits != 8) { fprintf(stderr, "QBITS deve essere 0 (f32) o 8 (int8)\n"); return 1; }
    int ngen  = getenv("NGEN") ? atoi(getenv("NGEN")) : 256;
    int maxctx= getenv("CTX")  ? atoi(getenv("CTX"))  : 4096;
    if (getenv("TEMP"))    g_temp = (float)atof(getenv("TEMP"));
    if (getenv("NUCLEUS")) g_nuc  = (float)atof(getenv("NUCLEUS"));
    if (getenv("SEED"))    g_rng  = (uint64_t)strtoull(getenv("SEED"),NULL,10) | 1u;
    int templ = getenv("CHAT_TEMPLATE") ? atoi(getenv("CHAT_TEMPLATE")) : 1;
    int64_t budget = budget_from_env(getenv("MEM_GB"), getenv("MEM_FRAC"), compat_total_ram_bytes());

    Model m;
    model_init_ex(&m, snap, qbits, budget, maxctx);
    banner(&m);
    if (m.c.max_pos > 0 && maxctx > m.c.max_pos) maxctx = m.c.max_pos;

    const char *refpath = getenv("REF");
    if (refpath) return run_ref(&m, refpath);

    char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
    Tok T; tok_load(&T, tokpath);
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
