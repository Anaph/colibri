/* Tokenizer GLM-5.2 in C puro (byte-level BPE stile cl100k / tiktoken).
 * Replica fedele di tokenizer.json:
 *   - model.type = BPE, ignore_merges=true, byte_fallback=false
 *   - pre_tokenizer: regex Split (pattern cl100k) + ByteLevel(add_prefix_space=false)
 *   - merges con rank = ordine nella lista; \p{L}/\p{N}/\s da tok_unicode.h
 *   - added_tokens (speciali e non) trattati come atomici in encode/decode
 * API:
 *   tok_load(&T, "tokenizer.json");
 *   int n = tok_encode(&T, text, len, out_ids, max);
 *   int m = tok_decode(&T, ids, n, out_buf, max);
 */
#ifndef TOK_H
#define TOK_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include "json.h"
#include "tok_unicode.h"

/* ---------- hash map (chiavi binarie con lunghezza) ---------- */
typedef struct { const char *k; int klen; int v; int used; } ment;
typedef struct { ment *e; int cap; } hmap;
static uint64_t tk_fnv(const char *s, int n){ uint64_t h=1469598103934665603ULL;
    for(int i=0;i<n;i++){ h^=(unsigned char)s[i]; h*=1099511628211ULL; } return h; }
static void hm_init(hmap *m, int cap){ m->cap=cap; m->e=(ment*)calloc(cap,sizeof(ment)); }
static void hm_put(hmap *m, const char *k, int klen, int v){
    uint64_t h=tk_fnv(k,klen)&(m->cap-1);
    while(m->e[h].used){ if(m->e[h].klen==klen && !memcmp(m->e[h].k,k,klen)){ m->e[h].v=v; return; } h=(h+1)&(m->cap-1); }
    m->e[h].k=k; m->e[h].klen=klen; m->e[h].v=v; m->e[h].used=1;
}
static int hm_get(hmap *m, const char *k, int klen){
    uint64_t h=tk_fnv(k,klen)&(m->cap-1);
    while(m->e[h].used){ if(m->e[h].klen==klen && !memcmp(m->e[h].k,k,klen)) return m->e[h].v; h=(h+1)&(m->cap-1); }
    return -1;
}

typedef struct { char *str; int len; int id; } Special;
typedef struct {
    hmap vocab;          /* stringa byte-level -> id */
    hmap merges;         /* "left\0right" -> rank */
    char **id2str; int *id_added; int n_ids;   /* id -> stringa; id_added=1 se added-token (output letterale) */
    Special *sp; int nsp;                       /* added tokens, ordinati per lunghezza decrescente */
    uint32_t byte2cp[256]; int byte2cp_len[256]; char byte2str[256][3];
    int16_t cp2byte[1024];
    /* modalita' SentencePiece (Gemma & co.): BPE su stringhe LETTERALI con
     * metaspazio U+2581 al posto degli spazi e byte-fallback <0xXX>.
     * Rilevata da model.byte_fallback==true nel tokenizer.json. */
    int mode;                 /* 0 = byte-level (GPT-2/cl100k), 1 = sentencepiece */
    int add_dummy_prefix;     /* normalizer Prepend "▁": prefissa il testo */
    int byte_tok[256];        /* id di "<0xXX>" oppure -1 */
    int16_t *id2byte;         /* [n_ids] inverso di byte_tok (-1 = non byte-token) */
    /* pool di stringhe: UNA malloc per tutte le chiavi vocab/merges e i
     * contenuti added (vs ~450k malloc del parse JSON, che viene liberato) */
    char *pool; int64_t pool_len, pool_off;
} Tok;

/* bump allocator sul pool (dimensionato esattamente da un passo di sizing:
 * lo sforamento e' un bug di conteggio, non una condizione di runtime) */
static char *tk_pool_take(Tok *T, int64_t n){
    if (T->pool_off + n > T->pool_len) { fprintf(stderr,"[tok] pool overflow (bug di sizing)\n"); exit(1); }
    char *p = T->pool + T->pool_off; T->pool_off += n; return p;
}
static char *tk_pool_dup(Tok *T, const char *s, int n){   /* copia NUL-terminata */
    char *p = tk_pool_take(T, n + 1);
    memcpy(p, s, n); p[n] = 0; return p;
}

/* ---------- UTF-8 ---------- */
static int u8_next(const unsigned char *s, int len, int i, uint32_t *cp){
    unsigned char c=s[i];
    if(c<0x80){ *cp=c; return 1; }
    if((c>>5)==0x6 && i+1<len){ *cp=((c&0x1F)<<6)|(s[i+1]&0x3F); return 2; }
    if((c>>4)==0xE && i+2<len){ *cp=((c&0x0F)<<12)|((s[i+1]&0x3F)<<6)|(s[i+2]&0x3F); return 3; }
    if((c>>3)==0x1E && i+3<len){ *cp=((c&0x07)<<18)|((s[i+1]&0x3F)<<12)|((s[i+2]&0x3F)<<6)|(s[i+3]&0x3F); return 4; }
    *cp=c; return 1;   /* byte invalido: trattato come singolo */
}
static int u8_put(char *o, uint32_t cp){
    if(cp<0x80){ o[0]=cp; return 1; }
    if(cp<0x800){ o[0]=0xC0|(cp>>6); o[1]=0x80|(cp&0x3F); return 2; }
    if(cp<0x10000){ o[0]=0xE0|(cp>>12); o[1]=0x80|((cp>>6)&0x3F); o[2]=0x80|(cp&0x3F); return 3; }
    o[0]=0xF0|(cp>>18); o[1]=0x80|((cp>>12)&0x3F); o[2]=0x80|((cp>>6)&0x3F); o[3]=0x80|(cp&0x3F); return 4;
}

/* ---------- mappa byte<->unicode di GPT-2/ByteLevel ---------- */
static void tk_build_bytemap(Tok *T){
    for(int i=0;i<1024;i++) T->cp2byte[i]=-1;
    int isdir[256]; memset(isdir,0,sizeof(isdir));
    for(int b=33;b<=126;b++) isdir[b]=1;
    for(int b=161;b<=172;b++) isdir[b]=1;
    for(int b=174;b<=255;b++) isdir[b]=1;
    int n=0;
    for(int b=0;b<256;b++){
        uint32_t cp = isdir[b] ? (uint32_t)b : (uint32_t)(256+n);
        if(!isdir[b]) n++;
        T->byte2cp[b]=cp;
        T->byte2cp_len[b]=u8_put(T->byte2str[b], cp);
        if(cp<1024) T->cp2byte[cp]=b;
    }
}

/* ---------- caricamento tokenizer.json ---------- */
static int cmp_sp_len(const void *a, const void *b){ return ((const Special*)b)->len - ((const Special*)a)->len; }

/* tabella byte-fallback <0xXX> e inversa id2byte: identica per tokenizer.json
 * e GGUF, il vocab e' gia' nelle hmap quando viene chiamata */
static void tk_byte_tables(Tok *T){
    T->id2byte=malloc(T->n_ids*sizeof(int16_t));
    for(int i=0;i<T->n_ids;i++) T->id2byte[i]=-1;
    for(int b=0;b<256;b++){
        char nm[8]; int nl=snprintf(nm,sizeof(nm),"<0x%02X>",b);
        T->byte_tok[b]=hm_get(&T->vocab,nm,nl);
        if(T->byte_tok[b]>=0) T->id2byte[T->byte_tok[b]]=(int16_t)b;
    }
}

static void tok_load(Tok *T, const char *path){
    memset(T,0,sizeof(*T));
    tk_build_bytemap(T);
    char *buf=slurp_file(path,NULL);
    jval *root=json_parse(buf);
    jval *model=json_get(root,"model");
    jval *vocab=json_get(model,"vocab");
    jval *merges=json_get(model,"merges");
    jval *added=json_get(root,"added_tokens");
    if(!vocab||!merges){ fprintf(stderr,"tokenizer.json: missing model.vocab/merges\n"); exit(1); }
    /* modalita': byte_fallback=true e' la firma SentencePiece (assente/false su
     * GLM/Qwen byte-level) */
    jval *bf=json_get(model,"byte_fallback");
    T->mode = (bf && bf->t==J_BOOL && bf->boolean) ? 1 : 0;
    /* add_dummy_prefix: nodo Prepend nel normalizer (diretto o in Sequence) */
    T->add_dummy_prefix = 0;
    jval *nrm=json_get(root,"normalizer");
    if(nrm && nrm->t==J_OBJ){
        jval *nt=json_get(nrm,"type");
        if(nt && nt->t==J_STR && !strcmp(nt->str,"Prepend")) T->add_dummy_prefix=1;
        jval *seq=json_get(nrm,"normalizers");
        if(seq && seq->t==J_ARR)
            for(int i=0;i<seq->len;i++){
                jval *st_=json_get(seq->kids[i],"type");
                if(st_ && st_->t==J_STR && !strcmp(st_->str,"Prepend")) T->add_dummy_prefix=1;
            }
    }
    if(T->mode) fprintf(stderr,"[tok] modalita' sentencepiece (byte_fallback), dummy_prefix=%d\n",T->add_dummy_prefix);

    /* id massimo per dimensionare id2str */
    int maxid=0;
    for(int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>maxid)maxid=id; }
    if(added) for(int i=0;i<added->len;i++){ int id=(int)json_get(added->kids[i],"id")->num; if(id>maxid)maxid=id; }
    T->n_ids=maxid+1;
    T->id2str=calloc(T->n_ids,sizeof(char*));
    T->id_added=calloc(T->n_ids,sizeof(int));

    /* sizing del pool: stessa aritmetica dei cicli di build qui sotto, cosi'
     * il bump allocator non puo' sforare. Le chiavi merges sono binarie
     * "left\0right" senza NUL finale (le hmap usano klen). */
    int64_t psz=0;
    for(int i=0;i<vocab->len;i++) psz += (int64_t)strlen(vocab->keys[i]) + 1;
    for(int i=0;i<merges->len;i++){
        jval *pr=merges->kids[i];
        if(pr->t==J_STR){
            const char *sp=strchr(pr->str,' ');
            if(!sp){ fprintf(stderr,"tokenizer.json: merge senza spazio: %s\n",pr->str); exit(1); }
            psz += (int64_t)strlen(pr->str);           /* ll+1+rl: lo spazio diventa NUL */
        } else {
            psz += (int64_t)strlen(pr->kids[0]->str) + 1 + strlen(pr->kids[1]->str);
        }
    }
    if(added) for(int i=0;i<added->len;i++) psz += (int64_t)strlen(json_get(added->kids[i],"content")->str) + 1;
    T->pool_len=psz; T->pool_off=0;
    T->pool=malloc(psz > 0 ? psz : 1);
    if(!T->pool){ fprintf(stderr,"[tok] OOM pool %lld\n",(long long)psz); exit(1); }

    /* vocab: stringa -> id  (capacita' potenza di 2, ~2-3x) */
    int vc=1; while(vc < vocab->len*2) vc<<=1;
    hm_init(&T->vocab, vc);
    for(int i=0;i<vocab->len;i++){
        int kl=(int)strlen(vocab->keys[i]); int id=(int)vocab->kids[i]->num;
        char *k=tk_pool_dup(T, vocab->keys[i], kl);
        hm_put(&T->vocab, k, kl, id);
        T->id2str[id]=k;
    }
    /* merges: "left\0right" -> rank=i
     * Due formati in circolazione: coppie ["left","right"] (GLM) oppure
     * stringhe "left right" con separatore spazio singolo (Qwen/GPT-2). */
    int mc=1; while(mc < merges->len*2) mc<<=1;
    hm_init(&T->merges, mc);
    for(int i=0;i<merges->len;i++){
        jval *pr=merges->kids[i];
        const char *l, *r; int ll, rl;
        if(pr->t==J_STR){
            const char *sp=strchr(pr->str,' ');   /* verificato dal sizing */
            l=pr->str; ll=(int)(sp-pr->str); r=sp+1; rl=(int)strlen(sp+1);
        } else {
            l=pr->kids[0]->str; ll=(int)strlen(l);
            r=pr->kids[1]->str; rl=(int)strlen(r);
        }
        char *key=tk_pool_take(T, ll+1+rl);
        memcpy(key,l,ll); key[ll]=0; memcpy(key+ll+1,r,rl);
        hm_put(&T->merges, key, ll+1+rl, i);
    }
    /* added tokens (speciali e non): atomici, output letterale */
    if(added){
        T->nsp=added->len; T->sp=calloc(T->nsp,sizeof(Special));
        for(int i=0;i<added->len;i++){
            jval *a=added->kids[i];
            const char *cs=json_get(a,"content")->str; int id=(int)json_get(a,"id")->num;
            char *content=tk_pool_dup(T, cs, (int)strlen(cs));
            T->sp[i].str=content; T->sp[i].len=(int)strlen(content); T->sp[i].id=id;
            T->id2str[id]=content; T->id_added[id]=1;
        }
        qsort(T->sp,T->nsp,sizeof(Special),cmp_sp_len);   /* match piu' lungo per primo */
    }
    tk_byte_tables(T);      /* <0xXX> (solo modalita' sp; -1 se il vocab non li ha) */
    /* tutte le stringhe vive stanno nel pool: il parse JSON (~450k malloc su
     * un vocab da 150k) e il testo del file si possono liberare */
    json_free(root); free(buf);
}

/* libera tutto lo stato del tokenizer (pool compreso: una free per tutte le stringhe) */
static void tok_free(Tok *T){
    free(T->pool); free(T->vocab.e); free(T->merges.e);
    free(T->id2str); free(T->id_added); free(T->sp); free(T->id2byte);
    memset(T,0,sizeof(*T));
}

#ifdef GGUF_H
/* carica il tokenizer dai metadati GGUF (tokenizer.ggml.*): stesso Tok di
 * tok_load, stringhe nel pool. Modello atteso "gpt2" (byte-level BPE, la
 * famiglia Qwen); i tokenizer sentencepiece in GGUF non sono supportati.
 * In GGUF l'id di un token E' il suo indice nell'array tokens. */
static void tok_load_gguf(Tok *T, GgufMeta *M) {
    memset(T, 0, sizeof(*T));
    tk_build_bytemap(T);
    int64_t ml; const char *mdl = gguf_str(M, "tokenizer.ggml.model", &ml);
    if (!mdl || ml != 4 || memcmp(mdl, "gpt2", 4)) {
        fprintf(stderr, "[tok] GGUF: tokenizer.ggml.model non 'gpt2' (solo byte-level BPE)\n"); exit(1);
    }
    T->mode = 0;
    garr toks, types, mrg;
    if (!gguf_arr(M, "tokenizer.ggml.tokens", &toks)) {
        fprintf(stderr, "[tok] GGUF: manca tokenizer.ggml.tokens\n"); exit(1); }
    int64_t nvoc = toks.left;
    int have_types  = gguf_arr(M, "tokenizer.ggml.token_type", &types);
    int have_merges = gguf_arr(M, "tokenizer.ggml.merges", &mrg);
    int64_t nmerg = have_merges ? mrg.left : 0;
    /* sizing del pool: token (+NUL) e chiavi merges ("left\0right" = len) */
    int64_t psz = 0, l;
    garr it = toks; while (garr_next_str(&it, &l)) psz += l + 1;
    if (have_merges) { it = mrg; while (garr_next_str(&it, &l)) psz += l; }
    T->pool_len = psz; T->pool_off = 0;
    T->pool = malloc(psz > 0 ? psz : 1);
    if (!T->pool) { fprintf(stderr, "[tok] OOM pool %lld\n", (long long)psz); exit(1); }
    T->n_ids = (int)nvoc;
    T->id2str = calloc(T->n_ids, sizeof(char*));
    T->id_added = calloc(T->n_ids, sizeof(int));
    int vc = 1; while (vc < nvoc*2) vc <<= 1;
    hm_init(&T->vocab, vc);
    int nsp = 0;
    for (int64_t id = 0; id < nvoc; id++) {
        const char *s = garr_next_str(&toks, &l);
        if (!s) { fprintf(stderr, "[tok] GGUF: array tokens troncato\n"); exit(1); }
        char *k = tk_pool_dup(T, s, (int)l);
        hm_put(&T->vocab, k, (int)l, (int)id);
        T->id2str[id] = k;
        int tt = have_types ? (int)garr_next_int(&types) : 1;
        if (tt == 3) { T->id_added[id] = 1; nsp++; }   /* CONTROL -> added token */
    }
    int mc = 1; while (mc < (nmerg > 0 ? nmerg : 1)*2) mc <<= 1;
    hm_init(&T->merges, mc);
    if (have_merges) for (int64_t i = 0; i < nmerg; i++) {
        const char *s = garr_next_str(&mrg, &l);
        const char *sp = s ? memchr(s, ' ', l) : NULL;
        if (!sp) { fprintf(stderr, "[tok] GGUF: merge senza spazio\n"); exit(1); }
        int ll = (int)(sp - s), rl = (int)(l - ll - 1);
        char *key = tk_pool_take(T, ll + 1 + rl);
        memcpy(key, s, ll); key[ll] = 0; memcpy(key+ll+1, sp+1, rl);
        hm_put(&T->merges, key, ll+1+rl, (int)i);
    }
    /* added token (CONTROL), ordinati per lunghezza decrescente */
    T->nsp = nsp; T->sp = calloc(nsp > 0 ? nsp : 1, sizeof(Special));
    int si = 0;
    for (int id = 0; id < T->n_ids; id++)
        if (T->id_added[id]) {
            T->sp[si].str = T->id2str[id];
            T->sp[si].len = (int)strlen(T->id2str[id]);
            T->sp[si].id = id; si++;
        }
    qsort(T->sp, T->nsp, sizeof(Special), cmp_sp_len);
    tk_byte_tables(T);
    fprintf(stderr, "[tok] GGUF: %d token, %lld merges, %d added\n",
            T->n_ids, (long long)nmerg, T->nsp);
}
#endif /* GGUF_H */

/* ---------- nucleo BPE: merge greedy per rank su una stringa di simboli.
 * sp_fallback=0 (byte-level): simboli fuori vocab vengono scartati.
 * sp_fallback=1 (sentencepiece): ogni byte grezzo del simbolo -> <0xXX>. */
static void bpe_core(Tok *T, const char *s, int sl, int *out, int *no, int max, int sp_fallback){
    /* ignore_merges: se l'intero pezzo e' un token, emettilo diretto */
    int whole=hm_get(&T->vocab,s,sl);
    if(whole>=0){ if(*no<max) out[(*no)++]=whole; return; }
    int *soff=malloc((sl+1)*sizeof(int)), *slen=malloc((sl+1)*sizeof(int)); int ns=0;
    for(int i=0;i<sl;){ uint32_t cp; int k=u8_next((const unsigned char*)s,sl,i,&cp);
        soff[ns]=i; slen[ns]=k; ns++; i+=k; }
    char *kbuf=malloc(2*sl+2);
    for(;;){
        int best=INT_MAX, bp=-1;
        for(int i=0;i+1<ns;i++){
            int ll=slen[i], rl=slen[i+1];
            memcpy(kbuf,s+soff[i],ll); kbuf[ll]=0; memcpy(kbuf+ll+1,s+soff[i+1],rl);
            int rk=hm_get(&T->merges,kbuf,ll+1+rl);
            if(rk>=0 && rk<best){ best=rk; bp=i; }
        }
        if(bp<0) break;
        slen[bp]=soff[bp+1]+slen[bp+1]-soff[bp];          /* fonde bp e bp+1 (contigui in s) */
        for(int j=bp+1;j<ns-1;j++){ soff[j]=soff[j+1]; slen[j]=slen[j+1]; }
        ns--;
    }
    for(int i=0;i<ns;i++){
        int id=hm_get(&T->vocab,s+soff[i],slen[i]);
        if(id>=0){ if(*no<max) out[(*no)++]=id; continue; }
        if(sp_fallback){
            for(int j=0;j<slen[i];j++){
                unsigned char bb=(unsigned char)s[soff[i]+j];
                if(T->byte_tok[bb]>=0){ if(*no<max) out[(*no)++]=T->byte_tok[bb]; }
                else fprintf(stderr,"[tok] byte 0x%02X senza fallback nel vocab\n",bb);
            }
        }
    }
    free(soff); free(slen); free(kbuf);
}

/* ---------- BPE byte-level su un pezzo: byte grezzi [a,b) -> id appesi a out ---------- */
static void bpe_piece(Tok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int nb=b-a;
    /* stringa byte-level (concatenazione di byte2str): <=2 byte per byte di input */
    char *s=malloc(2*nb+1); int sl=0;
    for(int i=a;i<b;i++){ int bb=p[i]; memcpy(s+sl,T->byte2str[bb],T->byte2cp_len[bb]); sl+=T->byte2cp_len[bb]; }
    s[sl]=0;
    bpe_core(T,s,sl,out,no,max,0);
    free(s);
}

/* ---------- pezzo sentencepiece: [a,b) letterale, spazio -> U+2581 ----------
 * is_start prefissa il metaspazio se il normalizer ha Prepend (add_dummy_prefix).
 * BPE sull'INTERO pezzo (niente pre-split): corretto per ogni vocab SP, anche
 * con token multi-parola; i pezzi sono comunque delimitati dagli added-token. */
static void sp_piece(Tok *T, const unsigned char *p, int a, int b, int is_start, int *out, int *no, int max){
    int nb=b-a;
    char *s=malloc(3*(int64_t)nb+4); int sl=0;
    if(is_start && T->add_dummy_prefix){ memcpy(s+sl,"\xE2\x96\x81",3); sl+=3; }
    for(int i=a;i<b;i++){
        if(p[i]==' '){ memcpy(s+sl,"\xE2\x96\x81",3); sl+=3; }
        else s[sl++]=(char)p[i];
    }
    s[sl]=0;
    bpe_core(T,s,sl,out,no,max,1);
    free(s);
}

/* ---------- pre-tokenizer regex (pattern cl100k) su una porzione di testo ----------
 * Decodifica i codepoint, applica le alternative IN ORDINE, e per ogni pezzo chiama bpe_piece. */
static void pretok_chunk(Tok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int nb=b-a; if(nb<=0) return;
    uint32_t *cp=malloc((nb+1)*sizeof(uint32_t)); int *off=malloc((nb+2)*sizeof(int)); int n=0;
    for(int i=a;i<b;){ uint32_t c; int k=u8_next(p,b,i,&c); off[n]=i; cp[n]=c; n++; i+=k; }
    off[n]=b;
    #define ISNL(c) ((c)=='\r'||(c)=='\n')
    #define LOW(c) (((c)>='A'&&(c)<='Z')?((c)+32):(c))
    int i=0;
    while(i<n){
        int start=i; uint32_t c=cp[i];
        /* 1) (?i:'s|'t|'re|'ve|'m|'ll|'d) */
        if(c=='\'' && i+1<n){
            uint32_t d=LOW(cp[i+1]);
            if(i+2<n){ uint32_t d2=LOW(cp[i+2]);
                if((d=='r'&&d2=='e')||(d=='v'&&d2=='e')||(d=='l'&&d2=='l')){ i+=3; bpe_piece(T,p,off[start],off[i],out,no,max); continue; } }
            if(d=='s'||d=='t'||d=='m'||d=='d'){ i+=2; bpe_piece(T,p,off[start],off[i],out,no,max); continue; }
        }
        /* 2) [^\r\n\p{L}\p{N}]? \p{L}+ */
        {
            int j=i;
            if(!is_L(c) && !ISNL(c) && !is_N(c)){ if(j+1<n && is_L(cp[j+1])) j++; else j=-1; }
            if(j>=0){
                if(is_L(cp[j])){ while(j<n && is_L(cp[j])) j++; i=j; bpe_piece(T,p,off[start],off[i],out,no,max); continue; }
            }
        }
        /* 3) \p{N}{1,3} */
        if(is_N(c)){ int j=i,k=0; while(j<n && is_N(cp[j]) && k<3){ j++; k++; } i=j; bpe_piece(T,p,off[start],off[i],out,no,max); continue; }
        /* 4) ' ?[^\s\p{L}\p{N}]+[\r\n]*' */
        {
            int j=i;
            if(c==' ' && j+1<n && !is_S(cp[j+1]) && !is_L(cp[j+1]) && !is_N(cp[j+1])) j++;
            if(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])){
                while(j<n && !is_S(cp[j]) && !is_L(cp[j]) && !is_N(cp[j])) j++;
                while(j<n && ISNL(cp[j])) j++;
                i=j; bpe_piece(T,p,off[start],off[i],out,no,max); continue;
            }
        }
        /* 5) \s*[\r\n]+  -> run di whitespace fino all'ultimo newline contiguo */
        {
            int r=i; while(r<n && is_S(cp[r])) r++;
            if(r>i){ int last=-1; for(int j=i;j<r;j++) if(ISNL(cp[j])) last=j;
                if(last>=0){ i=last+1; bpe_piece(T,p,off[start],off[i],out,no,max); continue; }
                /* 6) \s+(?!\S): se seguito da non-spazio lascia l'ultimo ws, altrimenti prendi tutto */
                int end = (r<n) ? r-1 : r;
                if(end<=i) end=i+1;                 /* \s+ minimo 1 (fallback alt 7) */
                i=end; bpe_piece(T,p,off[start],off[i],out,no,max); continue;
            }
        }
        i++;  /* salvagente: non dovrebbe accadere */
        bpe_piece(T,p,off[start],off[i],out,no,max);
    }
    #undef ISNL
    #undef LOW
    free(cp); free(off);
}

/* ---------- encode: testo -> id (split sugli added token, poi pretok+BPE) ---------- */
static int tok_encode(Tok *T, const char *text, int len, int *out, int max){
    const unsigned char *p=(const unsigned char*)text; int no=0; int i=0;
    while(i<len){
        /* prossima occorrenza di un added-token a partire da >= i (match piu' lungo) */
        int hitpos=-1, hitlen=0, hitid=-1;
        for(int j=i;j<len && hitpos<0;j++){
            for(int k=0;k<T->nsp;k++){
                int sl=T->sp[k].len;
                if(sl>0 && j+sl<=len && !memcmp(p+j,T->sp[k].str,sl)){ hitpos=j; hitlen=sl; hitid=T->sp[k].id; break; }
            }
        }
        int chunk_end = (hitpos<0) ? len : hitpos;
        if(chunk_end>i){
            if(T->mode==1) sp_piece(T,p,i,chunk_end,i==0,out,&no,max);
            else pretok_chunk(T,p,i,chunk_end,out,&no,max);
        }
        if(hitpos<0) break;
        if(no<max) out[no++]=hitid;
        i=hitpos+hitlen;
    }
    return no;
}

/* id di un added-token dato il suo contenuto (es. "<|endoftext|>"); -1 se assente */
static int tok_id_of(Tok *T, const char *content){
    for(int i=0;i<T->nsp;i++) if(!strcmp(T->sp[i].str,content)) return T->sp[i].id;
    return -1;
}

/* ---------- decode: id -> testo (byte-level inverso oppure sp; added letterali) ---------- */
static int tok_decode(Tok *T, const int *ids, int n, char *out, int max){
    int o=0;
    for(int i=0;i<n;i++){
        int id=ids[i]; if(id<0||id>=T->n_ids||!T->id2str[id]) continue;
        const char *s=T->id2str[id];
        if(T->id_added[id]){ int l=(int)strlen(s); for(int j=0;j<l && o<max;j++) out[o++]=s[j]; continue; }
        int sl=(int)strlen(s);
        if(T->mode==1){
            if(T->id2byte && T->id2byte[id]>=0){ if(o<max) out[o++]=(char)(unsigned char)T->id2byte[id]; continue; }
            /* letterale con metaspazio U+2581 -> spazio */
            for(int j=0;j<sl;){
                if(j+3<=sl && (unsigned char)s[j]==0xE2 && (unsigned char)s[j+1]==0x96 && (unsigned char)s[j+2]==0x81){
                    if(o<max) out[o++]=' '; j+=3;
                } else { if(o<max) out[o++]=s[j]; j++; }
            }
            continue;
        }
        for(int j=0;j<sl;){ uint32_t c; int k=u8_next((const unsigned char*)s,sl,j,&c); j+=k;
            if(c<1024 && T->cp2byte[c]>=0 && o<max) out[o++]=(char)(unsigned char)T->cp2byte[c]; }
    }
    if(o<max) out[o]=0;
    return o;
}

#endif
