/* Validazione del tokenizer C.
 * build da c/: make tests/test_tok
 * uso:  ./tests/test_tok <tokenizer.json> < cases   (righe "TEXT\tID,ID,..")
 *       ./tests/test_tok                            (fixture integrata)
 */
#define _GNU_SOURCE
#include "../tok.h"

/* --- fixture integrata: micro-vocabolario in entrambi i formati merges ---
 * vocab: h,e,l,o,Ġ,he,ll,hell,hello + speciale <|endoftext|>
 * "hello hello<|endoftext|>" -> [hello][Ġ][h][e][ll][o][eot]  (spazio: "Ġhello"
 * non e' nel vocab, quindi il secondo hello resta spezzato: Ġ h e ll o... in
 * realta' i merges lo ricompongono solo dove il rank lo permette) */
static const char *FIX_PAIRS =
  "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"h\":0,\"e\":1,\"l\":2,\"o\":3,\"\\u0120\":4,"
  "\"he\":5,\"ll\":6,\"hell\":7,\"hello\":8},"
  "\"merges\":[[\"h\",\"e\"],[\"l\",\"l\"],[\"he\",\"ll\"],[\"hell\",\"o\"]]},"
  "\"added_tokens\":[{\"id\":9,\"content\":\"<|endoftext|>\"}]}";
static const char *FIX_STRINGS =
  "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"h\":0,\"e\":1,\"l\":2,\"o\":3,\"\\u0120\":4,"
  "\"he\":5,\"ll\":6,\"hell\":7,\"hello\":8},"
  "\"merges\":[\"h e\",\"l l\",\"he ll\",\"hell o\"]},"
  "\"added_tokens\":[{\"id\":9,\"content\":\"<|endoftext|>\"}]}";

static void write_tmp(const char *path, const char *body){
    FILE *f=fopen(path,"wb"); if(!f){ perror(path); exit(1); }
    fwrite(body,1,strlen(body),f); fclose(f);
}

static int run_fixture(const char *name, const char *body){
    char path[512];
    const char *tmp=getenv("TMPDIR"); if(!tmp) tmp="/tmp";
    snprintf(path,sizeof(path),"%s/tok_fix_%s.json",tmp,name);
    write_tmp(path,body);
    Tok T; tok_load(&T,path);
    int fail=0;

    /* "hello" -> merges completi -> un token */
    int ids[64]; int n=tok_encode(&T,"hello",5,ids,64);
    if(!(n==1 && ids[0]==8)){ fprintf(stderr,"[%s] encode(hello): atteso [8], got n=%d\n",name,n); fail=1; }

    /* added token atomico + testo intorno */
    n=tok_encode(&T,"hello<|endoftext|>hello",23,ids,64);
    if(!(n==3 && ids[0]==8 && ids[1]==9 && ids[2]==8)){
        fprintf(stderr,"[%s] encode(hello<eot>hello): got n=%d [",name,n);
        for(int i=0;i<n;i++)fprintf(stderr," %d",ids[i]); fprintf(stderr," ]\n"); fail=1;
    }

    /* round-trip decode del testo normale */
    n=tok_encode(&T,"hello",5,ids,64);
    char dec[64]; int dn=tok_decode(&T,ids,n,dec,63);
    if(!(dn==5 && !memcmp(dec,"hello",5))){ fprintf(stderr,"[%s] decode round-trip fallito\n",name); fail=1; }

    remove(path);
    if(!fail) fprintf(stderr,"[%s] ok\n",name);
    return fail;
}

static int run_oracle(const char *tokpath){
    Tok T;
    tok_load(&T, tokpath);
    fprintf(stderr,"loaded: vocab_ids=%d specials=%d\n", T.n_ids, T.nsp);
    char *line=NULL; size_t cap=0; ssize_t nr;
    int pass=0, tot=0, dpass=0;
    while((nr=getline(&line,&cap,stdin))>=0){
        if(nr>0 && line[nr-1]=='\n'){ line[--nr]=0; }
        if(nr==0) continue;
        char *tab=strchr(line,'\t'); if(!tab) continue;
        *tab=0; const char *text=line; const char *idstr=tab+1;
        /* il testo puo' contenere \n e \t codificati come \\n \\t */
        char tbuf[4096]; int tn=0;
        for(const char *q=text; *q && tn<4095; q++){
            if(q[0]=='\\' && q[1]=='n'){ tbuf[tn++]='\n'; q++; }
            else if(q[0]=='\\' && q[1]=='t'){ tbuf[tn++]='\t'; q++; }
            else if(q[0]=='\\' && q[1]=='r'){ tbuf[tn++]='\r'; q++; }
            else if(q[0]=='\\' && q[1]=='\\'){ tbuf[tn++]='\\'; q++; }
            else tbuf[tn++]=*q;
        }
        tbuf[tn]=0;
        int exp[4096], ne=0;
        for(const char *q=idstr; *q; ){ while(*q==','||*q==' ')q++; if(!*q)break; exp[ne++]=atoi(q); while(*q&&*q!=',')q++; }
        int got[4096]; int ng=tok_encode(&T,tbuf,tn,got,4096);
        int ok = (ng==ne); for(int i=0;i<ng&&ok;i++) ok = (got[i]==exp[i]);
        tot++; if(ok) pass++;
        /* round-trip decode */
        char dec[8192]; int dn=tok_decode(&T,got,ng,dec,8191);
        int drt = (dn==tn) && !memcmp(dec,tbuf,tn);
        if(drt) dpass++;
        if(!ok || !drt){
            fprintf(stderr,"MISMATCH text=%s\n  exp(%d):",text,ne); for(int i=0;i<ne;i++)fprintf(stderr," %d",exp[i]);
            fprintf(stderr,"\n  got(%d):",ng); for(int i=0;i<ng;i++)fprintf(stderr," %d",got[i]);
            fprintf(stderr,"\n  decode_ok=%d\n", drt);
        }
    }
    printf("ENCODE: %d/%d  DECODE(round-trip): %d/%d\n", pass,tot, dpass,tot);
    return pass==tot ? 0 : 2;
}

int main(int argc, char **argv){
    if(argc>=2) return run_oracle(argv[1]);
    int fail = 0;
    fail |= run_fixture("pairs",   FIX_PAIRS);    /* formato GLM: coppie */
    fail |= run_fixture("strings", FIX_STRINGS);  /* formato Qwen: "a b" */
    if(!fail) printf("test_tok: ok\n");
    return fail;
}
