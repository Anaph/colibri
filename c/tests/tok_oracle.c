/* Harness di parita' del tokenizer C contro un oracolo HF (corpus-scale).
 * uso:  ./tok_oracle <tokenizer.json> < cases
 * dove cases = righe "TEXT\tID,ID,.." (TEXT con \n \t \r \\ escapati).
 * Stampa ENCODE m/n e DECODE round-trip; exit 0 solo con match completo.
 * E' lo strumento primario di bring-up per nuovi tokenizer (es. SP di Gemma). */
#define _GNU_SOURCE
#include "../tok.h"

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"usage: %s tokenizer.json < cases\n",argv[0]); return 1; }
    Tok T;
    tok_load(&T, argv[1]);
    fprintf(stderr,"loaded: vocab_ids=%d specials=%d\n", T.n_ids, T.nsp);
    char *line=NULL; size_t cap=0; ssize_t nr;
    int pass=0, tot=0, dpass=0;
    while((nr=getline(&line,&cap,stdin))>=0){
        if(nr>0 && line[nr-1]=='\n'){ line[--nr]=0; }
        if(nr==0) continue;
        char *tab=strchr(line,'\t'); if(!tab) continue;
        *tab=0; const char *text=line; const char *idstr=tab+1;
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
