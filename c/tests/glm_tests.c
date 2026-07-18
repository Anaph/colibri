/* Logica dei test del motore glm (C puro; glue gtest in glm_gtest.cc).
 * Ritorni: 0=ok, 1=fail, 2=skip. Include glm.c intero (main rinominato). */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define main coli_glm_main_unused
#include "../glm.c"
#undef main

static uint32_t gt_rng=0x12345678u;
static uint32_t gt_xr(void){ gt_rng^=gt_rng<<13; gt_rng^=gt_rng>>17; gt_rng^=gt_rng<<5; return gt_rng; }

static int32_t ref_i8i8(const int8_t *w, const int8_t *x, int I){
    int64_t s=0; for(int i=0;i<I;i++) s+=(int32_t)w[i]*x[i]; return (int32_t)s;
}
static int32_t ref_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int64_t s=0;
    for(int i=0;i<I;i++){ uint8_t b=w4[i>>1]; int v=(i&1)?((int)(b>>4)-8):((int)(b&0xF)-8); s+=v*x[i]; }
    return (int32_t)s;
}

/* dot_i8i8/dot_i4i8: uguaglianza ESATTA col riferimento C, qualunque SIMD */
int gt_idot_kernel(void){
    static const int sizes[]={1,2,15,16,17,31,32,33,63,64,65,100,127,128,1408,4096,4097};
    static int8_t w[8192], x[8192]; static uint8_t w4[4096];
    gt_rng=0x12345678u;
    for(unsigned t=0;t<sizeof(sizes)/sizeof(sizes[0]);t++){
        int I=sizes[t];
        for(int rep=0;rep<64;rep++){
            for(int i=0;i<I;i++) x[i]=(int8_t)((int)(gt_xr()%255)-127);
            for(int i=0;i<I;i++) w[i]=(int8_t)((int)(gt_xr()%256)-128);
            if(rep==0) for(int i=0;i<I;i++) w[i]=-128;
            if(rep==1) for(int i=0;i<I;i++){ w[i]=127; x[i]=(int8_t)(i&1?-127:127); }
            for(int i=0;i<(I+1)/2;i++) w4[i]=(uint8_t)(gt_xr()&0xFF);
            int32_t got=dot_i8i8(w,x,I), want=ref_i8i8(w,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i8i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
            got=dot_i4i8(w4,x,I); want=ref_i4i8(w4,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i4i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
        }
    }
    fprintf(stderr,"idot kernel exactness (%s): ok\n", IDOT_KERNEL);
    return 0;
}

/* driver matmul_qt_ex bit-esatto contro il riferimento C */
static void fill_qt(QT *w, int fmt, int O, int I){
    memset(w,0,sizeof *w);
    w->fmt=fmt; w->O=O; w->I=I;
    w->s=malloc((size_t)O*sizeof(float));
    for(int o=0;o<O;o++) w->s[o]=0.001f+(float)(gt_xr()%1000)*1e-6f;
    if(fmt==1){
        w->q8=malloc((size_t)O*I);
        for(int64_t i=0;i<(int64_t)O*I;i++) w->q8[i]=(int8_t)((int)(gt_xr()%256)-128);
    }else{
        size_t nb=(size_t)O*((I+1)/2);
        w->q4=malloc(nb);
        for(size_t i=0;i<nb;i++) w->q4[i]=(uint8_t)(gt_xr()&0xFF);
    }
}
static int check_driver(int fmt,int O,int I,int S){
    QT w; fill_qt(&w,fmt,O,I); int rb=(I+1)/2;
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y=malloc((size_t)S*O*sizeof(float));
    float *yref=malloc((size_t)S*O*sizeof(float));
    int8_t *xqr=malloc((size_t)S*I);
    float *sxr=malloc((size_t)S*sizeof(float));
    for(int64_t i=0;i<(int64_t)S*I;i++) x[i]=((float)(gt_xr()%4001)-2000.f)/500.f;
    matmul_qt_ex(y,x,&w,S,1);
    for(int s=0;s<S;s++) sxr[s]=qrow_i8(x+(int64_t)s*I, xqr+(int64_t)s*I, I);
    for(int o=0;o<O;o++) for(int s=0;s<S;s++){
        int32_t d=fmt==1 ? ref_i8i8(w.q8+(int64_t)o*I, xqr+(int64_t)s*I, I)
                         : ref_i4i8(w.q4+(int64_t)o*rb, xqr+(int64_t)s*I, I);
        yref[(int64_t)s*O+o]=(float)d*w.s[o]*sxr[s];
    }
    int rc=0;
    for(int64_t i=0;i<(int64_t)S*O;i++)
        if(memcmp(&y[i],&yref[i],sizeof(float))!=0){
            fprintf(stderr,"FAIL driver fmt=%d O=%d I=%d S=%d idx=%lld: %.9g != %.9g\n",
                    fmt,O,I,S,(long long)i,(double)y[i],(double)yref[i]); rc=1; break;
        }
    free(w.s); free(w.q8); free(w.q4); free(x); free(y); free(yref); free(xqr); free(sxr);
    return rc;
}

int gt_idot_driver(void){
    static const int Os[]={1,2,3,64,65};
    static const int Is[]={16,17,100,1408};
    static const int Ss[]={2,3,4,5,8};
    gt_rng=0x9E3779B9u;
    for(int rep=0;rep<4;rep++)
     for(unsigned a=0;a<sizeof Os/sizeof Os[0];a++)
      for(unsigned b=0;b<sizeof Is/sizeof Is[0];b++)
       for(unsigned c=0;c<sizeof Ss/sizeof Ss[0];c++)
        for(int fmt=1;fmt<=2;fmt++)
         if(check_driver(fmt,Os[a],Is[b],Ss[c])) return 1;
    fprintf(stderr,"idot driver exactness (%s): ok\n", IDOT_KERNEL);
    return 0;
}

/* kv_alloc deve sopravvivere alla ri-allocazione sullo stesso KVState */
int gt_kv_realloc(void){
    static Model m;
    memset(&m,0,sizeof m);
    m.c.n_layers=2; m.c.kv_lora=8; m.c.qk_rope=4;
    m.kv=calloc(1,sizeof(KVState));
    kv_alloc(&m,16);
    for(int i=0;i<m.c.n_layers+1;i++){ m.Lc[i][0]=1.0f; m.Rc[i][0]=1.0f; }
    kv_alloc(&m,32);
    for(int i=0;i<m.c.n_layers+1;i++){
        m.Lc[i][(int64_t)32*m.c.kv_lora-1]=2.0f;
        m.Rc[i][(int64_t)32*m.c.qk_rope-1]=2.0f;
    }
    return 0;
}

/* ---- io_uring (solo Linux; 2=skip dove il kernel/sandbox lo vieta) ---- */
#ifndef __linux__
int gt_uring(void){ return 2; }
#else
static int ur_fail(const char *s){ fprintf(stderr,"FAIL: %s\n",s); return 1; }

static int ur_expert_layout(int fd){
    Model m={0}; ESlot slot={0}; UringBatch batch={0};
    m.c.hidden=4; m.c.moe_inter=3; m.ebits=8;
    m.S.n=6; m.S.cap=6; m.S.t=calloc(6,sizeof(st_tensor));
    if(!m.S.t) return ur_fail("tensor metadata allocation");
    const char *proj[3]={"gate_proj","up_proj","down_proj"};
    int wbytes[3]={12,12,12}, sbytes[3]={12,12,16};
    unsigned char data[76];
    for(int i=0;i<36;i++) data[i]=(unsigned char)(i+1);
    float scales[10]; for(int i=0;i<10;i++) scales[i]=(float)i+0.5f;
    memcpy(data+36,scales,sizeof(scales));
    if(pwrite(fd,data,sizeof(data),0)!=(ssize_t)sizeof(data)){ free(m.S.t); return ur_fail("expert fixture write"); }
    int64_t wo=0,so=36;
    for(int k=0;k<3;k++){
        char name[300];
        snprintf(name,sizeof(name),"model.layers.1.mlp.experts.7.%s.weight",proj[k]);
        m.S.t[k]=(st_tensor){strdup(name),fd,wo,wbytes[k],3,wbytes[k]}; wo+=wbytes[k];
        size_t n=strlen(name); memcpy(name+n,".qs",4);
        m.S.t[3+k]=(st_tensor){strdup(name),fd,so,sbytes[k],2,sbytes[k]/4}; so+=sbytes[k];
    }
    if(uring_batch_init(&batch)){ free(m.S.t); return ur_fail("expert ring init"); }
    uring_batch_reset(&batch);
    int li=uring_load_add(&batch,&m,1,7,&slot,1);
    if(li!=0 || uring_submit_batch(&batch) || uring_finalize_load(&batch,li,1)){
        coli_uring_close(&batch.ring); free(m.S.t); return ur_fail("expert batch load");
    }
    int bad=slot.eid!=7 || slot.g.fmt!=1 || slot.u.fmt!=1 || slot.d.fmt!=1
        || memcmp(slot.g.q8,data,12) || memcmp(slot.u.q8,data+12,12) || memcmp(slot.d.q8,data+24,12)
        || memcmp(slot.g.s,scales,12) || memcmp(slot.u.s,scales+3,12) || memcmp(slot.d.s,scales+6,16);
    coli_uring_close(&batch.ring);
    compat_aligned_free(slot.slab); free(slot.fslab);
    if(bad){ for(int i=0;i<m.S.n;i++) free(m.S.t[i].name); free(m.S.t); return ur_fail("expert tensor views"); }

    m.c.n_experts=8; m.c.n_layers=2; m.ecap=2;
    m.pin=calloc(3,sizeof(ESlot*)); m.npin=calloc(3,sizeof(int));
    m.ecache=calloc(3,sizeof(ESlot*)); m.ecn=calloc(3,sizeof(int));
    m.ecache[1]=calloc(2,sizeof(ESlot));
    if(!m.pin||!m.npin||!m.ecache||!m.ecn||!m.ecache[1])
        return ur_fail("pilot fixture allocation");
    if(uring_batch_init(&g_ub_pilot)) return ur_fail("pilot ring init");
    memset(g_pilot_inflight,0,sizeof(g_pilot_inflight));
    atomic_store(&g_cur_moe_layer,-1); atomic_store(&g_pilot_loads,0); atomic_store(&g_pilot_drops,0);
    pilot_r=0; pilot_w=1; pilot_q[0].l=1; pilot_q[0].e=7;
    pilot_uring_batch(&m);
    bad=m.ecn[1]!=1 || m.ecache[1][0].eid!=7 || g_pilot_inflight[1]!=0
        || atomic_load(&g_pilot_loads)!=1 || atomic_load(&g_pilot_drops)!=0;
    coli_uring_close(&g_ub_pilot.ring);
    compat_aligned_free(m.ecache[1][0].slab); free(m.ecache[1][0].fslab);
    free(m.ecache[1]);
    free(m.pin); free(m.npin); free(m.ecache); free(m.ecn);
    for(int i=0;i<m.S.n;i++) free(m.S.t[i].name);
    free(m.S.t);
    return bad?ur_fail("pilot uring publication"):0;
}

int gt_uring(void){
    char path[]="/tmp/coli-uring-XXXXXX";
    int fd=mkstemp(path); if(fd<0) return ur_fail("mkstemp");
    unlink(path);
    enum { N=4, SZ=4096 };
    static unsigned char src[N][SZ],dst[N][SZ];
    for(int i=0;i<N;i++){
        memset(src[i],0,sizeof(src[i])); memset(dst[i],0,sizeof(dst[i]));
        for(int j=0;j<SZ;j++) src[i][j]=(unsigned char)(i*37+j);
        if(pwrite(fd,src[i],SZ,(off_t)i*SZ)!=SZ){ close(fd); return ur_fail("pwrite fixture"); }
    }
    ColiUring ring;
    if(coli_uring_init(&ring,8)){
        if(errno==EPERM || errno==ENOSYS || errno==EACCES){
            fprintf(stderr,"uring: skipped (%s)\n",strerror(errno)); close(fd); return 2;
        }
        close(fd); return ur_fail("io_uring_setup");
    }
    if(coli_uring_set_workers(&ring,4)){
        coli_uring_close(&ring); close(fd); return ur_fail("io-wq worker limit");
    }
    for(int i=0;i<N;i++) if(coli_uring_prep_read(&ring,fd,dst[i],SZ,(off_t)i*SZ,(uint64_t)i+1)){
        coli_uring_close(&ring); close(fd); return ur_fail("prepare read");
    }
    if(coli_uring_enter(&ring,0)<0){ coli_uring_close(&ring); close(fd); return ur_fail("submit"); }
    int seen[N]={0},complete=0;
    while(complete<N){
        struct io_uring_cqe cqe; int reaped=0;
        while(coli_uring_peek(&ring,&cqe)){
            reaped=1;
            if(cqe.user_data<1 || cqe.user_data>N || cqe.res!=SZ){
                coli_uring_close(&ring); close(fd); return ur_fail("bad completion");
            }
            int i=(int)cqe.user_data-1;
            if(seen[i]++){ coli_uring_close(&ring); close(fd); return ur_fail("duplicate completion"); }
            complete++;
        }
        if(!reaped && complete<N && coli_uring_enter(&ring,1)<0){
            coli_uring_close(&ring); close(fd); return ur_fail("completion wait");
        }
    }
    for(int i=0;i<N;i++) if(memcmp(src[i],dst[i],SZ)){
        coli_uring_close(&ring); close(fd); return ur_fail("read data mismatch");
    }
    coli_uring_close(&ring);
    if(ftruncate(fd,0) || ur_expert_layout(fd)){ close(fd); return 1; }
    close(fd);
    return 0;
}
#endif /* __linux__ */
