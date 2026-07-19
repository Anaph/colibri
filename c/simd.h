/* Kernel SIMD condivisi (una sola scala di #ifdef, niente dispatch a runtime):
 * qrow_i8 + dot_i8i8 sono portati pari pari dalla famiglia int8 collaudata di
 * glm.c (trucco del segno AVX512-VNNI, AVX-VNNI, maddubs AVX2, SDOT NEON, VSX,
 * coda scalare). Nuovi qui: dot_f32 (la riduzione f32 dei GEMV non si
 * auto-vettorizza senza -ffast-math) e i due kernel di riga del DeltaNet.
 * Autosufficiente: solo stdint/math + header intrinsics sotto guardia. */
#ifndef SIMD_H
#define SIMD_H
#include <stdint.h>
#include <math.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__VSX__)
#include <altivec.h>
#undef vector                     /* igiene: si usano __vector/__bool espliciti */
#undef pixel
#undef bool
#endif

/* nome del kernel int8 compilato (stessa semantica della scala di glm.c;
 * niente ramo i8mm qui: e' legato ai driver tiled di glm e non viene copiato) */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#define IDOT_KERNEL "avx512-vnni"
#elif defined(__AVXVNNI__) && defined(__AVX2__)
#define IDOT_KERNEL "avx-vnni"
#elif defined(__AVX2__)
#define IDOT_KERNEL "avx2"
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define IDOT_KERNEL "neon-dotprod"
#elif defined(__ARM_NEON)
#define IDOT_KERNEL "neon"
#elif defined(__VSX__)
#define IDOT_KERNEL "vsx"
#else
#define IDOT_KERNEL "scalar"
#endif

/* nome del kernel f32 compilato */
#if defined(__AVX512F__)
#define F32_KERNEL "avx512f"
#elif defined(__AVX2__) && defined(__FMA__)
#define F32_KERNEL "avx2-fma"
#elif defined(__ARM_NEON)
#define F32_KERNEL "neon"
#else
#define F32_KERNEL "scalar"
#endif

/* quantizzazione simmetrica per riga stile Q8_0 (absmax/127); ritorna la scala */
static inline float qrow_i8(const float *x, int8_t *q, int n){
    float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<n;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}

#ifdef __AVX2__
static inline int simd_hsum256_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
/* somma orizzontale di 8 float (stessa sequenza di hsum256 di glm.c);
 * esportata perche' i matmul int4 di nn.h accumulano in __m256 */
static inline float simd_hsum256_f32(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#endif
#if defined(__AVXVNNI__) && defined(__AVX2__)
/* hsum di un __m128i a 4 lane s32 (l'AVX-VNNI 128-bit accumula su 4 lane). */
static inline int simd_hsum128_i32(__m128i v){
    v=_mm_hadd_epi32(v,v); v=_mm_hadd_epi32(v,v); return _mm_cvtsi128_si32(v);
}
#endif

/* dot int8·int8: trucco del segno (|w| unsigned × x·sign(w) signed). Sicuro:
 * |x|<=127 da qrow_i8, coppie <= 128*127*2 = 32512 < 32767, accumulo s32
 * fino a n=16384. */
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int n){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* VNNI: vpdpbusd u8*s8 -> s32 diretto, 64 byte/iter, niente intermedio a
     * 16 bit. AVX-512 non ha vpsignb: |w| via abs, segno piegato in x con un
     * mask-negate (w==0 -> prodotto 0 in ogni caso). |x|<=127 (qrow_i8),
     * |w|<=128 come u8: ogni lane s32 somma <= 4*128*127, sicuro fino a
     * n=16384 come il bound AVX2. */
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=n;i+=64){
        __m512i wv=_mm512_loadu_si512((const void*)(w+i));
        __m512i xv=_mm512_loadu_si512((const void*)(x+i));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVXVNNI__) && defined(__AVX2__)
    /* AVX-VNNI 128-bit: vpdpbusd u8*s8 -> s32, 16 byte/iter. Stesso trucco del
     * segno della variante 512-bit: |w| via abs, segno piegato in x
     * (w==0 -> prodotto 0). __AVX2__ serve per _mm_sign_epi8 / abs. */
    __m128i acc=_mm_setzero_si128();
    for(;i+16<=n;i+=16){
        __m128i wv=_mm_loadu_si128((const __m128i*)(w+i));
        __m128i xv=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i xs=_mm_sign_epi8(xv,wv);              /* x * sign(w) */
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(wv),xs);
    }
    sum=simd_hsum128_i32(acc);
#elif defined(__AVX2__)
    __m256i acc=_mm256_setzero_si256(); const __m256i ones=_mm256_set1_epi16(1);
    for(;i+32<=n;i+=32){
        __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=simd_hsum256_i32(acc);
#elif defined(__ARM_NEON)
    /* ARM: SDOT nativo se disponibile (Apple Silicon: sempre); altrimenti
     * vmull/vpadal. Stesso bound anti-overflow: coppie <= 128*127*2 < 32767. */
#if defined(__ARM_FEATURE_DOTPROD)
    /* 4 accumulatori indipendenti: SDOT ha latenza ~3-4 cicli, con un solo acc
     * la catena seriale strozza il core; con 4 lane indipendenti il dot diventa
     * memory-bound (misurato in glm.c su M4: 26 -> 63 GB/s per core, 2.4x). */
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
    for(;i+64<=n;i+=64){
        a0=vdotq_s32(a0,vld1q_s8(w+i),   vld1q_s8(x+i));
        a1=vdotq_s32(a1,vld1q_s8(w+i+16),vld1q_s8(x+i+16));
        a2=vdotq_s32(a2,vld1q_s8(w+i+32),vld1q_s8(x+i+32));
        a3=vdotq_s32(a3,vld1q_s8(w+i+48),vld1q_s8(x+i+48));
    }
    int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
    for(;i+16<=n;i+=16) acc=vdotq_s32(acc,vld1q_s8(w+i),vld1q_s8(x+i));
    sum=vaddvq_s32(acc);
#else
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+16<=n;i+=16){
        int8x16_t wv=vld1q_s8(w+i), xv=vld1q_s8(x+i);
        int16x8_t p=vmull_s8(vget_low_s8(wv),vget_low_s8(xv));
        p=vmlal_s8(p,vget_high_s8(wv),vget_high_s8(xv));
        acc=vpadalq_s16(acc,p);
    }
    sum=vaddvq_s32(acc);
#endif
#elif defined(__VSX__)
    /* POWER8: vec_msum (s8 x u8 -> s32) somma i prodotti byte DIRETTAMENTE in
     * lane s32, 16 byte/iter. Stesso trucco del segno, ma |w| via select+sub
     * MODULO e non vec_abs: -128 deve diventare 128 u8, non saturare a 127.
     * |x|<=127 da qrow_i8, quindi negare x e' sicuro. */
    __vector signed int acc=vec_splats(0);
    const __vector signed char vz=vec_splats((signed char)0);
    for(;i+16<=n;i+=16){
        __vector signed char wv=vec_xl(0,(const signed char*)(w+i));
        __vector signed char xv=vec_xl(0,(const signed char*)(x+i));
        __vector __bool char neg=vec_cmplt(wv,vz);
        __vector signed char xs=vec_sel(xv,vec_sub(vz,xv),neg);
        __vector unsigned char wa=(__vector unsigned char)vec_sel(wv,vec_sub(vz,wv),neg);
        acc=vec_msum(xs,wa,acc);
    }
    sum=vec_extract(acc,0)+vec_extract(acc,1)+vec_extract(acc,2)+vec_extract(acc,3);
#endif
    for(;i<n;i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}

/* dot int4(packed)·int8: nibble -> int8 [-8,7] al volo, poi lo stesso trucco
 * del segno di dot_i8i8. Portato pari pari da glm.c (validato bit-esatto la':
 * chiave la conservazione dell'ORDINE dei nibble negli unpack/zip). Layout
 * dei pesi: due valori per byte, indice pari nel nibble basso, memorizzati
 * come v+8 (0..15), stride di riga (I+1)/2. */
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* 32 bytes = 64 nibbles -> int8 in [-8,7], one vpdpbusd per 64 values.
     * 256-bit unpack leaves values in per-128-lane order [0-15][32-47]/[16-31][48-63];
     * dot pairing is order-invariant, so permute x's 128-bit blocks to match
     * instead of re-ordering w (one vpermq per iter, off the critical unpack path). */
    const __m256i m4v=_mm256_set1_epi8(0x0F);
    const __m512i b8v=_mm512_set1_epi8(8);
    const __m512i xidx=_mm512_setr_epi64(0,1,4,5,2,3,6,7);
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        __m256i z0=_mm256_unpacklo_epi8(lo,hi), z1=_mm256_unpackhi_epi8(lo,hi);
        __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(z0),z1,1),b8v);
        __m512i xv=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i)));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVXVNNI__) && defined(__AVX2__)
    /* AVX-VNNI 128-bit, int4: 16 byte = 32 nibble -> int8 [-8,7] in due half
     * (n0/n1), ciascuno alimentato a un vpdpbusd da 16 byte. Stesso unpack
     * 128-bit del ramo AVX2 sotto; 32 elementi/iter. */
    const __m128i m4=_mm_set1_epi8(0x0F); const __m128i b8=_mm_set1_epi8(8);
    __m128i acc=_mm_setzero_si128();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));   /* 16 byte = 32 nibble */
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);  /* nibble in ordine */
        __m128i w0=_mm_sub_epi8(n0,b8), w1=_mm_sub_epi8(n1,b8);
        __m128i x0=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i x1=_mm_loadu_si128((const __m128i*)(x+i+16));
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(w0),_mm_sign_epi8(x0,w0));
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(w1),_mm_sign_epi8(x1,w1));
    }
    sum=simd_hsum128_i32(acc);
#elif defined(__AVX2__)
    const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi8(8);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));   /* 16 byte = 32 nibble */
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);   /* in ordine */
        __m256i wv=_mm256_sub_epi8(_mm256_set_m128i(n1,n0),b8);
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=simd_hsum256_i32(acc);
#elif defined(__ARM_NEON)
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
#if defined(__ARM_FEATURE_DOTPROD)
    /* 4 accumulatori indipendenti (vedi dot_i8i8): spezzano la catena seriale. */
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
    for(;i+64<=I;i+=64){
        uint8x16_t byA=vld1q_u8(w4+(i>>1)), byB=vld1q_u8(w4+(i>>1)+16);
        uint8x16x2_t zA=vzipq_u8(vandq_u8(byA,m4q), vshrq_n_u8(byA,4)); /* nibble in ordine */
        uint8x16x2_t zB=vzipq_u8(vandq_u8(byB,m4q), vshrq_n_u8(byB,4));
        a0=vdotq_s32(a0,vsubq_s8(vreinterpretq_s8_u8(zA.val[0]),b8q),vld1q_s8(x+i));
        a1=vdotq_s32(a1,vsubq_s8(vreinterpretq_s8_u8(zA.val[1]),b8q),vld1q_s8(x+i+16));
        a2=vdotq_s32(a2,vsubq_s8(vreinterpretq_s8_u8(zB.val[0]),b8q),vld1q_s8(x+i+32));
        a3=vdotq_s32(a3,vsubq_s8(vreinterpretq_s8_u8(zB.val[1]),b8q),vld1q_s8(x+i+48));
    }
    int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));                          /* 16 byte = 32 nibble */
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4)); /* nibble in ordine */
        acc=vdotq_s32(acc,vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q),vld1q_s8(x+i));
        acc=vdotq_s32(acc,vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q),vld1q_s8(x+i+16));
    }
    sum=vaddvq_s32(acc);
#else
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));                          /* 16 byte = 32 nibble */
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4)); /* nibble in ordine */
        int8x16_t w0=vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q);
        int8x16_t w1=vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q);
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
        int16x8_t p=vmull_s8(vget_low_s8(w0),vget_low_s8(x0));      /* |w|<=8: nessun overflow */
        p=vmlal_s8(p,vget_high_s8(w0),vget_high_s8(x0));
        acc=vpadalq_s16(acc,p);
        p=vmull_s8(vget_low_s8(w1),vget_low_s8(x1));
        p=vmlal_s8(p,vget_high_s8(w1),vget_high_s8(x1));
        acc=vpadalq_s16(acc,p);
    }
    sum=vaddvq_s32(acc);
#endif
#elif defined(__VSX__)
    /* vec_mergeh/l su ppc64le interallacciano come unpacklo/hi x86 (verificato
     * su POWER8): i nibble escono in ordine di memoria; poi il trucco del segno. */
    const __vector unsigned char m4v=vec_splats((unsigned char)0x0F);
    const __vector unsigned char sh4=vec_splats((unsigned char)4);
    const __vector signed char b8v=vec_splats((signed char)8);
    const __vector signed char vz=vec_splats((signed char)0);
    __vector signed int acc=vec_splats(0);
    for(;i+32<=I;i+=32){
        __vector unsigned char by=vec_xl(0,w4+(i>>1));               /* 16 byte = 32 nibble */
        __vector unsigned char lo=vec_and(by,m4v), hi=vec_sr(by,sh4);
        __vector signed char w0=vec_sub((__vector signed char)vec_mergeh(lo,hi),b8v);
        __vector signed char w1=vec_sub((__vector signed char)vec_mergel(lo,hi),b8v);
        __vector signed char x0=vec_xl(0,(const signed char*)(x+i));
        __vector signed char x1=vec_xl(0,(const signed char*)(x+i+16));
        __vector __bool char n0=vec_cmplt(w0,vz), n1=vec_cmplt(w1,vz);
        acc=vec_msum(vec_sel(x0,vec_sub(vz,x0),n0),
                     (__vector unsigned char)vec_sel(w0,vec_sub(vz,w0),n0),acc);
        acc=vec_msum(vec_sel(x1,vec_sub(vz,x1),n1),
                     (__vector unsigned char)vec_sel(w1,vec_sub(vz,w1),n1),acc);
    }
    sum=vec_extract(acc,0)+vec_extract(acc,1)+vec_extract(acc,2)+vec_extract(acc,3);
#endif
    for(;i+1<I;i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}

/* dot f32·int8 (dequant-on-use, SENZA quantizzare x): usato dove l'errore di
 * quantizzazione deve restare unilaterale (score dell'attention su KV int8).
 * Corpo = inner loop di matmul_q di glm.c. */
static inline float dot_f32i8(const float *x, const int8_t *w, int n){
    float a=0.f; int i=0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc=_mm256_setzero_ps();
    for(;i+8<=n;i+=8){
        __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w+i)));
        acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i), _mm256_cvtepi32_ps(wi), acc);
    }
    a=simd_hsum256_f32(acc);
#elif defined(__ARM_NEON)
    float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
    for(;i+8<=n;i+=8){
        int16x8_t w16=vmovl_s8(vld1_s8(w+i));
        ac0=vfmaq_f32(ac0, vld1q_f32(x+i),   vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
        ac1=vfmaq_f32(ac1, vld1q_f32(x+i+4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))));
    }
    a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
    for(;i<n;i++) a+=x[i]*(float)w[i];
    return a;
}

/* dot f32·f32: due accumulatori FMA indipendenti (spezzano la catena seriale
 * della riduzione), coda scalare. E' il cuore di matmul e degli score
 * dell'attention: senza -ffast-math il compilatore non puo' vettorizzare
 * la riduzione da solo. */
static inline float dot_f32(const float *a, const float *b, int n){
    float sum=0.f; int i=0;
#if defined(__AVX512F__)
    __m512 s0=_mm512_setzero_ps(), s1=_mm512_setzero_ps();
    for(;i+32<=n;i+=32){
        s0=_mm512_fmadd_ps(_mm512_loadu_ps(a+i),   _mm512_loadu_ps(b+i),   s0);
        s1=_mm512_fmadd_ps(_mm512_loadu_ps(a+i+16),_mm512_loadu_ps(b+i+16),s1);
    }
    for(;i+16<=n;i+=16) s0=_mm512_fmadd_ps(_mm512_loadu_ps(a+i),_mm512_loadu_ps(b+i),s0);
    sum=_mm512_reduce_add_ps(_mm512_add_ps(s0,s1));
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 s0=_mm256_setzero_ps(), s1=_mm256_setzero_ps();
    for(;i+16<=n;i+=16){
        s0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),  _mm256_loadu_ps(b+i),  s0);
        s1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),s1);
    }
    for(;i+8<=n;i+=8) s0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s0);
    __m256 s=_mm256_add_ps(s0,s1);
    __m128 lo=_mm256_castps256_ps128(s), hi=_mm256_extractf128_ps(s,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); sum=_mm_cvtss_f32(lo);
#elif defined(__ARM_NEON)
    float32x4_t s0=vdupq_n_f32(0), s1=vdupq_n_f32(0);
    for(;i+8<=n;i+=8){
        s0=vfmaq_f32(s0,vld1q_f32(a+i),  vld1q_f32(b+i));
        s1=vfmaq_f32(s1,vld1q_f32(a+i+4),vld1q_f32(b+i+4));
    }
    for(;i+4<=n;i+=4) s0=vfmaq_f32(s0,vld1q_f32(a+i),vld1q_f32(b+i));
    sum=vaddvq_f32(vaddq_f32(s0,s1));
#endif
    for(;i<n;i++) sum+=a[i]*b[i];
    return sum;
}

/* ---- kernel di riga del DeltaNet (stato ricorrente S[dk][dv]) ----
 * Vettorizzano la dimensione dv con broadcast di dec/ki/qi + FMA; la coda
 * scalare copre dv non multiplo della lane. */

/* per j: S[j] *= dec; kv[j] += S[j]*ki;  (decadimento riga + accumulo k.S) */
static inline void dn_row_decay_acc(float *restrict S, float dec, float ki,
                                    float *restrict kv, int dv){
    int j=0;
#if defined(__AVX512F__)
    __m512 vd=_mm512_set1_ps(dec), vk=_mm512_set1_ps(ki);
    for(;j+16<=dv;j+=16){
        __m512 s=_mm512_mul_ps(_mm512_loadu_ps(S+j),vd);
        _mm512_storeu_ps(S+j,s);
        _mm512_storeu_ps(kv+j,_mm512_fmadd_ps(s,vk,_mm512_loadu_ps(kv+j)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 vd=_mm256_set1_ps(dec), vk=_mm256_set1_ps(ki);
    for(;j+8<=dv;j+=8){
        __m256 s=_mm256_mul_ps(_mm256_loadu_ps(S+j),vd);
        _mm256_storeu_ps(S+j,s);
        _mm256_storeu_ps(kv+j,_mm256_fmadd_ps(s,vk,_mm256_loadu_ps(kv+j)));
    }
#elif defined(__ARM_NEON)
    float32x4_t vd=vdupq_n_f32(dec), vk=vdupq_n_f32(ki);
    for(;j+4<=dv;j+=4){
        float32x4_t s=vmulq_f32(vld1q_f32(S+j),vd);
        vst1q_f32(S+j,s);
        vst1q_f32(kv+j,vfmaq_f32(vld1q_f32(kv+j),s,vk));
    }
#endif
    for(;j<dv;j++){ S[j]*=dec; kv[j]+=S[j]*ki; }
}

/* per j: S[j] += ki*delta[j]; oh[j] += S[j]*qi;  (update riga + accumulo q.S) */
static inline void dn_row_update_dot(float *restrict S, float ki,
                                     const float *restrict delta, float qi,
                                     float *restrict oh, int dv){
    int j=0;
#if defined(__AVX512F__)
    __m512 vk=_mm512_set1_ps(ki), vq=_mm512_set1_ps(qi);
    for(;j+16<=dv;j+=16){
        __m512 s=_mm512_fmadd_ps(vk,_mm512_loadu_ps(delta+j),_mm512_loadu_ps(S+j));
        _mm512_storeu_ps(S+j,s);
        _mm512_storeu_ps(oh+j,_mm512_fmadd_ps(s,vq,_mm512_loadu_ps(oh+j)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 vk=_mm256_set1_ps(ki), vq=_mm256_set1_ps(qi);
    for(;j+8<=dv;j+=8){
        __m256 s=_mm256_fmadd_ps(vk,_mm256_loadu_ps(delta+j),_mm256_loadu_ps(S+j));
        _mm256_storeu_ps(S+j,s);
        _mm256_storeu_ps(oh+j,_mm256_fmadd_ps(s,vq,_mm256_loadu_ps(oh+j)));
    }
#elif defined(__ARM_NEON)
    float32x4_t vk=vdupq_n_f32(ki), vq=vdupq_n_f32(qi);
    for(;j+4<=dv;j+=4){
        float32x4_t s=vfmaq_f32(vld1q_f32(S+j),vk,vld1q_f32(delta+j));
        vst1q_f32(S+j,s);
        vst1q_f32(oh+j,vfmaq_f32(vld1q_f32(oh+j),s,vq));
    }
#endif
    for(;j<dv;j++){ S[j]+=ki*delta[j]; oh[j]+=S[j]*qi; }
}

#endif /* SIMD_H */
