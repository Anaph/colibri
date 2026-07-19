/* Logica dei test degli header (C puro; il glue gtest sta in headers_gtest.cc).
 * Ogni funzione ht_* ritorna 0=ok, 1=fail, 2=skip; i dettagli vanno su stderr. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "../json.h"
#include "../st.h"
#include "../gguf.h"
#include "tiny_gguf.h"
#include "../tok.h"
#include "../tier.h"
#include "../grammar.h"
#include "../schema_gbnf.h"
#include "../decode_batch.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

/* ---------------- json.h ---------------- */
int ht_json(void) {
    jval *root = json_parse(
        "{\"name\":\"Colibri\\nCPU\",\"enabled\":true,\"empty\":null,"
        "\"values\":[1,-2.5,3e2],\"unicode\":\"\\u03bb \\uD83D\\uDE80\"}",
        NULL
    );
    CHECK(root && root->t == J_OBJ);
    CHECK(strcmp(json_get(root, "name")->str, "Colibri\nCPU") == 0);
    CHECK(json_get(root, "enabled")->boolean == 1);
    CHECK(json_get(root, "empty")->t == J_NULL);
    CHECK(json_get(root, "missing") == NULL);
    jval *values = json_get(root, "values");
    CHECK(values->t == J_ARR && values->len == 3);
    CHECK(values->kids[0]->num == 1.0);
    CHECK(values->kids[1]->num == -2.5);
    CHECK(values->kids[2]->num == 300.0);
    CHECK(strcmp(json_get(root, "unicode")->str, "λ 🚀") == 0);
    return 0;
}

/* json_free: albero annidato liberato senza crash, riparsabile dopo (sotto
 * ASAN in CI questo e' anche l'assert di assenza di leak/double-free) */
int ht_json_free(void) {
    const char *doc =
        "{\"a\":{\"b\":[1,\"x\",{\"c\":null,\"d\":[true,false]}],\"e\":\"\"},"
        "\"f\":[],\"g\":{},\"h\":\"fine\"}";
    for (int rep = 0; rep < 3; rep++) {
        jval *root = json_parse(doc, NULL);
        CHECK(root && root->t == J_OBJ && root->len == 4);
        CHECK(json_get(json_get(root,"a"),"b")->len == 3);
        CHECK(strcmp(json_get(root,"h")->str, "fine") == 0);
        json_free(root);
    }
    json_free(NULL);                        /* deve essere un no-op */
    return 0;
}

/* ---------------- st.h ---------------- */
int ht_st(void) {
    CHECK(bf16_to_f32(0x3f80) == 1.0f);
    CHECK(bf16_to_f32(0xc020) == -2.5f);
    CHECK(f16_to_f32(0x3c00) == 1.0f);
    CHECK(f16_to_f32(0xc100) == -2.5f);
    CHECK(f16_to_f32(0x0001) > 0.0f);
    CHECK(isinf(f16_to_f32(0x7c00)));
    CHECK(st_hash("tensor.weight") == st_hash("tensor.weight"));
    CHECK(st_hash("tensor.weight") != st_hash("tensor.bias"));
    return 0;
}

/* ---------------- gguf.h ---------------- */
static const char *ht_gguf_tmp(const char *name, char *path, int cap) {
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    snprintf(path, cap, "%s/ht_%s.gguf", tmp, name);
    return path;
}

/* header v3: KV tipati (str/u32/f32/array), traduzione nomi, lettura f32 */
int ht_gguf_header(void) {
    char path[512]; ht_gguf_tmp("hdr", path, sizeof(path));
    tg_reset();
    tg_kv_str("general.architecture", "qwen3");
    tg_kv_u32("qwen3.embedding_length", 16);
    tg_kv_f32("qwen3.rope.freq_base", 1000000.0f);
    static const char *toks[3] = { "a", "bb", "ccc" };
    tg_kv_arr_str("tokenizer.ggml.tokens", toks, 3);
    static const int32_t tt[3] = { 1, 1, 3 };
    tg_kv_arr_i32("tokenizer.ggml.token_type", tt, 3);
    float emb[8*16], qw[16*16];
    for (int i = 0; i < 8*16; i++) emb[i] = (float)i * 0.25f;
    for (int i = 0; i < 16*16; i++) qw[i] = (float)(i % 7) - 3.f;
    tg_tensor_f32("token_embd.weight", 8, 16, emb);
    tg_tensor_f32("blk.0.attn_q.weight", 16, 16, qw);
    tg_write(path);
    shards S; GgufMeta M;
    gguf_index(&S, &M, path);
    CHECK(!strcmp(M.arch, "qwen3"));
    CHECK(gguf_int(&M, "qwen3.embedding_length", -1) == 16);
    CHECK(fabs(gguf_float(&M, "qwen3.rope.freq_base", 0) - 1000000.0) < 1);
    CHECK(gguf_int(&M, "assente", -7) == -7);
    /* traduzione nomi llama.cpp -> HF */
    CHECK(st_numel(&S, "model.embed_tokens.weight") == 8*16);
    CHECK(st_numel(&S, "model.layers.0.self_attn.q_proj.weight") == 16*16);
    CHECK(!st_has(&S, "token_embd.weight"));
    /* dati identici byte a byte */
    float r[16*16];
    st_read_f32(&S, "model.embed_tokens.weight", r, 0);
    CHECK(memcmp(r, emb, sizeof(emb)) == 0);
    st_read_slice_f32(&S, "model.layers.0.self_attn.q_proj.weight", 2*16, 16, r, 0);
    CHECK(memcmp(r, qw + 2*16, 16*sizeof(float)) == 0);
    /* iterazione array: stringhe e interi */
    garr a; int64_t l;
    CHECK(gguf_arr(&M, "tokenizer.ggml.tokens", &a));
    const char *s0 = garr_next_str(&a, &l); CHECK(l == 1 && s0[0] == 'a');
    garr_next_str(&a, &l); CHECK(l == 2);
    garr_next_str(&a, &l); CHECK(l == 3);
    CHECK(garr_next_str(&a, &l) == NULL);
    CHECK(gguf_arr(&M, "tokenizer.ggml.token_type", &a));
    CHECK(garr_next_int(&a) == 1 && garr_next_int(&a) == 1 && garr_next_int(&a) == 3);
    remove(path);
    return 0;
}

/* Q8_0/Q4_0: dequant esatto su blocchi costruiti a mano (scale f16 note) */
int ht_gguf_q40_q80(void) {
    /* Q8_0: d=2.0 (0x4000), q=i-16 -> out=2*(i-16) */
    uint8_t b8[34]; uint16_t d = 0x4000; memcpy(b8, &d, 2);
    for (int i = 0; i < 32; i++) ((int8_t*)(b8+2))[i] = (int8_t)(i - 16);
    float out[32];
    gguf_dq_q8_0(b8, 1, out);
    for (int i = 0; i < 32; i++) CHECK(out[i] == 2.f*(float)(i-16));
    /* Q4_0: d=0.5 (0x3800); byte i = elem i (basso) ed elem i+16 (alto) */
    uint8_t b4[18]; d = 0x3800; memcpy(b4, &d, 2);
    for (int i = 0; i < 16; i++) b4[2+i] = (uint8_t)((i & 0xF) | (((15-i) & 0xF) << 4));
    gguf_dq_q4_0(b4, 1, out);
    for (int i = 0; i < 16; i++) {
        CHECK(out[i]      == 0.5f*(float)(i - 8));
        CHECK(out[i + 16] == 0.5f*(float)((15-i) - 8));
    }
    /* repack -> int4 grouped: dequant del NOSTRO layout bit-identico */
    uint8_t q4[16]; float qs[1];
    gguf_repack_q4_0(b4, q4, qs, 1, 32);
    CHECK(qs[0] == 0.5f);
    for (int i = 0; i < 32; i++) {
        uint8_t byte = q4[i >> 1];
        int v = (i & 1) ? (int)(byte >> 4) - 8 : (int)(byte & 0xF) - 8;
        CHECK(qs[0]*(float)v == out[i]);
    }
    return 0;
}

/* K-quants: superblocchi costruiti a mano vs riferimento per-elemento */
int ht_gguf_kquants(void) {
    float out[256];
    /* Q4_K: d=1.0, dmin=0.5, scales[12]=j+1, qs[i]=pattern */
    uint8_t b[210]; memset(b, 0, sizeof(b));
    uint16_t one = 0x3C00, half = 0x3800;
    memcpy(b, &one, 2); memcpy(b+2, &half, 2);
    for (int j = 0; j < 12; j++) b[4+j] = (uint8_t)(j + 1);
    for (int i = 0; i < 128; i++) b[16+i] = (uint8_t)((i*7) & 0xFF);
    gguf_dq_q4k(b, 1, out);
    { const uint8_t *sc = b+4, *q = b+16; int is = 0; float *y = out;
      for (int j = 0; j < 256; j += 64) {
          uint8_t s1, m1, s2, m2;
          gg_k4_scale(is+0, sc, &s1, &m1); gg_k4_scale(is+1, sc, &s2, &m2);
          for (int l = 0; l < 32; l++) CHECK(*y++ == 1.f*s1*(float)(q[l] & 0xF) - 0.5f*m1);
          for (int l = 0; l < 32; l++) CHECK(*y++ == 1.f*s2*(float)(q[l] >>  4) - 0.5f*m2);
          q += 32; is += 2;
      } }
    /* Q5_K: come Q4_K ma con il bit alto da qh */
    memset(b, 0, sizeof(b));
    memcpy(b, &one, 2); memcpy(b+2, &half, 2);
    for (int j = 0; j < 12; j++) b[4+j] = (uint8_t)(13 - j);
    for (int i = 0; i < 32; i++) b[16+i] = (uint8_t)(i*11);
    for (int i = 0; i < 128; i++) b[48+i] = (uint8_t)(255 - i);
    gguf_dq_q5k(b, 1, out);
    { const uint8_t *sc = b+4, *qh = b+16, *ql = b+48; int is = 0; float *y = out;
      uint8_t u1 = 1, u2 = 2;
      for (int j = 0; j < 256; j += 64) {
          uint8_t s1, m1, s2, m2;
          gg_k4_scale(is+0, sc, &s1, &m1); gg_k4_scale(is+1, sc, &s2, &m2);
          for (int l = 0; l < 32; l++) CHECK(*y++ == 1.f*s1*(float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - 0.5f*m1);
          for (int l = 0; l < 32; l++) CHECK(*y++ == 1.f*s2*(float)((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - 0.5f*m2);
          ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
      } }
    /* Q6_K: ql/qh/scales int8/d */
    memset(b, 0, sizeof(b));
    for (int i = 0; i < 128; i++) b[i] = (uint8_t)(i*3);
    for (int i = 0; i < 64; i++) b[128+i] = (uint8_t)(i*5);
    for (int i = 0; i < 16; i++) ((int8_t*)(b+192))[i] = (int8_t)(i - 8);
    memcpy(b+208, &one, 2);
    gguf_dq_q6k(b, 1, out);
    { const uint8_t *ql = b, *qh = b+128; const int8_t *sc = (const int8_t*)(b+192);
      float *y = out;
      for (int n = 0; n < 256; n += 128) {
          for (int l = 0; l < 32; l++) {
              int is = l/16;
              int q1 = (int)((ql[l]    & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
              int q2 = (int)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
              int q3 = (int)((ql[l]    >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
              int q4v = (int)((ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
              CHECK(y[l]      == 1.f*sc[is]   * (float)q1);
              CHECK(y[l + 32] == 1.f*sc[is+2] * (float)q2);
              CHECK(y[l + 64] == 1.f*sc[is+4] * (float)q3);
              CHECK(y[l + 96] == 1.f*sc[is+6] * (float)q4v);
          }
          y += 128; ql += 64; qh += 32; sc += 8;
      } }
    g_st_dequant_fn = NULL;          /* non inquinare gli altri test st */
    return 0;
}

/* ---------------- tok.h (fixture, entrambi i formati merges) ---------------- */
static const char *TOK_FIX_PAIRS =
  "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"h\":0,\"e\":1,\"l\":2,\"o\":3,\"\\u0120\":4,"
  "\"he\":5,\"ll\":6,\"hell\":7,\"hello\":8},"
  "\"merges\":[[\"h\",\"e\"],[\"l\",\"l\"],[\"he\",\"ll\"],[\"hell\",\"o\"]]},"
  "\"added_tokens\":[{\"id\":9,\"content\":\"<|endoftext|>\"}]}";
static const char *TOK_FIX_STRINGS =
  "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"h\":0,\"e\":1,\"l\":2,\"o\":3,\"\\u0120\":4,"
  "\"he\":5,\"ll\":6,\"hell\":7,\"hello\":8},"
  "\"merges\":[\"h e\",\"l l\",\"he ll\",\"hell o\"]},"
  "\"added_tokens\":[{\"id\":9,\"content\":\"<|endoftext|>\"}]}";

static int tok_write_tmp(const char *name, const char *body, char *path, int cap) {
    const char *tmp = getenv("TMPDIR"); if (!tmp) tmp = "/tmp";
    snprintf(path, cap, "%s/tok_fix_%s.json", tmp, name);
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); return 1; }
    fwrite(body, 1, strlen(body), f); fclose(f);
    return 0;
}

static int tok_run_fixture(const char *name, const char *body) {
    char path[512];
    CHECK(tok_write_tmp(name, body, path, sizeof(path)) == 0);
    Tok T; tok_load(&T, path);
    int ids[64]; int n = tok_encode(&T, "hello", 5, ids, 64);
    CHECK(n == 1 && ids[0] == 8);
    n = tok_encode(&T, "hello<|endoftext|>hello", 23, ids, 64);
    CHECK(n == 3 && ids[0] == 8 && ids[1] == 9 && ids[2] == 8);
    n = tok_encode(&T, "hello", 5, ids, 64);
    char dec[64]; int dn = tok_decode(&T, ids, n, dec, 63);
    CHECK(dn == 5 && !memcmp(dec, "hello", 5));
    remove(path);
    return 0;
}

int ht_tok_pairs(void)   { return tok_run_fixture("pairs",   TOK_FIX_PAIRS); }
int ht_tok_strings(void) { return tok_run_fixture("strings", TOK_FIX_STRINGS); }

/* pool di stringhe: dopo tok_load il JSON e' liberato e tutte le stringhe
 * vivono nel pool; load->uso->tok_free->reload ripetuto non degrada nulla */
int ht_tok_arena(void) {
    char path[512];
    CHECK(tok_write_tmp("arena", TOK_FIX_STRINGS, path, sizeof(path)) == 0);
    for (int rep = 0; rep < 3; rep++) {
        Tok T; tok_load(&T, path);
        CHECK(T.pool != NULL && T.pool_off == T.pool_len);   /* sizing esatto */
        /* le stringhe vive puntano DENTRO il pool, non nel JSON liberato */
        CHECK(T.id2str[8] >= T.pool && T.id2str[8] < T.pool + T.pool_len);
        CHECK(T.sp[0].str >= T.pool && T.sp[0].str < T.pool + T.pool_len);
        int ids[64]; int n = tok_encode(&T, "hello<|endoftext|>hello", 23, ids, 64);
        CHECK(n == 3 && ids[0] == 8 && ids[1] == 9 && ids[2] == 8);
        char dec[64]; int dn = tok_decode(&T, ids, 1, dec, 63);
        CHECK(dn == 5 && !memcmp(dec, "hello", 5));
        CHECK(tok_id_of(&T, "<|endoftext|>") == 9);
        tok_free(&T);
        CHECK(T.pool == NULL && T.vocab.e == NULL);
    }
    remove(path);
    return 0;
}

/* ---- modalita' sentencepiece (Gemma): metaspazio + byte-fallback ---- */
static const char *TOK_FIX_SP =
  "{\"model\":{\"type\":\"BPE\",\"byte_fallback\":true,"
  "\"vocab\":{\"h\":0,\"e\":1,\"l\":2,\"o\":3,\"\\u2581\":4,"
  "\"he\":5,\"ll\":6,\"hell\":7,\"hello\":8,\"\\u2581hello\":9,"
  "\"<0x41>\":10,\"<0x01>\":11},"
  "\"merges\":[\"h e\",\"l l\",\"he ll\",\"hell o\",\"\\u2581 hello\"]},"
  "\"added_tokens\":[{\"id\":12,\"content\":\"<eos>\"}]}";

int ht_tok_sp(void) {
    char path[512];
    CHECK(tok_write_tmp("sp", TOK_FIX_SP, path, sizeof(path)) == 0);
    Tok T; tok_load(&T, path);
    CHECK(T.mode == 1);                          /* byte_fallback -> sp */
    CHECK(T.add_dummy_prefix == 0);              /* niente Prepend nel fixture */
    int ids[64];
    int n = tok_encode(&T, "hello", 5, ids, 64);
    CHECK(n == 1 && ids[0] == 8);
    n = tok_encode(&T, "hello hello", 11, ids, 64);          /* spazio -> "▁hello" */
    CHECK(n == 2 && ids[0] == 8 && ids[1] == 9);
    n = tok_encode(&T, "hello\x01", 6, ids, 64);             /* byte-fallback <0x01> */
    CHECK(n == 2 && ids[0] == 8 && ids[1] == 11);
    n = tok_encode(&T, "hello<eos>hello", 15, ids, 64);      /* added atomico */
    CHECK(n == 3 && ids[0] == 8 && ids[1] == 12 && ids[2] == 8);
    /* round-trip: "hello hello" -> ids -> "hello hello" (▁->spazio) */
    n = tok_encode(&T, "hello hello", 11, ids, 64);
    char dec[64]; int dn = tok_decode(&T, ids, n, dec, 63);
    CHECK(dn == 11 && !memcmp(dec, "hello hello", 11));
    /* byte-fallback round-trip */
    n = tok_encode(&T, "hello\x01", 6, ids, 64);
    dn = tok_decode(&T, ids, n, dec, 63);
    CHECK(dn == 6 && !memcmp(dec, "hello\x01", 6));
    remove(path);
    return 0;
}

int ht_tok_sp_detect(void) {
    /* stesso vocab senza byte_fallback -> resta byte-level */
    char path[512];
    CHECK(tok_write_tmp("nosp", TOK_FIX_STRINGS, path, sizeof(path)) == 0);
    Tok T; tok_load(&T, path);
    CHECK(T.mode == 0);
    remove(path);
    return 0;
}

/* ---------------- tier.h ---------------- */
int ht_tier(void) {
    uint32_t heat[6] = {20,2,8,3,30,1};
    int pinned[2] = {0,1}, slot = -1, eid = -1; long gain = 0;
    CHECK(tier_pick_swap(heat,6,pinned,2,&slot,&eid,&gain));
    CHECK(slot==1 && eid==4 && gain==28);
    uint32_t stable[4] = {20,18,24,4}; int resident[2] = {0,1};
    CHECK(!tier_pick_swap(stable,4,resident,2,&slot,&eid,&gain));
    tier_decay(heat,6);
    CHECK(heat[0]==10 && heat[1]==1 && heat[4]==15);
    uint32_t freq[5] = {10,10,2,18,18}, last[5] = {10,90,95,20,99};
    int live[2] = {0,1};
    CHECK(tier_pick_lfru(freq,last,100,5,live,2,&slot,&eid,&gain));
    CHECK(slot==0 && eid==4);
    return 0;
}

/* ---------------- grammar.h ---------------- */
static int gr_feed(GrState *S, const char *s) {
    int n = 0;
    while (s[n]) { if (gr_accept(S,(unsigned char)s[n]) != 1) break; n++; }
    return n;
}

int ht_grammar(void) {
    static Grammar G;
    GrState S;
    char buf[512];
    unsigned char m[32]; int end;

    CHECK(gr_parse(&G,"root ::= \"{\\\"id\\\":\"")==0);
    gr_state_init(&S,&G);
    CHECK(S.alive);
    CHECK(gr_forced(&S,buf,sizeof buf)==6);
    CHECK(!memcmp(buf,"{\"id\":",6));
    CHECK(gr_feed(&S,"{\"id\":")==6);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    CHECK(gr_parse(&G,"root ::= \"a\" (\"b\" | \"c\") \"d\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='a');
    CHECK(gr_feed(&S,"ab")==2);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='d');
    gr_free(&G);

    CHECK(gr_parse(&G,
        "root ::= \"\\\"\" val \"\\\"\"\n"
        "val  ::= \"no_fit\" | \"partial_fit\" | \"good_fit\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='\"');
    CHECK(gr_feed(&S,"\"n")==2);
    CHECK(gr_forced(&S,buf,sizeof buf)==6 && !memcmp(buf,"o_fit\"",6));
    gr_free(&G);

    CHECK(gr_parse(&G,"root ::= \"a\" [0-9]* \"b\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_feed(&S,"a")==1);
    CHECK(gr_admissible(&S,m,&end)==11 && end==0);
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    CHECK(gr_feed(&S,"42b")==3);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    CHECK(gr_parse(&G,"root ::= (\"x\" \"\\n\")+")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==2 && buf[0]=='x' && buf[1]=='\n');
    CHECK(gr_feed(&S,"x\n")==2);
    CHECK(gr_admissible(&S,m,&end)==1 && end==1);
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    gr_free(&G);

    CHECK(gr_parse(&G,"root ::= \"ab\"+ \"c\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==2 && !memcmp(buf,"ab",2));
    CHECK(gr_feed(&S,"ab")==2);
    CHECK(gr_admissible(&S,m,&end)==2 && end==0);
    CHECK(gr_feed(&S,"abc")==3);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    CHECK(gr_parse(&G,"root ::= \"\\\"\" [^\"]* \"\\\"\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_feed(&S,"\"")==1);
    CHECK(gr_admissible(&S,m,&end)==256 && end==0);
    CHECK(gr_feed(&S,"ciao \\ mondo\"")==13);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    CHECK(gr_parse(&G,"root ::= \"ab\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_accept(&S,'x')==0);
    CHECK(gr_accept(&S,'a')==1 && gr_accept(&S,'b')==1);
    gr_free(&G);

    CHECK(gr_parse(&G,
        "# grammatica di prova\n"
        "root ::= \"\\x41\"   # una A\n"
        "         [\\x30-\\x32]\n")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='A');
    CHECK(gr_feed(&S,"A1")==2);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    CHECK(gr_parse(&G,"root ::= foo")!=0);
    CHECK(gr_parse(&G,"a ::= \"x\"")!=0);
    CHECK(gr_parse(&G,"root ::= \"x\" )")!=0);

    CHECK(gr_parse(&G,"root ::= root \"a\" | \"b\"")==0);
    gr_state_init(&S,&G);
    CHECK(!S.alive);
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    gr_free(&G);

    CHECK(gr_parse(&G,
        "root ::= riga+\n"
        "riga ::= \"{\\\"id\\\":\\\"\" chiave \"\\\",\\\"fit_category\\\":\\\"\" cat \"\\\"}\" \"\\n\"\n"
        "chiave ::= [a-z0-9-]+\n"
        "cat  ::= \"no_fit\" | \"partial_fit\" | \"good_fit\"\n")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==7 && !memcmp(buf,"{\"id\":\"",7));
    CHECK(gr_feed(&S,"{\"id\":\"ocds-123\"")==16);
    int nf = gr_forced(&S,buf,sizeof buf);
    buf[nf]=0;
    CHECK(nf==17 && !strcmp(buf,",\"fit_category\":\""));
    CHECK(gr_feed(&S,",\"fit_category\":\"p")==18);
    nf = gr_forced(&S,buf,sizeof buf); buf[nf]=0;
    CHECK(nf==13 && !strcmp(buf,"artial_fit\"}\n"));
    CHECK(gr_feed(&S,"artial_fit\"}\n")==13);
    CHECK(gr_admissible(&S,m,&end)==1 && end==1);
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    CHECK(gr_feed(&S,"{")==1);
    CHECK(gr_forced(&S,buf,sizeof buf)==6 && !memcmp(buf,"\"id\":\"",6));
    gr_free(&G);
    return 0;
}

/* ---------------- schema_gbnf.h ---------------- */
static int sg_compile(const char *schema, Grammar *G) {
    char err[160] = {0};
    char *g = schema_to_gbnf(schema, err, sizeof err);
    if (!g) return -1;
    int rc = gr_parse(G, g);
    if (rc) fprintf(stderr, "  gr_parse error: %s\nGBNF:\n%s\n", G->err, g);
    free(g);
    return rc;
}

static int sg_walk(GrState *S, const char *bytes) {
    int n = 0;
    for (const char *p = bytes; *p; p++, n++)
        if (gr_accept(S, (unsigned char)*p) != 1) break;
    return n;
}

int ht_schema_gbnf(void) {
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{"
            "\"score\":{\"type\":\"integer\"},\"verdict\":{\"type\":\"string\"}},"
            "\"required\":[\"score\",\"verdict\"]}";
        CHECK(sg_compile(sc, &G) == 0);
        gr_state_init(&S, &G);
        char f[256]; int n = gr_forced(&S, f, sizeof f);
        CHECK(n == 0);
        CHECK(sg_walk(&S, "{\"") == 2);
        n = gr_forced(&S, f, sizeof f);
        CHECK(n > 0 && strncmp(f, "score\"", 6) == 0);
        const char *rest = "score\":-42,\"verdict\":\"no_fit\"}";
        CHECK(sg_walk(&S, rest) == (int)strlen(rest));
        unsigned char mask[32]; int can_end = 0;
        gr_admissible(&S, mask, &can_end);
        CHECK(can_end == 1);
        gr_state_init(&S, &G);
        const char *sloppy = "{ \"score\" : -42 ,\n  \"verdict\" : \"no_fit\" }";
        CHECK(sg_walk(&S, sloppy) == (int)strlen(sloppy));
        gr_admissible(&S, mask, &can_end);
        CHECK(can_end == 1);
        gr_free(&G);
    }
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{"
            "\"fit\":{\"type\":\"string\",\"enum\":[\"no_fit\",\"partial_fit\",\"strong_fit\"]}},"
            "\"required\":[\"fit\"]}";
        CHECK(sg_compile(sc, &G) == 0);
        gr_state_init(&S, &G);
        char f[256]; int n;
        CHECK(sg_walk(&S, "{\"fit\":\"p") == 9);
        n = gr_forced(&S, f, sizeof f);
        CHECK(n > 0 && strncmp(f, "artial_fit\"", 11) == 0);
        gr_free(&G);
    }
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{"
            "\"meta\":{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"]},"
            "\"rows\":{\"type\":\"array\",\"minItems\":1,\"items\":{\"type\":\"object\","
              "\"properties\":{\"v\":{\"type\":\"number\"},\"note\":{\"type\":\"null\"}},"
              "\"required\":[\"v\",\"note\"]}}},"
            "\"required\":[\"meta\",\"rows\"]}";
        CHECK(sg_compile(sc, &G) == 0);
        gr_state_init(&S, &G);
        const char *inst = "{\"meta\":{\"ok\":true},\"rows\":[{\"v\":3.5,\"note\":null},{\"v\":-1e-3,\"note\":null}]}";
        CHECK(sg_walk(&S, inst) == (int)strlen(inst));
        unsigned char mask[32]; int can_end = 0;
        gr_admissible(&S, mask, &can_end);
        CHECK(can_end == 1);
        gr_free(&G);
    }
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{"
            "\"k\\\"x\":{\"const\":\"a\\\\b\"}},\"required\":[\"k\\\"x\"]}";
        CHECK(sg_compile(sc, &G) == 0);
        gr_state_init(&S, &G);
        const char *inst = "{\"k\\\"x\":\"a\\\\b\"}";
        CHECK(sg_walk(&S, inst) == (int)strlen(inst));
        gr_free(&G);
    }
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{\"t\":{\"type\":\"string\"}},\"required\":[\"t\"]}";
        CHECK(sg_compile(sc, &G) == 0);
        gr_state_init(&S, &G);
        const char *inst = "{\"t\":\"hello \\\"w\\\" \\u00e9\\n x\"}";
        CHECK(sg_walk(&S, inst) == (int)strlen(inst));
        gr_free(&G);
    }
    {
        char err[160];
        CHECK(schema_to_gbnf("{\"oneOf\":[{\"type\":\"string\"}]}", err, sizeof err) == NULL);
        CHECK(schema_to_gbnf("{\"type\":\"string\",\"pattern\":\"a+\"}", err, sizeof err) == NULL);
        CHECK(schema_to_gbnf("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"},"
                             "\"b\":{\"type\":\"string\"}},\"required\":[\"a\"]}", err, sizeof err) == NULL);
    }
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}},\"required\":[\"n\"]}";
        CHECK(sg_compile(sc, &G) == 0);
        gr_state_init(&S, &G);
        CHECK(sg_walk(&S, "{\"n\":0}") == 7);
        gr_state_init(&S, &G);
        CHECK(sg_walk(&S, "{\"n\":01}") < 8);
        gr_free(&G);
    }
    {
        Grammar G; GrState S;
        const char *sc = "{\"type\":\"object\",\"properties\":{\"b\":{\"enum\":[1,2,3]}},\"required\":[\"b\"]}";
        CHECK(sg_compile(sc, &G) == 0);
        gr_state_init(&S, &G);
        CHECK(sg_walk(&S, "{\"b\":2}") == 7);
        gr_free(&G);
    }
    return 0;
}

/* ---------------- decode_batch.h ---------------- */
int ht_decode_batch(void) {
    {
        float sequence_a[4 * 3] = {0};
        float sequence_b[4 * 3] = {0};
        float *a2 = coli_kv_row(sequence_a, 2, 3);
        float *b1 = coli_kv_row(sequence_b, 1, 3);
        a2[0] = 20.0f;
        b1[2] = 12.0f;
        CHECK(a2 == &sequence_a[6]);
        CHECK(b1 == &sequence_b[3]);
        CHECK(sequence_a[6] == 20.0f);
        CHECK(sequence_b[5] == 12.0f);
        CHECK(sequence_a[5] == 0.0f);
        CHECK(sequence_b[6] == 0.0f);
    }
    {
        float storage[5 * 7] = {0};
        const float *row = coli_kv_row(storage, 4, 7);
        CHECK(row == &storage[28]);
    }
    {
        ColiSubmit sub;
        CHECK(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95", &sub));
        CHECK(sub.id == 42 && sub.slot == 3 && sub.bytes == 17);
        CHECK(sub.max_tokens == 64 && sub.temperature > .69f && sub.top_p > .94f);
        CHECK(!coli_submit_parse("SUBMIT 1 -1 2 3 0.7 1", &sub));
        CHECK(!coli_submit_parse("SUBMIT 1 0 2 0 0.7 1", &sub));
        CHECK(!coli_submit_parse("SUBMIT 1 0 2 3 4 1", &sub));
        CHECK(!coli_submit_parse("SUBMIT 0 0 2 3 1 1", &sub));
        CHECK(!coli_submit_parse("SUBMIT 1 0 2 3 nan 1", &sub));
        CHECK(!coli_submit_parse("SUBMIT 1 0 2 3 1 inf", &sub));
        CHECK(coli_submit_parse("SUBMIT 1 0 16777216 3 1 1", &sub));
        CHECK(!coli_submit_parse("SUBMIT 1 0 16777217 3 1 1", &sub));
        CHECK(!coli_submit_parse("SUBMIT 1 0 2 3 1 1 trailing", &sub));
    }
    return 0;
}

/* ---------------- compat.h: O_DIRECT windows (skip su POSIX) ---------------- */
#ifndef _WIN32
int ht_compat_direct(void) { return 2; }
#else
#include <fcntl.h>
#include <io.h>
#include "../compat.h"

#define CD_FSZ (1u<<20)
#define CD_TMPF "test_direct.tmp"
static int cd_fail(const char *m){ fprintf(stderr,"compat direct test failed: %s\n",m); return 1; }

int ht_compat_direct(void) {
    FILE *w=fopen(CD_TMPF,"wb"); if(!w) return cd_fail("create temp");
    uint8_t *pat=malloc(CD_FSZ);
    for(uint32_t i=0;i<CD_FSZ;i++) pat[i]=(uint8_t)(i*2246822519u>>24);
    if(fwrite(pat,1,CD_FSZ,w)!=CD_FSZ){ fclose(w); return cd_fail("short write"); }
    fclose(w);
    int dfd = compat_open_direct(CD_TMPF);
    if(dfd<0) return cd_fail("compat_open_direct returned -1");
    void *buf=NULL;
    if(posix_memalign(&buf,4096,64*1024)!=0) return cd_fail("alloc aligned");
    if(pread(dfd, buf, 64*1024, 4096)!=64*1024) return cd_fail("aligned pread size");
    if(memcmp(buf, pat+4096, 64*1024)!=0) return cd_fail("aligned pread data mismatch");
    ssize_t r = pread(dfd, buf, 64*1024, 1000);
    if(r>0 && memcmp(buf, pat+1000, (size_t)r)!=0) return cd_fail("misaligned read returned wrong data");
    if(compat_fsize(dfd)!=(off_t)CD_FSZ) return cd_fail("compat_fsize on direct fd");
    int bfd = open(CD_TMPF, COMPAT_O_RDONLY);
    if(compat_fsize(bfd)!=(off_t)CD_FSZ) return cd_fail("compat_fsize on buffered fd");
    close(bfd);
    if(compat_open_direct("no_such_file.tmp")>=0) return cd_fail("open missing file must fail");
    if(compat_fsize(-1)>=0) return cd_fail("compat_fsize on bad fd must be negative");
    int wfd = open(CD_TMPF, COMPAT_O_RDONLY);
    if(wfd<0) return cd_fail("open buffered for fadvise");
    if(posix_fadvise(wfd, 0, (off_t)CD_FSZ, POSIX_FADV_WILLNEED)!=0) return cd_fail("WILLNEED returned nonzero");
    if(posix_fadvise(wfd, 0, (off_t)CD_FSZ, POSIX_FADV_DONTNEED)!=0) return cd_fail("DONTNEED should be a safe no-op (return 0)");
    if(posix_fadvise(-1, 0, (off_t)CD_FSZ, POSIX_FADV_WILLNEED)!=0) return cd_fail("WILLNEED on bad fd should no-op (return 0)");
    if(posix_fadvise(wfd, 0, 0, POSIX_FADV_WILLNEED)!=0) return cd_fail("WILLNEED with len<=0 should no-op");
    uint8_t *verify=malloc(CD_FSZ);
    if(pread(wfd, verify, CD_FSZ, 0)!=(ssize_t)CD_FSZ) return cd_fail("fadvise: pread size");
    if(memcmp(verify, pat, CD_FSZ)!=0) return cd_fail("fadvise: data corrupted by cache-warmer");
    free(verify);
    close(wfd);
    close(dfd);
    compat_aligned_free(buf); free(pat); remove(CD_TMPF);
    return 0;
}
#endif /* _WIN32 */

/* ---------------- accumulatore AVX-512 int4 (skip senza AVX-512) ---------------- */
#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>

static inline float i4_dot_avx512(const uint8_t *w,const float *x,int I){
    const __m128i m4=_mm_set1_epi8(0x0F); const __m512i b8=_mm512_set1_epi32(8);
    __m512 acc0=_mm512_setzero_ps(),acc1=_mm512_setzero_ps(); int i=0;
    for(;i+32<=I;i+=32){ __m128i by=_mm_loadu_si128((const __m128i*)(w+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi),n1=_mm_unpackhi_epi8(lo,hi);
        __m512 w0=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n0),b8));
        __m512 w1=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n1),b8));
        acc0=_mm512_fmadd_ps(_mm512_loadu_ps(x+i),w0,acc0);
        acc1=_mm512_fmadd_ps(_mm512_loadu_ps(x+i+16),w1,acc1);
    }
    float a=_mm512_reduce_add_ps(_mm512_add_ps(acc0,acc1));
    for(;i<I;i++){ uint8_t b=w[i>>1]; a+=x[i]*(float)(((b>>((i&1)*4))&15)-8); }
    return a;
}
static float i4_dot_scalar(const uint8_t *w,const float *x,int I){
    float a=0;
    for(int i=0;i<I;i++){ uint8_t b=w[i>>1]; a+=x[i]*(float)(((b>>((i&1)*4))&15)-8); }
    return a;
}
static double i4_dot_double(const uint8_t *w,const float *x,int I){
    double a=0;
    for(int i=0;i<I;i++){ uint8_t b=w[i>>1]; a+=(double)x[i]*(double)(((b>>((i&1)*4))&15)-8); }
    return a;
}
static uint64_t i4_rng=0x243F6A8885A308D3ULL;
static double i4_rndu(void){ i4_rng^=i4_rng<<13; i4_rng^=i4_rng>>7; i4_rng^=i4_rng<<17;
    return (double)(i4_rng>>11)*(1.0/9007199254740992.0); }
static double i4_rndn(void){ double u1=i4_rndu()+1e-18,u2=i4_rndu();
    return sqrt(-2.0*log(u1))*cos(6.283185307179586*u2); }

static int i4_trial(int I, int rows, double xscale, const char *label){
    uint8_t *w=malloc((size_t)(I+1)/2);
    float *x=malloc((size_t)I*sizeof(float));
    double e512=0,esca=0; int bad=0;
    for(int r=0;r<rows;r++){
        for(int i=0;i<I;i+=2){ int q0=(int)(i4_rndu()*16),q1=(int)(i4_rndu()*16);
            w[i>>1]=(uint8_t)(q0|(q1<<4)); }
        for(int i=0;i<I;i++) x[i]=(float)(i4_rndn()*xscale);
        double ref=i4_dot_double(w,x,I);
        float v512=i4_dot_avx512(w,x,I), vsca=i4_dot_scalar(w,x,I);
        if(!isfinite(v512)) bad++;
        double den=fabs(ref); if(den<1.0) den=1.0;
        double e5=fabs((double)v512-ref)/den, es=fabs((double)vsca-ref)/den;
        if(e5>e512) e512=e5;
        if(es>esca) esca=es;
    }
    free(w); free(x);
    int ok = !bad && e512 <= esca*2.0 + 1e-7;
    fprintf(stderr,"  %-24s I=%-5d avx512 max %.3e | scalar max %.3e | %s\n",
        label,I,e512,esca,ok?"ok":"FAIL");
    return ok;
}

int ht_i4_acc512(void){
    int ok=1;
    ok &= i4_trial(6144, 2000, 1.0,  "gate/up rows");
    ok &= i4_trial(6144, 2000, 30.0, "gate/up large x");
    ok &= i4_trial(2048, 2000, 1.0,  "down rows");
    ok &= i4_trial(2048, 2000, 0.02, "down small x");
    ok &= i4_trial(6143, 1000, 1.0,  "tail I=6143");
    ok &= i4_trial(96,   5000, 1.0,  "short rows");
    return ok ? 0 : 1;
}
#else
int ht_i4_acc512(void){ return 2; }
#endif
