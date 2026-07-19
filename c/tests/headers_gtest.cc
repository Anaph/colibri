// Glue gtest per headers_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
// Tutta la logica sta nel C; qui si mappa 0=ok, 1=fail, 2=skip.
#include <gtest/gtest.h>

extern "C" {
int ht_json(void);
int ht_json_free(void);
int ht_st(void);
int ht_gguf_header(void);
int ht_gguf_q40_q80(void);
int ht_gguf_kquants(void);
int ht_tok_pairs(void);
int ht_tok_strings(void);
int ht_tok_arena(void);
int ht_tok_gguf(void);
int ht_tok_sp(void);
int ht_tok_sp_detect(void);
int ht_tier(void);
int ht_grammar(void);
int ht_schema_gbnf(void);
int ht_decode_batch(void);
int ht_compat_direct(void);
int ht_i4_acc512(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { int r = fn(); if (r == 2) GTEST_SKIP(); EXPECT_EQ(0, r); }

C_TEST(Json,        Parse,          ht_json)
C_TEST(Json,        FreeTree,       ht_json_free)
C_TEST(Safetensors, Primitives,     ht_st)
C_TEST(Gguf,        HeaderIndex,    ht_gguf_header)
C_TEST(Gguf,        Q40Q80Repack,   ht_gguf_q40_q80)
C_TEST(Gguf,        KQuants,        ht_gguf_kquants)
C_TEST(Tok,         MergesPairs,    ht_tok_pairs)
C_TEST(Tok,         MergesStrings,  ht_tok_strings)
C_TEST(Tok,         StringPool,     ht_tok_arena)
C_TEST(Tok,         FromGguf,       ht_tok_gguf)
C_TEST(Tok,         SentencePiece,  ht_tok_sp)
C_TEST(Tok,         SpDetection,    ht_tok_sp_detect)
C_TEST(Tier,        SwapDecayLfru,  ht_tier)
C_TEST(Grammar,     Pda,            ht_grammar)
C_TEST(SchemaGbnf,  Compile,        ht_schema_gbnf)
C_TEST(DecodeBatch, Helpers,        ht_decode_batch)
C_TEST(Compat,      DirectIo,       ht_compat_direct)
C_TEST(I4Acc512,    Accuracy,       ht_i4_acc512)
