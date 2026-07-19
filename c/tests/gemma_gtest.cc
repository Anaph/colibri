// Glue gtest per gemma_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int gm_prope(void);
int gm_geglu(void);
int gm_rmsnorm(void);
int gm_sliding(void);
int gm_tiny(void);
int gm_tiny_shared(void);
int gm_prefill_chunk(void);
int gm_kv_i8_sliding(void);
int gm_kv_i8_shared(void);
int gm_kv_i8_keqv(void);
int gm_tiny_keqv(void);
int gm_memknob_parity(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { int r = fn(); if (r == 2) GTEST_SKIP(); EXPECT_EQ(0, r); }

C_TEST(GemmaRope,    Proportional,       gm_prope)
C_TEST(GemmaMlp,     GeluTanh,           gm_geglu)
C_TEST(GemmaNorm,    BothConventions,    gm_rmsnorm)
C_TEST(GemmaAttn,    SlidingVsBruteForce, gm_sliding)
C_TEST(GemmaTiny,    HybridDeterministic, gm_tiny)
C_TEST(GemmaTiny,    KvShared,           gm_tiny_shared)
C_TEST(GemmaPrefill, ChunkBitExact,      gm_prefill_chunk)
C_TEST(GemmaKv8,     SlidingTolerance,   gm_kv_i8_sliding)
C_TEST(GemmaKv8,     SharedAliases,      gm_kv_i8_shared)
C_TEST(GemmaKv8,     KEqualsV,           gm_kv_i8_keqv)
C_TEST(GemmaTiny,    KEqualsV,           gm_tiny_keqv)
C_TEST(GemmaMemKnob, StreamTokenParity,  gm_memknob_parity)
