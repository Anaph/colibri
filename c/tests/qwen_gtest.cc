// Glue gtest per qwen_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int qt_rope(void);
int qt_gqa(void);
int qt_quant(void);
int qt_quant_batch(void);
int qt_sampler(void);
int qt_edges(void);
int qt_gated_layout(void);
int qt_deltanet_small(void);
int qt_deltanet_large(void);
int qt_tiny_dense(void);
int qt_tiny_qbits(void);
int qt_tiny_hybrid(void);
int qt_memknob_parity(void);
int qt_memknob_env(void);
int qt_micro_parity(void);
int qt_stw_st_parity(void);
int qt_lora_zero_noop(void);
int qt_lora_effect(void);
int qt_lora_roundtrip(void);
int qt_bw_rope_inv(void);
int qt_bw_rmsnorm(void);
int qt_grad_fd(void);
int qt_train_descends(void);
int qt_train_e2e(void);
int qt_tta_off_bitexact(void);
int qt_tta_cache_boost(void);
int qt_tta_bias_direction(void);
int qt_tta_ppl_proxy(void);
int qt_tta_lora_off_noop(void);
int qt_tta_lora_direction(void);
int qt_tta_lora_reset(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { int r = fn(); if (r == 2) GTEST_SKIP(); EXPECT_EQ(0, r); }

C_TEST(QwenRope,     MatchesDoubleRef,  qt_rope)
C_TEST(QwenGqa,      MatchesReplicated, qt_gqa)
C_TEST(QwenQuant,    Int8Tolerance,     qt_quant)
C_TEST(QwenQuant,    BatchBitExact,     qt_quant_batch)
C_TEST(QwenSampler,  Deterministic,     qt_sampler)
C_TEST(QwenDeltanet, NumericEdges,      qt_edges)
C_TEST(QwenGated,    QProjLayout,       qt_gated_layout)
C_TEST(QwenDeltanet, DoubleRefSmall,    qt_deltanet_small)
C_TEST(QwenDeltanet, DoubleRefLarge,    qt_deltanet_large)
C_TEST(QwenTiny,     DenseDeterministic, qt_tiny_dense)
C_TEST(QwenTiny,     Qbits8Finite,       qt_tiny_qbits)
C_TEST(QwenTiny,     HybridDeterministic, qt_tiny_hybrid)
C_TEST(QwenMemKnob,  StreamTokenParity,   qt_memknob_parity)
C_TEST(QwenMemKnob,  EnvPrecedence,       qt_memknob_env)
C_TEST(QwenMicro,    StreamBitExact,      qt_micro_parity)
C_TEST(QwenLora,     StwStParity,         qt_stw_st_parity)
C_TEST(QwenLora,     ZeroBNoop,           qt_lora_zero_noop)
C_TEST(QwenLora,     EffectMatchesRef,    qt_lora_effect)
C_TEST(QwenLora,     Roundtrip,           qt_lora_roundtrip)
C_TEST(QwenTrain,    RopeInverse,         qt_bw_rope_inv)
C_TEST(QwenTrain,    RmsNormBackward,     qt_bw_rmsnorm)
C_TEST(QwenTrain,    GradFiniteDiff,      qt_grad_fd)
C_TEST(QwenTrain,    LossDescends,        qt_train_descends)
C_TEST(QwenTrain,    EndToEnd,            qt_train_e2e)
C_TEST(QwenTta,      OffBitExact,         qt_tta_off_bitexact)
C_TEST(QwenTta,      CacheBoost,          qt_tta_cache_boost)
C_TEST(QwenTta,      BiasDirection,       qt_tta_bias_direction)
C_TEST(QwenTta,      PplProxy,            qt_tta_ppl_proxy)
C_TEST(QwenTta,      LoraZeroBNoop,       qt_tta_lora_off_noop)
C_TEST(QwenTta,      LoraDirection,       qt_tta_lora_direction)
C_TEST(QwenTta,      LoraReset,           qt_tta_lora_reset)
