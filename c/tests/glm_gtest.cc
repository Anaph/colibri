// Glue gtest per glm_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int gt_idot_kernel(void);
int gt_idot_driver(void);
int gt_kv_realloc(void);
int gt_uring(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { int r = fn(); if (r == 2) GTEST_SKIP(); EXPECT_EQ(0, r); }

C_TEST(GlmIdot,  KernelExact,   gt_idot_kernel)
C_TEST(GlmIdot,  DriverExact,   gt_idot_driver)
C_TEST(GlmKv,    Realloc,       gt_kv_realloc)
C_TEST(GlmUring, ReadAndExpert, gt_uring)
