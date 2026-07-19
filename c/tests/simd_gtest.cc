// Glue gtest per simd_tests.c: SOLO dichiarazioni extern "C" e macro TEST.
#include <gtest/gtest.h>

extern "C" {
int st_dot_i8i8(void);
int st_dot_i4i8(void);
int st_dot_f32i8(void);
int st_dot_f32(void);
int st_qrow(void);
int st_dn_rows(void);
int st_report(void);
}

#define C_TEST(suite, name, fn) \
    TEST(suite, name) { int r = fn(); if (r == 2) GTEST_SKIP(); EXPECT_EQ(0, r); }

C_TEST(SimdIdot, ExactVsInt64Ref,   st_dot_i8i8)
C_TEST(SimdI4,   ExactVsInt64Ref,   st_dot_i4i8)
C_TEST(SimdF32I8, MatchesDoubleRef, st_dot_f32i8)
C_TEST(SimdF32,  MatchesDoubleRef,  st_dot_f32)
C_TEST(SimdQrow, Reconstruction,    st_qrow)
C_TEST(SimdDn,   RowKernelsVsScalar, st_dn_rows)
C_TEST(SimdInfo, ReportKernels,     st_report)
