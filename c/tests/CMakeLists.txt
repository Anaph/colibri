# Percorso di build SEPARATO per i test (i motori si compilano con il make puro).
# Tutta la logica di test e' C compilato dal compilatore C; il C++ e' confinato
# ai file *_gtest.cc (solo macro TEST + dichiarazioni extern "C").
#
#   cmake -S tests -B tests/build && cmake --build tests/build -j
#   ctest --test-dir tests/build --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(colibri_tests LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# gtest: sistema se presente, altrimenti FetchContent (tag fissato, shallow)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

find_package(OpenMP)
enable_testing()
include(GoogleTest)
set(ENGDIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# opzionale: -march per abilitare i kernel AVX512/i8mm nei test locali
set(COLIBRI_TEST_MARCH "" CACHE STRING "march per i sorgenti C dei test (es. native)")

function(colibri_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE ${ENGDIR})
  target_link_libraries(${name} PRIVATE GTest::gtest_main m)
  if(OpenMP_C_FOUND)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_C)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(COLIBRI_TEST_MARCH)
    foreach(src ${ARGN})
      if(src MATCHES "\\.c$")
        set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-march=${COLIBRI_TEST_MARCH}")
      endif()
    endforeach()
  endif()
  gtest_discover_tests(${name})
endfunction()

colibri_test(test_headers headers_tests.c headers_gtest.cc)
colibri_test(test_qwen    qwen_tests.c    qwen_gtest.cc)
colibri_test(test_glm     glm_tests.c     glm_gtest.cc)

# utility C pura (non un test ctest): parita' del tokenizer contro un corpus HF
add_executable(tok_oracle tok_oracle.c)
target_include_directories(tok_oracle PRIVATE ${ENGDIR})
target_link_libraries(tok_oracle PRIVATE m)
