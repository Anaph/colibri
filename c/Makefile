# --- Target detection ---
# Ask the compiler what it actually targets (rofl0r's note on #129): a gcc/clang
# toolchain reports its target triple via `-dumpmachine`, e.g.
#   x86_64-w64-mingw32   x86_64-pc-cygwin   x86_64-unknown-linux-gnu
#   arm64-apple-darwin   powerpc64le-unknown-linux-gnu
# The triple follows the toolchain, not the host shell that `uname` reports, so
# it is correct under a native-Windows shell (no `uname` on PATH) AND when
# cross-compiling (e.g. make CC=x86_64-w64-mingw32-gcc on Linux). CC is chosen
# per-target below, so probe with the user's CC if they set one, else gcc
# (present on Linux and mingw, and a clang shim on macOS).
DETECT_CC := $(if $(filter default,$(origin CC)),gcc,$(CC))
TRIPLET   := $(shell $(DETECT_CC) -dumpmachine 2>/dev/null)

MINGW  := $(findstring mingw,$(TRIPLET))
CYGWIN := $(findstring cygwin,$(TRIPLET))
DARWIN := $(findstring darwin,$(TRIPLET))
LINUX  := $(findstring linux,$(TRIPLET))
IS_WIN := $(MINGW)$(CYGWIN)

# Fallbacks for the rare toolchain that does not answer -dumpmachine: keep the
# #129 signal (OS=Windows_NT, set in every Windows shell) for Windows, then
# `uname` for everything else, so no host regresses.
ifeq ($(TRIPLET),)
IS_WIN := $(if $(filter Windows_NT,$(OS)),1,)
ifeq ($(IS_WIN),)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
DARWIN  := $(findstring Darwin,$(UNAME_S))
endif
endif

TARGET_CPU := $(firstword $(subst -, ,$(TRIPLET)))
ifeq ($(TARGET_CPU),)
TARGET_CPU := $(UNAME_M)
endif
X86_64  := $(filter x86_64 amd64,$(TARGET_CPU))
AARCH64 := $(filter aarch64 arm64,$(TARGET_CPU))
PPC64   := $(filter powerpc64% ppc64%,$(TARGET_CPU))

ifneq (,$(DARWIN))
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
# `brew --prefix libomp` can print the formula's prospective path even when it
# is not installed, so verify both artifacts before adding unusable flags.
ifneq ($(and $(OMPDIR),$(wildcard $(OMPDIR)/include/omp.h),$(wildcard $(OMPDIR)/lib/libomp.*)),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp not found: building single-threaded. For multithreading: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
# Opt-in: ARCH=native appends -mcpu=native (arm64 clang uses -mcpu, not -march),
# which unlocks the i8mm SMMLA int8/int4 dot kernels in glm.c. ARCH unset ->
# no -mcpu, default build byte-identical. Apple clang knows apple-m4 / native.
ifneq ($(ARCH),)
CFLAGS += -mcpu=$(ARCH)
endif
LDFLAGS = -lm $(OMPL)
EXE     =
else ifneq ($(IS_WIN),)
# --- Windows 11 x86-64 (MinGW-w64 / MSYS2) ---
# GCC + libgomp + winpthreads: pthread, OpenMP, clock_gettime, opendir/readdir,
# AVX2 intrinsics - tutto gratis, nessun porting.
# ARCH default = x86-64-v3 (portable binary with AVX2). For max speed on THIS
# machine use ARCH=native: on AVX-VNNI CPUs (Intel Alder Lake+, Meteor Lake+)
# it also unlocks the 128-bit VPDPBUSD int8/int4 dot kernel (dot_i8i8/dot_i4i8),
# which the x86-64-v3 baseline does not define. The #ifdef guards in glm.c mean
# a v3 build simply compiles out the VNNI path - safe on any x86-64.
CC      = gcc
ARCH   ?= x86-64-v3
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
# -lpsapi: compat.h calls GetProcessMemoryInfo (rss_gb). It's linked via
# #pragma comment(lib,"psapi.lib") for MSVC, but MinGW gcc ignores that pragma
# (warns), so link psapi explicitly or the build fails undefined-reference on
# modern GCC (e.g. 16.x under UCRT). Harmless where the pragma also resolves it.
LDFLAGS = -lm -fopenmp -static -lpsapi
EXE     = .exe
else
ifneq (,$(PPC64))
# --- Linux PowerPC (POWER8/POWER9/POWER10) ---
# PowerPC GCC uses -mcpu, not -march. ARCH=native works on gcc >= 4.7.
# The AVX2/NEON kernels fall back to the portable scalar C path
# (validated token-exact vs the transformers oracle on a POWER8 S824).
CC      = gcc
ARCH   ?= native
CFLAGS  = -O3 -mcpu=$(ARCH) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread
EXE     =
else
# --- Linux / *BSD x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
# -pthread: Linux lo tira dentro via -fopenmp, i *BSD no e le pthread_* non risolvono (#219).
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread
EXE     =
endif
endif

all: glm$(EXE) olmoe$(EXE) qwen$(EXE) gemma$(EXE)

# phony aliases → '<name>.exe' on Windows (so 'make glm' works on every platform)
glm: glm$(EXE)
olmoe: olmoe$(EXE)
qwen: qwen$(EXE)
gemma: gemma$(EXE)

glm$(EXE): glm.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h
	$(CC) $(CFLAGS) glm.c -o glm$(EXE) $(LDFLAGS)

olmoe$(EXE): olmoe.c st.h json.h compat.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe$(EXE) $(LDFLAGS)

qwen$(EXE): qwen.c nn.h st.h json.h tok.h tok_unicode.h compat.h
	$(CC) $(CFLAGS) qwen.c -o qwen$(EXE) $(LDFLAGS)

gemma$(EXE): gemma.c nn.h st.h json.h tok.h tok_unicode.h compat.h
	$(CC) $(CFLAGS) gemma.c -o gemma$(EXE) $(LDFLAGS)

# Use a baseline that matches the compiler target. macOS already targets a
# portable baseline when ARCH is empty; forcing the x86 value there breaks
# Apple Silicon. Unknown targets use native rather than an invalid x86 flag.
ifneq (,$(DARWIN))
PORTABLE_ARCH =
else ifneq (,$(AARCH64))
PORTABLE_ARCH = armv8-a
else ifneq (,$(PPC64))
PORTABLE_ARCH = power8
else ifneq (,$(X86_64))
PORTABLE_ARCH = x86-64-v3
else
PORTABLE_ARCH = native
endif

portable:
	$(MAKE) all ARCH=$(PORTABLE_ARCH)

# GoogleTest suite: separate CMake build path (engines stay on plain make).
# First configure downloads a pinned gtest via FetchContent unless a system
# GTest is found.
TESTS_BUILD ?= tests/build
test:
	cmake -S tests -B $(TESTS_BUILD)
	cmake --build $(TESTS_BUILD) -j
	ctest --test-dir $(TESTS_BUILD) --output-on-failure

# Local validation: one portable CPU build and dependency-free tests.
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

clean:
	rm -f glm glm.exe olmoe olmoe.exe qwen qwen.exe gemma gemma.exe
	rm -rf $(TESTS_BUILD)

.PHONY: all glm olmoe qwen gemma portable test check clean
