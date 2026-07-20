# Immagine multi-stadio di colibri.
#   docker build -t colibri .                  -> runtime slim con i 4 motori
#   docker build --target test -t colibri-test .  -> compila ED esegue la suite gtest
# I binari sono la build PORTABILE (baseline per l'arch dell'immagine: x86-64-v3
# su amd64, armv8-a su arm64), quindi l'immagine gira su qualunque host moderno
# della stessa architettura. Per il massimo su UNA macchina: ARCH=native.
# Dietro un mirror di registry: --build-arg BASE=<mirror>/debian:bookworm-slim.

ARG BASE=debian:bookworm-slim
FROM ${BASE} AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY c/ c/
ARG ARCH=
RUN make -C c $(test -n "$ARCH" && echo "all ARCH=$ARCH" || echo portable)

# stadio test (opzionale, non entra nell'immagine finale): l'intera suite
# gtest gira DENTRO la build — se un test fallisce, la build fallisce.
FROM build AS test
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN make -C c test

# runtime: solo i binari + libgomp (OpenMP e' linkato dinamico)
FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/c/qwen /src/c/gemma /src/c/glm /src/c/olmoe /usr/local/bin/
WORKDIR /work
# i modelli si montano qui: -v ~/models:/models:ro, poi SNAP=/models/<dir>
# (oppure GGUF=/models/<file.gguf> per il single-file)
VOLUME /models
ENTRYPOINT ["/usr/local/bin/qwen"]
