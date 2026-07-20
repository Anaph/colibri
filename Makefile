.PHONY: all glm olmoe qwen portable test check clean docker docker-test docker-run

all glm olmoe qwen portable test check clean:
	$(MAKE) -C c $@

# ---- docker ----
DOCKER_IMAGE ?= colibri
MODELS_DIR   ?= $(HOME)/models

docker:
	docker build -t $(DOCKER_IMAGE) .

# compila ED esegue la suite gtest dentro la build (fallisce se un test fallisce)
docker-test:
	docker build --target test -t $(DOCKER_IMAGE)-test .

# chat qwen interattiva; le manopole passano dall'ambiente (SNAP=/models/...)
docker-run:
	docker run --rm -it -v $(MODELS_DIR):/models:ro \
	  -e SNAP -e GGUF -e PROMPT -e REF -e NGEN -e CTX -e TEMP -e NUCLEUS -e SEED \
	  -e QBITS -e QGROUP -e KV_BITS -e MEM_GB -e MEM_FRAC -e MICRO -e MICRO_DROP \
	  -e PREFILL_CHUNK -e THREADS -e TOKENS -e TTA -e LORA \
	  $(DOCKER_IMAGE)
