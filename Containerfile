# syntax=docker/dockerfile:1
# AI-hint: Runtime image for mios-micro -- the upstream llama.cpp server image plus the pinned, sha256-verified Qwen2.5-Coder-1.5B GGUF; the ModelPack artifact is attached to this image's digest as an OCI referrer.
# AI-related: pyproject.toml ([tool.mios-micro.package] gguf_file), Kitfile (model.path), .github/workflows/package-oci.yml, usr/share/mios/llamacpp/mios-llm-light.yaml

FROM ghcr.io/ggml-org/llama.cpp:server

LABEL org.opencontainers.image.title="mios-micro" \
      org.opencontainers.image.description="Dedicated resident miniature model (1.5B) and OCI artifact for MiOS" \
      org.opencontainers.image.vendor="MiOS" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.url="https://github.com/mios-dev/mios-micro"

RUN mkdir -p /var/lib/mios/llamacpp/slots

# Must equal [tool.mios-micro.package].gguf_file in pyproject.toml. The
# package-oci workflow downloads it to models/ at gguf_revision and verifies
# gguf_sha256 before this build runs.
ARG GGUF_FILE=qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
COPY models/${GGUF_FILE} /models/${GGUF_FILE}

ENV PORT=8500
ENV MODEL_PATH=/models/${GGUF_FILE}

EXPOSE 8500

ENTRYPOINT ["/bin/sh", "-c", "exec /app/llama-server --model \"$MODEL_PATH\" --port \"$PORT\" --host 0.0.0.0 --ctx-size 8192 --parallel 1 --cache-reuse 256 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --slot-save-path /var/lib/mios/llamacpp/slots --jinja"]
