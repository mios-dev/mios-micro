# syntax=docker/dockerfile:1
# AI-hint: Multi-stage Containerfile for mios-micro — packaging the Qwen2.5-Coder-1.5B quantized GGUF and static llama-server into an OCI container and artifact.
# AI-related: /usr/share/containers/systemd/, usr/share/mios/llamacpp/mios-llm-light.yaml, usr/lib/bootc/bound-images.d/

ARG LLAMA_VERSION=b3600
FROM ghcr.io/ggerganov/llama.cpp:server AS server-base

FROM registry.fedoraproject.org/fedora-minimal:latest AS runner

LABEL org.opencontainers.image.title="mios-micro" \
      org.opencontainers.image.description="Dedicated resident miniature model (1.5B) and OCI artifact for MiOS" \
      org.opencontainers.image.vendor="MiOS" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.url="https://github.com/mios-dev/mios-micro"

RUN microdnf install -y ca-certificates libgomp libstdc++ && \
    microdnf clean all && \
    mkdir -p /models /app /var/lib/mios/llamacpp/slots

# Copy llama-server from upstream static/optimized base
COPY --from=server-base /app/llama-server /app/llama-server

# Model layer (placeholder or populated from build-arg/artifacts)
# Note: For production builds, weights are packaged into /models/mios-micro-1.5b.gguf
ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then \
      curl -L -o /models/mios-micro-1.5b.gguf "$MODEL_URL"; \
    fi

EXPOSE 8500

ENV PORT=8500
ENV MODEL_PATH=/models/mios-micro-1.5b.gguf

ENTRYPOINT ["/bin/sh", "-c", "exec /app/llama-server --model \"$MODEL_PATH\" --port \"$PORT\" --host 0.0.0.0 --ctx-size 8192 --parallel 1 --cache-reuse 256 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --slot-save-path /var/lib/mios/llamacpp/slots --jinja"]
