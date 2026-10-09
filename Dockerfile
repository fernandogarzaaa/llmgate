# syntax=docker/dockerfile:1.7
# ---- build: produce a wheel
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir "build>=1.2"
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

# ---- runtime
FROM python:3.12-slim
ARG EXTRAS="redis,postgres,otel"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    LLMGATE_CONFIG=/etc/llmgate/config.yaml
RUN groupadd --system --gid 10001 llmgate \
 && useradd --system --uid 10001 --gid llmgate --home-dir /app --shell /usr/sbin/nologin llmgate
WORKDIR /app
COPY --from=build /dist/*.whl /tmp/
# Optional integrations (costroute, injectguard) are installed from wheels placed in ./wheels/ if present.
COPY wheels/ /tmp/wheels/
RUN set -eux; whl="$(ls /tmp/llmgate-*.whl)"; \
    pip install "${whl}[${EXTRAS}]"; \
    if ls /tmp/wheels/*.whl >/dev/null 2>&1; then pip install /tmp/wheels/*.whl; fi; \
    rm -rf /tmp/*.whl /tmp/wheels
COPY examples/config.mock.yaml /etc/llmgate/config.yaml
RUN mkdir -p /data && chown llmgate:llmgate /data
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"
ENTRYPOINT ["llmgate"]
CMD ["serve"]
