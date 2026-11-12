FROM python:3.13-slim

# uv: misma versión que en local
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Solo dependencias (capa cacheada: se reinstalan solo si cambia el lock).
#    La caché de uv se monta durante el build y no queda dentro de la imagen.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) Código del paquete
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Usuario sin privilegios: solo puede escribir en logs/
RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

EXPOSE 8000

# Docker comprueba cada 30 s que la API responde
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "kkbox_churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]