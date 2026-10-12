# SPDX-License-Identifier: BSD-3-Clause
#
# Workbench: the server runs as root from /opt/workbench (unreadable to the agent);
# the model's commands run as the unprivileged `agent` user inside /app.

FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/moarshy/workbench" \
      org.opencontainers.image.description="Workbench: OpenEnv workspace tasks solved with run_bash, graded on output files" \
      org.opencontainers.image.licenses="BSD-3-Clause"

# Workspace tooling the agent can use, plus tini so the server (PID 1) reaps stray processes.
RUN apt-get update && \
    apt-get install -y --no-install-recommends tini git jq sqlite3 ripgrep less procps && \
    rm -rf /var/lib/apt/lists/*

# The agent's python: system interpreter with common data libraries.
RUN pip install --no-cache-dir pandas==2.2.3 numpy==2.1.3 pytest==8.3.4

# The server's own environment, kept separate from the agent's.
RUN pip install --no-cache-dir uv
COPY . /opt/workbench
WORKDIR /opt/workbench
RUN uv sync --frozen --no-editable 2>/dev/null || uv sync --no-editable

RUN useradd --create-home --shell /bin/bash agent && \
    mkdir -p /app && chown agent:agent /app && \
    chmod -R go-rwx /opt/workbench

ENV WORKBENCH_ROOT=/app \
    PYTHONPATH=/opt/workbench

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/workbench/.venv/bin/uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
