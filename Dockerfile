FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential pkg-config libssl-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY python/ python/
COPY pyproject.toml ./
COPY benchmarks/results/ benchmarks/results/

RUN pip install --no-cache-dir maturin && \
    maturin build --release -o /wheels && \
    pip install --no-cache-dir /wheels/*.whl

FROM python:3.13-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY python/ python/
COPY benchmarks/results/ benchmarks/results/

ENV HOST=0.0.0.0
EXPOSE 10000

CMD ["python", "-m", "tokenisers.dashboard"]
