FROM nvidia/cuda:13.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    vim \
    git \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /code/requirements.txt && \
    /opt/venv/bin/python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (ipykernel)"

COPY . /code

RUN find /code -maxdepth 2 -name "*.sh" -exec sed -i 's/\r$//' {} + && \
    chmod +x /code/predict.sh /code/start_jupyter.sh /code/docker_build_and_verify.sh && \
    mkdir -p /result

CMD ["/bin/bash", "/code/predict.sh"]
