# A Hugging Face Space runs this container as uid 1000 and routes traffic to $PORT (7860
# by default). Both matter: a build that writes caches into / or listens on 8000 starts
# cleanly and then serves nothing.
FROM python:3.10-slim

# libgomp1 is OpenMP, which torch links against; libglib2.0-0 is pulled in by the imaging
# stack. Neither is in -slim, and the failure without them is an ImportError at startup.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    PYTHONUNBUFFERED=1

USER user
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .

# CPU torch first and on its own index. torchxrayvision and medmnist both declare a torch
# dependency; installed normally they drag in the CUDA build - gigabytes of driver libraries
# for a Space with no GPU - so they go in last with --no-deps, against the torch already here.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-space.txt && \
    pip install --no-cache-dir --no-deps torchxrayvision==1.5.2 medmnist==3.0.2

COPY --chown=user . .

# main.py resolves models/ and the UI page relative to its own directory.
WORKDIR /home/user/app/api
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
