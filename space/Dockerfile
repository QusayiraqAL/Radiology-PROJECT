# Runs on any container host. Two details are not optional: the process must not need root
# (a Hugging Face Space runs it as uid 1000, and running as root elsewhere is bad practice
# either way), and caches must land somewhere that user can write - torch, matplotlib and
# huggingface_hub all write on first use, and the failure is a crash at startup, not a
# warning. Port 7860 is the Spaces default; Caddy or -p maps it to whatever the host wants.
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

# torch first, and where it comes from depends on the architecture:
#   x86_64  - PyPI's default wheel is the CUDA build, gigabytes of driver libraries for a
#             machine with no GPU, so it comes from the CPU index instead.
#   aarch64 - there is no CUDA build for ARM at all; PyPI's wheel is already CPU-only, and
#             download.pytorch.org/whl/cpu carries no aarch64 wheel, so asking for that
#             index on ARM fails to resolve. Verified against PyPI: torch 2.13.0 ships
#             cp310-cp310-manylinux_2_28_aarch64.
# torchxrayvision and medmnist go in last with --no-deps: both declare a torch dependency
# and, resolved freely, reinstall it over whichever build landed here.
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$(uname -m)" = "aarch64" ]; then \
        pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0; \
    else \
        pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    pip install --no-cache-dir -r requirements-space.txt && \
    pip install --no-cache-dir --no-deps torchxrayvision==1.5.2 medmnist==3.0.2

COPY --chown=user . .

# main.py resolves models/ and the UI page relative to its own directory.
WORKDIR /home/user/app/api
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
