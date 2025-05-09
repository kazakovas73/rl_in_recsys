FROM ubuntu:20.04
WORKDIR /workspace

# python, dependencies for mujoco-py, from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    build-essential \
    patchelf \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
# installing mujoco distr
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV MUJOCO_PY_MJKEY_PATH=/root/.mujoco/mjkey.txt

ENV WANDB_API_KEY=723e97417e331c9ed2a41b55373e54f6bd3b4479

# installing poetry & env setup, mujoco_py compilation
COPY requirements/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --timeout=100

# RUN python -c "import mujoco_py"

COPY . /app
WORKDIR /app
