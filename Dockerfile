FROM node:21-slim AS frontend

RUN echo "registry=https://registry.npmmirror.com/" > ~/.npmrc

WORKDIR /app

COPY manifest.json manifest.json
COPY frontend frontend

WORKDIR /app/frontend

RUN npm ci
RUN npm run build

FROM nvidia/cuda:11.6.1-devel-ubuntu20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -yq git curl wget build-essential ninja-build aria2 jq software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get install -y g++-11 python3.10 python3.10-distutils python3.10-dev && \
    curl -sS http://mirrors.aliyun.com/pypi/get-pip.py | python3.10 && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir cmake

FROM runtime AS librwkv

WORKDIR /app

RUN git clone --depth 1 https://github.com/RWKV/rwkv.cpp.git && \
    cd rwkv.cpp && \
    git submodule update --init --recursive --depth=1 && \
    mkdir -p build && \
    cd build && \
    cmake -G Ninja .. && \
    cmake --build .

FROM runtime AS final

WORKDIR /app

COPY ./backend-python/requirements.txt ./backend-python/requirements.txt

RUN python3.10 -m pip install --no-cache-dir --quiet -r ./backend-python/requirements.txt

COPY . .
COPY --from=frontend /app/frontend/dist /app/frontend/dist
COPY --from=librwkv /app/rwkv.cpp/build/librwkv.so /app/backend-python/rwkv_pip/cpp/librwkv.so

EXPOSE 27777

CMD ["python3.10", "./backend-python/main.py", "--port", "27777", "--host", "0.0.0.0", "--webui"]
