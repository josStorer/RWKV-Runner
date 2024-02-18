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

RUN apt update && \
    apt install -yq git curl wget build-essential ninja-build aria2 jq software-properties-common

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt install -y g++-11 python3.10 python3.10-distutils python3.10-dev && \
    curl -sS http://mirrors.aliyun.com/pypi/get-pip.py | python3.10

FROM runtime AS final

WORKDIR /app

COPY ./backend-python/requirements.txt ./backend-python/requirements.txt

RUN python3.10 -m pip install --quiet -r ./backend-python/requirements.txt

COPY . .
COPY --from=frontend /app/frontend/dist /app/frontend/dist

EXPOSE 27777

CMD ["python3.10", "./backend-python/main.py", "--port", "27777", "--host", "0.0.0.0", "--webui"]
