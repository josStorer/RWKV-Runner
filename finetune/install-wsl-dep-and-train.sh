echo $@

if [[ ${cnMirror} == 1 ]]; then
  export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
  if grep -q "mirrors.aliyun.com" /etc/apt/sources.list; then
    echo "apt cnMirror already set"
  else
    sudo sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list
    sudo apt update
  fi
fi

if dpkg -s "gcc" >/dev/null 2>&1; then
  echo "gcc installed"
else
  sudo apt -y install gcc
fi

if dpkg -s "python3-pip" >/dev/null 2>&1; then
  echo "pip installed"
else
  sudo apt -y install python3-pip
fi

if dpkg -s "python3-dev" >/dev/null 2>&1; then
  echo "python3-dev installed"
else
  sudo apt -y install python3-dev
fi

if dpkg -s "ninja-build" >/dev/null 2>&1; then
  echo "ninja installed"
else
  sudo apt -y install ninja-build
fi

if dpkg -s "cuda" >/dev/null 2>&1 && dpkg -s "cuda" | grep Version | awk '{print $2}' | grep -q "12"; then
  echo "cuda 12 installed"
else
  wget -N https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
  sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget -N https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
  sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
  sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cuda
fi

if python3 -c "import pkg_resources; pkg_resources.require(open('./finetune/requirements.txt',mode='r'))" &>/dev/null; then
  echo "requirements satisfied"
else
  python3 -m pip install -r ./finetune/requirements.txt
fi

echo "loading $loadModel"
modelInfo=$(python3 ./finetune/get_layer_and_embd.py $loadModel 6.0)
echo $modelInfo
if [[ $modelInfo =~ "--n_layer" ]]; then
  sudo rm -rf /root/.cache/torch_extensions
  python3 ./finetune/lora/$modelInfo $@ --proj_dir lora-models --data_type binidx --lora \
    --lora_parts=att,ffn,time,ln --strategy deepspeed_stage_2 --accelerator gpu --ds_bucket_mb 2
else
  echo "modelInfo is invalid"
  exit 1
fi
