if [[ ${cnMirror} == 1 ]]; then
  export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
  if grep -q "mirrors.aliyun.com" /etc/apt/sources.list; then
    echo "apt cnMirror already set"
  else
    sudo sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list
    sudo apt update
  fi
fi

if dpkg -s "python3-pip" >/dev/null 2>&1; then
  echo "pip installed"
else
  sudo apt install python3-pip
fi

if dpkg -s "ninja-build" >/dev/null 2>&1; then
  echo "ninja installed"
else
  sudo apt install ninja-build
fi

if dpkg -s "cuda" >/dev/null 2>&1; then
  echo "cuda installed"
else
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
  sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.0-1_amd64.deb
  sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.0-1_amd64.deb
  sudo cp /var/cuda-repo-wsl-ubuntu-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cuda
fi

if python3 -c "import pkg_resources; pkg_resources.require(open('./finetune/requirements.txt',mode='r'))" &>/dev/null; then
  echo "requirements satisfied"
else
  python3 -m pip install -r ./finetune/requirements.txt
fi

echo "loading $loadModel"
modelInfo=$(python3 ./finetune/get_layer_and_embd.py $loadModel)
echo $modelInfo

python3 ./finetune/lora/train.py $modelInfo $@ --proj_dir lora-models --data_type binidx --lora \
  --lora_parts=att,ffn,time,ln --strategy deepspeed_stage_2 --accelerator gpu
