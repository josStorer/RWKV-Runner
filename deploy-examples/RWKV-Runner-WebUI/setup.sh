# install git python3.10 npm by yourself
# change model and strategy according to your hardware

sudo apt install python3-dev

git clone https://github.com/josStorer/RWKV-Runner --depth=1
python3 -m pip install torch torchvision torchaudio
python3 -m pip install -r RWKV-Runner/backend-python/requirements.txt
cd RWKV-Runner/frontend
npm ci
npm run build
cd ..

# optional: export ngrok_token=YOUR_NGROK_TOKEN
python3 ./backend-python/main.py --webui > log.txt & # this is only an example, you should use screen or other tools to run it in background

if [ ! -d models ]; then
    mkdir models
fi
wget -N https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-0.1B-v1-20230520-ctx4096.pth -P models/

curl http://127.0.0.1:8000/switch-model -X POST -H "Content-Type: application/json" -d '{"model":"./models/RWKV-4-World-0.1B-v1-20230520-ctx4096.pth","strategy":"cpu fp32","deploy":"true"}'
