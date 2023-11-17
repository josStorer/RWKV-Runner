: install git python3.10 npm by yourself
: change model and strategy according to your hardware

git clone https://github.com/josStorer/RWKV-Runner --depth=1
python -m pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r RWKV-Runner/backend-python/requirements.txt
cd RWKV-Runner/frontend
call npm ci
call npm run build
cd ..

: optional: set ngrok_token=YOUR_NGROK_TOKEN
start python ./backend-python/main.py --webui
start "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" "http://127.0.0.1:8000"

powershell -Command "(Test-Path ./models) -or (mkdir models)"
powershell -Command "Import-Module BitsTransfer"
powershell -Command "(Test-Path ./models/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth) -or (Start-BitsTransfer https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth ./models/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth)"
powershell -Command "Invoke-WebRequest http://127.0.0.1:8000/switch-model -Method POST -ContentType 'application/json' -Body '{\"model\":\"./models/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth\",\"strategy\":\"cuda fp32 *20+\",\"deploy\":\"true\"}'"
