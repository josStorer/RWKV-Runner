Client Download URL:
客户端下载地址:
クライアントのダウンロードURL:
https://github.com/josStorer/RWKV-Runner/releases/latest/download/RWKV-Runner_linux_x64

Please execute this program in an empty directory. All related dependencies will be placed in this directory.
请将本程序放在一个空目录内执行, 所有相关依赖均会放置于此目录.
このプログラムを空のディレクトリで実行してください。関連するすべての依存関係は、このディレクトリに配置されます。

On Linux, install Python dependencies manually. Use a Python 3.10 virtual environment instead of running pip against the system Python. Install Python 3.10 first if needed; if your distribution packages venv or Python development headers separately, install the versions that match your Python 3.10 interpreter.
Linux 上需要手动安装 Python 依赖。请使用 Python 3.10 虚拟环境，不要用 pip 直接修改系统 Python。必要时先安装 Python 3.10；如果发行版单独提供 venv 或 Python 开发头文件的软件包，请安装与 Python 3.10 解释器对应的版本。
Linux では Python の依存関係を手動でインストールします。システムの Python に直接 pip でインストールせず、Python 3.10 の仮想環境を使用してください。必要に応じて Python 3.10 を先にインストールし、venv や Python の開発用ヘッダーが別パッケージの場合は Python 3.10 に対応するものをインストールしてください。

From the directory containing the executable, create the environment, print its interpreter path, and launch RWKV-Runner to create the backend-python directory.
在可执行文件所在目录中创建虚拟环境，输出其解释器路径，并启动 RWKV-Runner 来生成 backend-python 目录。
実行ファイルがあるディレクトリで仮想環境を作成し、インタプリタのパスを表示してから、RWKV-Runner を起動して backend-python ディレクトリを作成してください。

python3.10 -m venv .venv
printf '%s\n' "$PWD/.venv/bin/python"
chmod +x ./RWKV-Runner_linux_x64
./RWKV-Runner_linux_x64

Paste the printed absolute path into Settings > Advanced > Custom Python Path. Once backend-python exists, run the install command in another terminal from the same directory as the executable.
将上面输出的绝对路径粘贴到「设置 > 高级 > 自定义Python路径」。backend-python 目录生成后，另开一个终端，在可执行文件所在目录运行安装命令。
表示された絶対パスを「設定 > 高度な > カスタムPythonパス」に貼り付けます。backend-python ディレクトリが作成されたら、別のターミナルで実行ファイルがあるディレクトリに移動してインストールコマンドを実行します。

./.venv/bin/python -m pip install -r backend-python/requirements.txt
# Alternatively / 或者 / または:
# ./.venv/bin/python -m pip install -r backend-python/requirements_without_cyac.txt

# See More: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
