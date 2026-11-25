Client Download URL:
客户端下载地址:
クライアントのダウンロードURL:
https://github.com/josStorer/RWKV-Runner/releases/latest/download/RWKV-Runner_macos_universal.zip

Please execute this program in an empty directory. All related dependencies will be placed in this directory.
请将本程序放在一个空目录内执行, 所有相关依赖均会放置于此目录.
このプログラムを空のディレクトリで実行してください。関連するすべての依存関係は、このディレクトリに配置されます。

Please execute the following command in the terminal to remove the permission restrictions of this app, and then this program can work properly:
请在终端执行以下命令解除本app的权限限制, 然后本程序才可以正常工作:
このアプリの権限制限を解除するために、ターミナルで以下のコマンドを実行してください。その後、このプログラムは正常に動作するようになります:

sudo xattr -r -d com.apple.quarantine ./RWKV-Runner.app

Use Python 3.10 in a virtual environment so pip does not try to modify an externally managed Python installation (for example, Homebrew's).
请在虚拟环境中使用 Python 3.10，避免 pip 修改由外部工具（如 Homebrew）管理的 Python 安装。
Homebrew などで管理されている Python 環境を pip で変更しないよう、Python 3.10 の仮想環境を使用してください。

Install Python 3.10 first if python3.10 is not already available.
如果找不到 python3.10 命令，请先安装 Python 3.10。
python3.10 コマンドがない場合は、先に Python 3.10 をインストールしてください。

In Terminal, run these commands from the directory containing RWKV-Runner.app. Use a directory whose full path has no spaces.
请在终端中进入 RWKV-Runner.app 所在目录并运行以下命令。该目录的完整路径不能包含空格。
ターミナルで RWKV-Runner.app があるディレクトリに移動して、以下のコマンドを実行してください。空白を含まない絶対パスのディレクトリを使用してください。

python3.10 -m venv .venv
printf '%s\n' "$PWD/.venv/bin/python"

Open RWKV-Runner and paste the printed absolute path into Settings > Advanced > Custom Python Path. Then run a model and use the Install prompt if dependencies are missing. Activating the virtual environment in Terminal alone does not configure the app.
打开 RWKV-Runner，将上面输出的绝对路径粘贴到「设置 > 高级 > 自定义Python路径」。然后运行模型，如果缺少依赖，按提示点击「安装」。仅在终端中激活虚拟环境不会改变应用所用的解释器。
RWKV-Runner を開き、表示された絶対パスを「設定 > 高度な > カスタムPythonパス」に貼り付けてください。その後モデルを実行し、依存関係が不足している場合は案内に従ってインストールしてください。ターミナルで仮想環境を有効にするだけではアプリの設定は変わりません。
