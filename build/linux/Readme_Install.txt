Client Download URL:
客户端下载地址:
クライアントのダウンロードURL:
https://github.com/josStorer/RWKV-Runner/releases/latest/download/RWKV-Runner_linux_x64

For Mac and Linux users, please manually install Python 3.10 (usually the latest systems come with it built-in). You can specify the Python interpreter to use in Settings.
对于Mac和Linux用户，请手动安装 Python3.10 (通常最新的系统已经内置了). 你可以在设置中指定使用的Python解释器.
MacおよびLinuxのユーザーの方は、Python3.10を手動でインストールしてください（通常、最新のシステムには既に組み込まれています）。 設定メニューで使用するPythonインタプリタを指定することができます。

Please execute this program in an empty directory. All related dependencies will be placed in this directory.
请将本程序放在一个空目录内执行, 所有相关依赖均会放置于此目录.
このプログラムを空のディレクトリで実行してください。関連するすべての依存関係は、このディレクトリに配置されます。

On Linux system, this program cannot invoke the terminal for automatic dependency installation. You must manually execute the following commands for installation so that it can be used normally:
在Linux系统下, 本程序无法调用终端自动安装依赖, 你必须手动执行以下命令进行安装, 之后方可正常使用:
Linuxシステムでは、このプログラムはターミナルを自動的に呼び出して依存関係をインストールすることができません。以下のコマンドを手動で実行する必要があります。それが完了した後に、正常に使用することができます:

sudo apt install python3-dev
chmod +x ./RWKV-Runner
./RWKV-Runner
cd backend-python
pip3 install -r requirements.txt # or pip3 install -r requirements_without_cyac.txt

# See More: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
