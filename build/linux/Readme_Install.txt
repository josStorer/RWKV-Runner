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
