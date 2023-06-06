Please execute this program in an empty directory. All related dependencies will be placed in this directory.
请将本程序放在一个空目录内执行, 所有相关依赖均会放置于此目录.
このプログラムを空のディレクトリで実行してください。関連するすべての依存関係は、このディレクトリに配置されます。

Please execute the following command in the terminal to remove the permission restrictions of this app, and then this program can work properly:
请在终端执行以下命令解除本app的权限限制, 然后本程序才可以正常工作:
このアプリの権限制限を解除するために、ターミナルで以下のコマンドを実行してください。その後、このプログラムは正常に動作するようになります:

sudo xattr -r -d com.apple.quarantine ./RWKV-Runner.app
