import os
import global_var


def ngrok_connect():
    from pyngrok import ngrok, conf

    conf.set_default(
        conf.PyngrokConfig(ngrok_path="./ngrok.exe" if os.name == "nt" else "./ngrok")
    )
    ngrok.set_auth_token(os.environ["ngrok_token"])
    http_tunnel = ngrok.connect(global_var.get(global_var.Args).port)
    print(f"ngrok url: {http_tunnel.public_url}")
