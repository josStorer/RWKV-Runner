import os


def ngrok_connect():
    from pyngrok import ngrok, conf
    conf.set_default(conf.PyngrokConfig(ngrok_path="./ngrok"))
    ngrok.set_auth_token(os.environ["ngrok_token"])
    http_tunnel = ngrok.connect(8000)
    print(http_tunnel.public_url)
