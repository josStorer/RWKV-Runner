from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

webui_server = FastAPI()

webui_server.add_middleware(GZipMiddleware, minimum_size=1000)
webui_server.mount(
    "/", StaticFiles(directory="frontend/dist", html=True), name="static"
)

if __name__ == "__main__":
    uvicorn.run("webui_server:webui_server")
