import os
import sysconfig


def set_torch():
    torch_path = os.path.join(sysconfig.get_paths()["purelib"], f"torch{os.sep}lib")
    paths = os.environ.get("PATH", "")
    if os.path.exists(torch_path):
        print(f"torch found: {torch_path}")
        if torch_path in paths:
            print("torch already set")
        else:
            os.environ["PATH"] = paths + os.pathsep + torch_path + os.pathsep
            print("torch set")
            # print("run:")
            # print(f"set Path={paths + os.pathsep + torch_path + os.pathsep}")
    else:
        print("torch not found")


def torch_gc():
    try:
        import torch

        if torch.cuda.is_available():
            with torch.cuda.device(0):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    except:
        pass  # prevent 'torch' has no attribute 'cuda' error, so user can use CPU or WebGPU
