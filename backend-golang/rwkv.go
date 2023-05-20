package backend_golang

import (
	"errors"
	"os/exec"
	"strconv"
)

func (a *App) StartServer(port int) (string, error) {
	python, err := GetPython()
	if err != nil {
		return "", err
	}
	return Cmd(python, "./backend-python/main.py", strconv.Itoa(port))
}

func (a *App) ConvertModel(modelPath string, strategy string, outPath string) (string, error) {
	python, err := GetPython()
	if err != nil {
		return "", err
	}
	return Cmd(python, "./backend-python/convert_model.py", "--in", modelPath, "--out", outPath, "--strategy", strategy)
}

func (a *App) DepCheck() error {
	python, err := GetPython()
	if err != nil {
		return err
	}
	out, err := exec.Command(python, "./backend-python/dep_check.py").CombinedOutput()
	if err != nil {
		return errors.New("DepCheck Error: " + string(out))
	}
	return nil
}

func (a *App) InstallPyDep() (string, error) {
	python, err := GetPython()
	if err != nil {
		return "", err
	}
	_, err = Cmd(python, "./backend-python/get-pip.py")
	if err != nil {
		return "", err
	}
	ChangeFileLine("./py310/python310._pth", 3, "Lib\\site-packages")
	_, err = Cmd(python, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu117")
	if err != nil {
		return "", err
	}
	return Cmd(python, "-m", "pip", "install", "-r", "./backend-python/requirements_versions.txt")
}
