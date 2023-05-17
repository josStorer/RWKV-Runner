package backend_golang

import (
	"os/exec"
	"path/filepath"
	"strconv"
)

func (a *App) StartServer(port int) (string, error) {
	cmdHelper, err := filepath.Abs("./cmd-helper")
	if err != nil {
		return "", err
	}
	cmd := exec.Command(cmdHelper, "python", "./backend-python/main.py", strconv.Itoa(port))
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func (a *App) ConvertModel(modelPath string, strategy string, outPath string) (string, error) {
	cmdHelper, err := filepath.Abs("./cmd-helper")
	if err != nil {
		return "", err
	}
	cmd := exec.Command(cmdHelper, "python", "./backend-python/convert_model.py", "--in", modelPath, "--out", outPath, "--strategy", strategy)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}
