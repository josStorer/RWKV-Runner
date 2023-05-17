package backend_golang

import (
	"os/exec"
	"path/filepath"
	"strconv"
)

func cmd(args ...string) (string, error) {
	cmdHelper, err := filepath.Abs("./cmd-helper")
	if err != nil {
		return "", err
	}
	cmd := exec.Command(cmdHelper, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func (a *App) StartServer(port int) (string, error) {
	return cmd("python", "./backend-python/main.py", strconv.Itoa(port))
}

func (a *App) ConvertModel(modelPath string, strategy string, outPath string) (string, error) {
	return cmd("python", "./backend-python/convert_model.py", "--in", modelPath, "--out", outPath, "--strategy", strategy)
}
