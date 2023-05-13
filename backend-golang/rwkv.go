package backend_golang

import (
	"os/exec"
)

func (a *App) StartServer(strategy string, modelPath string) (string, error) {
	//cmd := exec.Command(`explorer`, `/select,`, `e:\RWKV-4-Raven-7B-v10-Eng49%25-Chn50%25-Other1%25-20230420-ctx4096.pth`)
	cmd := exec.Command("cmd-helper", "python", "./backend-python/main.py", strategy, modelPath)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}
