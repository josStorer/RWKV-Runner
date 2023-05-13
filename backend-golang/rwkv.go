package backend_golang

import (
	"os/exec"
)

func (a *App) StartServer(strategy string, modelPath string) (string, error) {
	cmd := exec.Command("cmd-helper", "python", "./backend-python/main.py", strategy, modelPath)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}
