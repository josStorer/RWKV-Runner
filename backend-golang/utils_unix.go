//go:build darwin || linux

package backend_golang

import (
	"os/exec"
	"strings"
)

func CmdSetHideWindow(cmd *exec.Cmd, hideWindow bool) {
}

func (a *App) CommandOutput(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}
