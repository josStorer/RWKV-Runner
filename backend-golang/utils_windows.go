//go:build windows

package backend_golang

import (
	"os/exec"
	"strings"
	"syscall"
)

func CmdSetHideWindow(cmd *exec.Cmd, hideWindow bool) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = hideWindow
}

func (a *App) CommandOutput(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	CmdSetHideWindow(cmd, true)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}
