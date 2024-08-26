//go:build windows

package backend_golang

import (
	"bufio"
	"bytes"
	"io"
	"os/exec"
	"syscall"

	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/transform"
)

func (a *App) CmdInteractive(args []string, eventId string) error {
	cmd := exec.Command(args[0], args[1:]...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = cmd.Stdout

	cmd.SysProcAttr = &syscall.SysProcAttr{}
	cmd.SysProcAttr.HideWindow = true
	err = cmd.Start()
	if err != nil {
		return err
	}
	cmds[eventId] = args
	cmdProcesses[eventId] = cmd.Process

	reader := bufio.NewReader(stdout)
	for {
		line, _, err := reader.ReadLine()
		if err != nil {
			delete(cmds, eventId)
			delete(cmdProcesses, eventId)
			if err == io.EOF {
				wruntime.EventsEmit(a.ctx, eventId+"-finish")
				return nil
			}
			return err
		}
		reader := transform.NewReader(bytes.NewReader(line), simplifiedchinese.GBK.NewDecoder())
		line2, err := io.ReadAll(reader)
		if err == nil {
			line = line2
		}
		strLine := string(line)
		wruntime.EventsEmit(a.ctx, eventId+"-output", strLine)
	}
}
