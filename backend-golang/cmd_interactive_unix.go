//go:build darwin || linux

package backend_golang

import (
	"bufio"
	"io"
	"os/exec"

	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

func (a *App) CmdInteractive(args []string, uuid string) error {
	cmd := exec.Command(args[0], args[1:]...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = cmd.Stdout

	err = cmd.Start()
	if err != nil {
		return err
	}
	unregisterStop := wruntime.EventsOnce(a.ctx, uuid+"-stop", func(optionalData ...any) {
		cmd.Process.Kill()
	})
	defer unregisterStop()

	reader := bufio.NewReader(stdout)
	for {
		line, _, err := reader.ReadLine()
		if err != nil {
			if err == io.EOF {
				wruntime.EventsEmit(a.ctx, uuid+"-finish")
				return nil
			}
			return err
		}
		strLine := string(line)
		wruntime.EventsEmit(a.ctx, uuid+"-output", strLine)
	}
}
