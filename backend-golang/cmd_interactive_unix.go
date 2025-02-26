//go:build darwin || linux

package backend_golang

import (
	"bufio"
	"io"
	"os/exec"
	"strconv"
	"time"

	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

func (a *App) CmdInteractive(args []string, taskName string) error {
	currentTime := time.Now().UnixMilli()
	threadID := taskName + "_" + strconv.FormatInt(currentTime, 10)

	wruntime.EventsEmit(a.ctx, "cmd_event")

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

	cmds[threadID] = args
	cmdProcesses[threadID] = cmd.Process
	reader := bufio.NewReader(stdout)

	for {
		line, _, err := reader.ReadLine()
		if err != nil {
			delete(cmds, threadID)
			delete(cmdProcesses, threadID)
			if err == io.EOF {
				wruntime.EventsEmit(a.ctx, "cmd_event", threadID)
				return nil
			}
			return err
		}
		strLine := string(line)
		wruntime.EventsEmit(a.ctx, "cmd_event", threadID, strLine)
	}
}
