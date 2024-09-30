//go:build darwin || linux

package backend_golang

import (
	"bufio"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"time"

	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

func (a *App) CmdInteractive(args []string, taskName string) error {
	fmt.Println("🚧✅ CmdInteractive", args, taskName)
	currentTime := time.Now().UnixMilli()
	threadID := taskName + "_" + strconv.FormatInt(currentTime, 10)
	wruntime.EventsEmit(a.ctx, "cmd_event")
	// fmt.Println("0")
	cmd := exec.Command(args[0], args[1:]...)
	// fmt.Println("1")
	stdout, err := cmd.StdoutPipe()
	// fmt.Println("2")
	if err != nil {
		// fmt.Println(err)
		return err
	}
	// fmt.Println("3")
	cmd.Stderr = cmd.Stdout
	// fmt.Println("4")
	err = cmd.Start()
	// fmt.Println("5")
	if err != nil {
		// fmt.Println(err)
		return err
	}
	cmds[threadID] = args
	cmdProcesses[threadID] = cmd.Process
	// fmt.Println("6")
	reader := bufio.NewReader(stdout)
	for {
		// fmt.Println("7")
		line, _, err := reader.ReadLine()
		// fmt.Println("8", line, err)
		if err != nil {
			delete(cmds, threadID)
			delete(cmdProcesses, threadID)
			if err == io.EOF {
				wruntime.EventsEmit(a.ctx, "cmd_event", threadID)
				return nil
			}
			// fmt.Println("9", err)
			return err
		}
		// fmt.Println("10", line)
		strLine := string(line)
		// fmt.Println("11", strLine)
		wruntime.EventsEmit(a.ctx, "cmd_event", threadID, strLine)
		// fmt.Println("12")
	}
}
