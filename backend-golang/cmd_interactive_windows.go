//go:build windows

package backend_golang

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os/exec"
	"syscall"

	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/transform"
)

func (a *App) CmdInteractive(args []string, eventId string) error {
	fmt.Println("1")
	cmd := exec.Command(args[0], args[1:]...)
	fmt.Println("2")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fmt.Println("3")
		return err
	}
	fmt.Println("4")
	cmd.Stderr = cmd.Stdout
	fmt.Println("5")
	cmd.SysProcAttr = &syscall.SysProcAttr{}
	cmd.SysProcAttr.HideWindow = true
	err = cmd.Start()
	if err != nil {
		fmt.Println(err)
		return err
	}
	fmt.Println("7")
	cmds[eventId] = args
	cmdProcesses[eventId] = cmd.Process
	fmt.Println("8")
	reader := bufio.NewReader(stdout)
	fmt.Println("9")
	for {
		line, _, err := reader.ReadLine()
		fmt.Println("10")
		if err != nil {
			fmt.Println("11")
			delete(cmds, eventId)
			delete(cmdProcesses, eventId)
			if err == io.EOF {
				fmt.Println("12")
				wruntime.EventsEmit(a.ctx, eventId+"-finish")
				return nil
			}
			return err
		}
		fmt.Println("13")
		reader := transform.NewReader(bytes.NewReader(line), simplifiedchinese.GBK.NewDecoder())
		fmt.Println("14")
		line2, err := io.ReadAll(reader)
		if err == nil {
			line = line2
		}
		fmt.Println("15")
		strLine := string(line)
		fmt.Println("16")
		wruntime.EventsEmit(a.ctx, eventId+"-output", strLine)
	}
}
