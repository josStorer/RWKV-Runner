//go:build windows

package backend_golang

import (
	"bufio"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	su "github.com/nyaosorg/go-windows-su"
	wsl "github.com/ubuntu/gowsl"
	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

var distro *wsl.Distro
var stdin io.WriteCloser
var cmd *exec.Cmd

func isWslRunning() (bool, error) {
	if distro == nil {
		return false, nil
	}
	state, err := distro.State()
	if err != nil {
		return false, err
	}
	if state != wsl.Running {
		distro = nil
		return false, nil
	}
	return true, nil
}

func (a *App) WslStart() error {
	running, err := isWslRunning()
	if err != nil {
		return err
	}
	if running {
		return nil
	}
	distros, err := wsl.RegisteredDistros(context.Background())
	if err != nil {
		return err
	}
	for _, d := range distros {
		if strings.Contains(d.Name(), "Ubuntu") {
			distro = &d
			break
		}
	}
	if distro == nil {
		return errors.New("ubuntu not found")
	}

	cmd = exec.Command("wsl", "-d", distro.Name(), "-u", "root")

	stdin, err = cmd.StdinPipe()
	if err != nil {
		return err
	}

	stdout, err := cmd.StdoutPipe()
	cmd.Stderr = cmd.Stdout
	if err != nil {
		// stdin.Close()
		stdin = nil
		return err
	}

	go func() {
		reader := bufio.NewReader(stdout)
		for {
			if stdin == nil {
				break
			}
			line, _, err := reader.ReadLine()
			if err != nil {
				wruntime.EventsEmit(a.ctx, "wslerr", err.Error())
				break
			}
			wruntime.EventsEmit(a.ctx, "wsl", string(line))
		}
		// stdout.Close()
	}()

	if err := cmd.Start(); err != nil {
		return err
	}
	return nil
}

func (a *App) WslCommand(command string) error {
	running, err := isWslRunning()
	if err != nil {
		return err
	}
	if !running {
		return errors.New("wsl not running")
	}
	_, err = stdin.Write([]byte(command + "\n"))
	if err != nil {
		return err
	}
	return nil
}

func (a *App) WslStop() error {
	running, err := isWslRunning()
	if err != nil {
		return err
	}
	if !running {
		return errors.New("wsl not running")
	}
	err = cmd.Process.Kill()
	cmd = nil
	// stdin.Close()
	stdin = nil
	distro = nil
	if err != nil {
		return err
	}
	return nil
}

func (a *App) WslIsEnabled() error {
	ex, err := os.Executable()
	if err != nil {
		return err
	}
	exDir := filepath.Dir(ex)

	data, err := os.ReadFile(exDir + "/wsl.state")
	if err == nil {
		if strings.Contains(string(data), "Enabled") {
			return nil
		}
	}

	cmd := `-Command (Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux).State | Out-File -Encoding utf8 -FilePath ` + exDir + "/wsl.state"
	_, err = su.ShellExecute(su.RUNAS, "powershell", cmd, exDir)
	if err != nil {
		return err
	}
	time.Sleep(2 * time.Second)
	data, err = os.ReadFile(exDir + "/wsl.state")
	if err != nil {
		return err
	}
	if strings.Contains(string(data), "Enabled") {
		return nil
	} else {
		return errors.New("wsl is not enabled")
	}
}

func (a *App) WslEnable(forceMode bool) error {
	cmd := `/online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux`
	_, err := su.ShellExecute(su.RUNAS, "dism", cmd, `C:\`)
	if err != nil {
		return err
	}
	if forceMode {
		os.WriteFile("./wsl.state", []byte("Enabled"), 0644)
	}
	return nil
}

func (a *App) WslInstallUbuntu() error {
	_, err := Cmd("ms-windows-store://pdp/?ProductId=9PN20MSR04DW")
	return err
}
