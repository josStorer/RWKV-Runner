//go:build darwin || linux

package backend_golang

import (
	"errors"
)

func (a *App) WslStart() error {
	return errors.New("wsl not supported")
}

func (a *App) WslCommand(command string) error {
	return errors.New("wsl not supported")
}

func (a *App) WslStop() error {
	return errors.New("wsl not supported")
}

func (a *App) WslIsEnabled() error {
	return errors.New("wsl not supported")
}

func (a *App) WslEnable(forceMode bool) error {
	return errors.New("wsl not supported")
}

func (a *App) WslInstallUbuntu() error {
	return errors.New("wsl not supported")
}
