// Considering some whitespace and multilingual support, the functions in rwkv.go should always be executed with cwd as RWKV-Runner, and never use a.GetAbsPath() here.
package backend_golang

import (
	"errors"
	"os"
	"os/exec"
)

func (a *App) DepCheck(python string) error {
	var err error
	if python == "" {
		python, err = a.GetPython()
	}
	if err != nil {
		return err
	}
	out, err := exec.Command(python, a.exDir+"backend-python/dep_check.py").CombinedOutput()
	if err != nil {
		return errors.New("DepCheck Error: " + string(out) + " GError: " + err.Error())
	}
	return nil
}

func (a *App) GetPyError() string {
	content, err := os.ReadFile("./error.txt")
	if err != nil {
		return ""
	}
	return string(content)
}
