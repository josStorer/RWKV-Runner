package backend_golang

import "os"

var cmds = make(map[string][]string)
var cmdProcesses = make(map[string]*os.Process)

func (a *App) GetCmds() map[string][]string {
	return cmds
}

func (a *App) KillCmd(eventId string) error {
	cmd, ok := cmdProcesses[eventId]
	if !ok {
		return nil
	}
	delete(cmds, eventId)
	delete(cmdProcesses, eventId)
	return cmd.Kill()
}
