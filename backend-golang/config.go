package backend_golang

import (
	"encoding/json"
	"os"
)

func (a *App) SaveConfig(config interface{}) string {
	jsonData, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err.Error()
	}

	if err := os.WriteFile("config.json", jsonData, 0644); err != nil {
		return err.Error()
	}
	return ""
}
