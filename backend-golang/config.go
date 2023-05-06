package backend_golang

import (
	"encoding/json"
	"os"
)

func (a *App) SaveJson(fileName string, jsonData interface{}) string {
	text, err := json.MarshalIndent(jsonData, "", "  ")
	if err != nil {
		return err.Error()
	}

	if err := os.WriteFile(fileName, text, 0644); err != nil {
		return err.Error()
	}
	return ""
}
