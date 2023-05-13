package backend_golang

import (
	"encoding/json"
	"os"

	"github.com/cavaliergopher/grab/v3"
)

func (a *App) SaveJson(fileName string, jsonData any) error {
	text, err := json.MarshalIndent(jsonData, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(fileName, text, 0644); err != nil {
		return err
	}
	return nil
}

func (a *App) ReadJson(fileName string) (any, error) {
	file, err := os.ReadFile(fileName)
	if err != nil {
		return nil, err
	}

	var data any
	err = json.Unmarshal(file, &data)
	if err != nil {
		return nil, err
	}

	return data, nil
}

func (a *App) FileExists(fileName string) (bool, error) {
	_, err := os.Stat(fileName)
	if err == nil {
		return true, nil
	}
	return false, err
}

func (a *App) FileInfo(fileName string) (any, error) {
	info, err := os.Stat(fileName)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"name":  info.Name(),
		"size":  info.Size(),
		"isDir": info.IsDir(),
	}, nil
}

func (a *App) DownloadFile(path string, url string) error {
	_, err := grab.Get(path, url)
	if err != nil {
		return err
	}
	return nil
}
