package backend_golang

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
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

func (a *App) FileExists(fileName string) bool {
	_, err := os.Stat(fileName)
	return err == nil
}

type FileInfo struct {
	Name    string `json:"name"`
	Size    int64  `json:"size"`
	IsDir   bool   `json:"isDir"`
	ModTime string `json:"modTime"`
}

func (a *App) ReadFileInfo(fileName string) (FileInfo, error) {
	info, err := os.Stat(fileName)
	if err != nil {
		return FileInfo{}, err
	}
	return FileInfo{
		Name:    info.Name(),
		Size:    info.Size(),
		IsDir:   info.IsDir(),
		ModTime: info.ModTime().Format(time.RFC3339),
	}, nil
}

func (a *App) ListDirFiles(dirPath string) ([]FileInfo, error) {
	files, err := os.ReadDir(dirPath)
	if err != nil {
		return nil, err
	}

	var filesInfo []FileInfo
	for _, file := range files {
		info, err := file.Info()
		if err != nil {
			return nil, err
		}
		filesInfo = append(filesInfo, FileInfo{
			Name:    info.Name(),
			Size:    info.Size(),
			IsDir:   info.IsDir(),
			ModTime: info.ModTime().Format(time.RFC3339),
		})
	}
	return filesInfo, nil
}

func (a *App) DeleteFile(path string) error {
	err := os.Remove(path)
	if err != nil {
		return err
	}
	return nil
}

func (a *App) OpenFileFolder(path string) error {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	switch os := runtime.GOOS; os {
	case "windows":
		cmd := exec.Command("explorer", "/select,", absPath)
		err := cmd.Run()
		if err != nil {
			return err
		}
	case "darwin":
		fmt.Println("Running on macOS")
	case "linux":
		fmt.Println("Running on Linux")
	}
	return nil
}
