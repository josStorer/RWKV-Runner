package backend_golang

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

func (a *App) SaveJson(fileName string, jsonData any) error {
	text, err := json.MarshalIndent(jsonData, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(a.exDir+fileName, text, 0644); err != nil {
		return err
	}
	return nil
}

func (a *App) ReadJson(fileName string) (any, error) {
	file, err := os.ReadFile(a.exDir + fileName)
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
	_, err := os.Stat(a.exDir + fileName)
	return err == nil
}

type FileInfo struct {
	Name    string `json:"name"`
	Size    int64  `json:"size"`
	IsDir   bool   `json:"isDir"`
	ModTime string `json:"modTime"`
}

func (a *App) ReadFileInfo(fileName string) (FileInfo, error) {
	info, err := os.Stat(a.exDir + fileName)
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
	files, err := os.ReadDir(a.exDir + dirPath)
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
	err := os.Remove(a.exDir + path)
	if err != nil {
		return err
	}
	return nil
}

func (a *App) CopyFile(src string, dst string) error {
	sourceFile, err := os.Open(a.exDir + src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	err = os.MkdirAll(a.exDir+dst[:strings.LastIndex(dst, "/")], 0755)
	if err != nil {
		return err
	}

	destFile, err := os.Create(a.exDir + dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return err
	}
	return nil
}

func (a *App) OpenFileFolder(path string) error {
	absPath, err := filepath.Abs(a.exDir + path)
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
		return nil
	case "darwin":
		cmd := exec.Command("open", "-R", absPath)
		err := cmd.Run()
		if err != nil {
			return err
		}
		return nil
	case "linux":
		cmd := exec.Command("xdg-open", absPath)
		err := cmd.Run()
		if err != nil {
			return err
		}
		return nil
	}
	return errors.New("unsupported OS")
}
