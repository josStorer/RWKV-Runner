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

	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
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

func (a *App) OpenSaveFileDialog(filterPattern string, defaultFileName string, savedContent string) (string, error) {
	return a.OpenSaveFileDialogBytes(filterPattern, defaultFileName, []byte(savedContent))
}

func (a *App) OpenSaveFileDialogBytes(filterPattern string, defaultFileName string, savedContent []byte) (string, error) {
	path, err := wruntime.SaveFileDialog(a.ctx, wruntime.SaveDialogOptions{
		DefaultFilename: defaultFileName,
		Filters: []wruntime.FileFilter{{
			Pattern: filterPattern,
		}},
		CanCreateDirectories: true,
	})
	if err != nil {
		return "", err
	}
	if path == "" {
		return "", nil
	}
	if err := os.WriteFile(path, savedContent, 0644); err != nil {
		return "", err
	}
	return path, nil
}

func (a *App) OpenFileFolder(path string, relative bool) error {
	var absPath string
	var err error
	if relative {
		absPath, err = filepath.Abs(a.exDir + path)
	} else {
		absPath, err = filepath.Abs(path)
	}
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
