package backend_golang

import (
	"crypto/md5"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

func (a *App) GetAbsPath(path string) (string, error) {
	var absPath string
	var err error
	if filepath.IsAbs(path) {
		absPath = filepath.Clean(path)
	} else {
		absPath, err = filepath.Abs(filepath.Join(a.exDir, path))
		if err != nil {
			return "", err
		}
	}
	absPath = strings.ReplaceAll(absPath, "/", string(os.PathSeparator))
	println("GetAbsPath:", absPath)
	return absPath, nil
}

func (a *App) SaveFile(path string, savedContent []byte) error {
	absPath, err := a.GetAbsPath(path)
	if err != nil {
		return err
	}
	if err := os.WriteFile(absPath, savedContent, 0644); err != nil {
		return err
	}
	return nil
}

func (a *App) SaveJson(path string, jsonData any) error {
	text, err := json.MarshalIndent(jsonData, "", "  ")
	if err != nil {
		return err
	}

	absPath, err := a.GetAbsPath(path)
	if err != nil {
		return err
	}
	if err := os.WriteFile(absPath, text, 0644); err != nil {
		return err
	}
	return nil
}

func (a *App) ReadJson(path string) (any, error) {
	absPath, err := a.GetAbsPath(path)
	if err != nil {
		return nil, err
	}
	file, err := os.ReadFile(absPath)
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

func (a *App) FileExists(path string) bool {
	absPath, err := a.GetAbsPath(path)
	if err != nil {
		return false
	}
	_, err = os.Stat(absPath)
	return err == nil
}

type FileInfo struct {
	Name    string `json:"name"`
	Size    int64  `json:"size"`
	IsDir   bool   `json:"isDir"`
	ModTime string `json:"modTime"`
}

func (a *App) ReadFileInfo(path string) (*FileInfo, error) {
	absPath, err := a.GetAbsPath(path)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(absPath)
	if err != nil {
		return nil, err
	}
	return &FileInfo{
		Name:    info.Name(),
		Size:    info.Size(),
		IsDir:   info.IsDir(),
		ModTime: info.ModTime().Format(time.RFC3339),
	}, nil
}

func (a *App) ListDirFiles(dirPath string) ([]FileInfo, error) {
	absDirPath, err := a.GetAbsPath(dirPath)
	if err != nil {
		return nil, err
	}
	files, err := os.ReadDir(absDirPath)
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
	absPath, err := a.GetAbsPath(path)
	if err != nil {
		return err
	}
	err = os.Remove(absPath)
	if err != nil {
		return err
	}
	return nil
}

func (a *App) CopyFile(src string, dst string, checkHash bool) error {
	absSrc, err := a.GetAbsPath(src)
	if err != nil {
		return err
	}
	absDst, err := a.GetAbsPath(dst)
	if err != nil {
		return err
	}

	if checkHash {
		if a.FileExists(absDst) {
			srcHash, err := a.GetFileCrc32(absSrc)
			if err != nil {
				return err
			}
			dstHash, err := a.GetFileCrc32(absDst)
			if err != nil {
				return err
			}
			if srcHash == dstHash {
				return nil
			}
		}
	}

	sourceFile, err := os.Open(absSrc)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	err = os.MkdirAll(filepath.Dir(absDst), 0755)
	if err != nil {
		return err
	}

	destFile, err := os.Create(absDst)
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

func (a *App) CopyFolderFiles(src string, dst string, checkHash bool) error {
	absSrc, err := a.GetAbsPath(src)
	if err != nil {
		return err
	}
	absDst, err := a.GetAbsPath(dst)
	if err != nil {
		return err
	}

	files, err := os.ReadDir(absSrc)
	if err != nil {
		return err
	}

	for _, file := range files {
		srcPath := filepath.Join(absSrc, file.Name())
		dstPath := filepath.Join(absDst, file.Name())
		err = a.CopyFile(srcPath, dstPath, checkHash)
		if err != nil {
			return err
		}
	}
	return nil
}

func (a *App) GetFileSha256(path string) (string, error) {
	absPath, err := a.GetAbsPath(path)
	if err != nil {
		return "", err
	}
	content, err := os.ReadFile(absPath)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", sha256.Sum256(content)), nil
}

func (a *App) GetFileMd5(path string) (string, error) {
	absPath, err := a.GetAbsPath(path)
	if err != nil {
		return "", err
	}
	content, err := os.ReadFile(absPath)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", md5.Sum(content)), nil
}

func (a *App) GetFileCrc32(path string) (string, error) {
	absPath, err := a.GetAbsPath(path)
	if err != nil {
		return "", err
	}
	content, err := os.ReadFile(absPath)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", crc32.Checksum(content, crc32.IEEETable)), nil
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

// Only return the path of the selected file, because communication between frontend and backend is slow. Use AssetServer Handler to read the file.
func (a *App) OpenOpenFileDialog(filterPattern string) (string, error) {
	path, err := wruntime.OpenFileDialog(a.ctx, wruntime.OpenDialogOptions{
		Filters: []wruntime.FileFilter{{Pattern: filterPattern}},
	})
	if err != nil {
		return "", err
	}
	if path == "" {
		return "", nil
	}
	return path, nil
}

func (a *App) OpenFileFolder(path string) error {
	absPath, err := a.GetAbsPath(path)
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

func (a *App) StartFile(path string) error {
	cmd, err := CmdHelper(true, path)
	if err != nil {
		return err
	}
	err = cmd.Start()
	return err
}
