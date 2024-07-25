package backend_golang

import (
	"archive/zip"
	"bufio"
	"crypto/sha256"
	"embed"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

func CmdHelper(hideWindow bool, args ...string) (*exec.Cmd, error) {
	if runtime.GOOS != "windows" {
		return nil, errors.New("unsupported OS")
	}
	ex, err := os.Executable()
	if err != nil {
		return nil, err
	}
	exDir := filepath.Dir(ex) + "/"
	path := exDir + "cmd-helper.bat"
	_, err = os.Stat(path)
	if err != nil {
		if err := os.WriteFile(path, []byte("start %*"), 0644); err != nil {
			return nil, err
		}
	}
	cmdHelper, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}

	if strings.Contains(cmdHelper, " ") {
		for _, arg := range args {
			if strings.Contains(arg, " ") {
				return nil, errors.New("path contains space") // golang bug https://github.com/golang/go/issues/17149#issuecomment-473976818
			}
		}
	}
	cmd := exec.Command(cmdHelper, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{}
	//go:custom_build windows cmd.SysProcAttr.HideWindow = hideWindow
	return cmd, nil
}

func Cmd(args ...string) (string, error) {
	switch platform := runtime.GOOS; platform {
	case "windows":
		cmd, err := CmdHelper(true, args...)
		if err != nil {
			return "", err
		}
		_, err = cmd.CombinedOutput()
		if err != nil {
			return "", err
		}
		return "", nil
	case "darwin":
		ex, err := os.Executable()
		if err != nil {
			return "", err
		}
		exDir := filepath.Dir(ex) + "/../../../"
		cmd := exec.Command("osascript", "-e", `tell application "Terminal" to do script "`+"cd "+exDir+" && "+strings.Join(args, " ")+`"`)
		err = cmd.Start()
		if err != nil {
			return "", err
		}
		cmd.Wait()
		return "", nil
	case "linux":
		cmd := exec.Command(args[0], args[1:]...)
		err := cmd.Start()
		if err != nil {
			return "", err
		}
		cmd.Wait()
		return "", nil
	}
	return "", errors.New("unsupported OS")
}

func CopyEmbed(efs embed.FS) error {
	ex, err := os.Executable()
	if err != nil {
		return err
	}
	var prefix string
	if runtime.GOOS == "darwin" {
		prefix = filepath.Dir(ex) + "/../../../"
	} else {
		prefix = filepath.Dir(ex) + "/"
	}

	err = fs.WalkDir(efs, ".", func(path string, d fs.DirEntry, err error) error {
		if d.IsDir() {
			return nil
		}
		if err != nil {
			return err
		}
		content, err := efs.ReadFile(path)
		if err != nil {
			return err
		}

		path = prefix + path
		err = os.MkdirAll(path[:strings.LastIndex(path, "/")], 0755)
		if err != nil {
			return err
		}

		executeWrite := true
		existedContent, err := os.ReadFile(path)
		if err == nil {
			if fmt.Sprintf("%x", sha256.Sum256(existedContent)) == fmt.Sprintf("%x", sha256.Sum256(content)) {
				executeWrite = false
			}
		}

		if executeWrite {
			err = os.WriteFile(path, content, 0644)
			if err != nil {
				return err
			}
		}

		return nil
	})
	return err
}

func GetPython(a *App) (string, error) {
	switch platform := runtime.GOOS; platform {
	case "windows":
		pyexe := a.exDir + "py310/python.exe"
		_, err := os.Stat(pyexe)
		if err != nil {
			_, err := os.Stat(a.exDir + "python-3.10.11-embed-amd64.zip")
			if err != nil {
				return "", errors.New("python zip not found")
			} else {
				err := Unzip(a.exDir+"python-3.10.11-embed-amd64.zip", a.exDir+"py310")
				if err != nil {
					return "", errors.New("failed to unzip python")
				} else {
					return "py310/python.exe", nil
				}
			}
		} else {
			return "py310/python.exe", nil
		}
	case "darwin":
		return "python3", nil
	case "linux":
		return "python3", nil
	}
	return "", errors.New("unsupported OS")
}

func ChangeFileLine(filePath string, lineNumber int, newText string) error {
	file, err := os.OpenFile(filePath, os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	content := make([]string, 0)
	for scanner.Scan() {
		content = append(content, scanner.Text())
	}

	content[lineNumber-1] = newText

	newContent := strings.Join(content, "\n")

	err = file.Truncate(0)
	if err != nil {
		return err
	}
	_, err = file.Seek(0, io.SeekStart)
	if err != nil {
		return err
	}
	_, err = file.WriteString(newContent)
	if err != nil {
		return err
	}
	return nil
}

// https://gist.github.com/paulerickson/6d8650947ee4e3f3dbcc28fde10eaae7
func Unzip(source, destination string) error {
	archive, err := zip.OpenReader(source)
	if err != nil {
		return err
	}
	defer archive.Close()
	for _, file := range archive.Reader.File {
		reader, err := file.Open()
		if err != nil {
			return err
		}
		defer reader.Close()
		path := filepath.Join(destination, file.Name)
		// Remove file if it already exists; no problem if it doesn't; other cases can error out below
		_ = os.Remove(path)
		// Create a directory at path, including parents
		err = os.MkdirAll(path, os.ModePerm)
		if err != nil {
			return err
		}
		// If file is _supposed_ to be a directory, we're done
		if file.FileInfo().IsDir() {
			continue
		}
		// otherwise, remove that directory (_not_ including parents)
		err = os.Remove(path)
		if err != nil {
			return err
		}
		// and create the actual file.  This ensures that the parent directories exist!
		// An archive may have a single file with a nested path, rather than a file for each parent dir
		writer, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, file.Mode())
		if err != nil {
			return err
		}
		defer writer.Close()
		_, err = io.Copy(writer, reader)
		if err != nil {
			return err
		}
	}
	return nil
}

func (a *App) IsPortAvailable(port int) bool {
	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%s", strconv.Itoa(port)))
	if err != nil {
		return false
	}
	defer l.Close()
	return true
}
