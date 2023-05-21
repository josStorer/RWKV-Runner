package backend_golang

import (
	"archive/zip"
	"bufio"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

func Cmd(args ...string) (string, error) {
	_, err := os.Stat("cmd-helper.bat")
	if err != nil {
		if err := os.WriteFile("./cmd-helper.bat", []byte("start %*"), 0644); err != nil {
			return "", err
		}
	}
	cmdHelper, err := filepath.Abs("./cmd-helper")
	if err != nil {
		return "", err
	}
	cmd := exec.Command(cmdHelper, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func GetPython() (string, error) {
	switch platform := runtime.GOOS; platform {
	case "windows":
		_, err := os.Stat("py310/python.exe")
		if err != nil {
			_, err := os.Stat("python-3.10.11-embed-amd64.zip")
			if err != nil {
				return "", errors.New("python zip not found")
			} else {
				err := Unzip("python-3.10.11-embed-amd64.zip", "py310")
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
