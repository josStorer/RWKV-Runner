package backend_golang

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/minio/selfupdate"
	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

// App struct
type App struct {
	ctx           context.Context
	HasConfigData bool
	ConfigData    map[string]any
	exDir         string
	cmdPrefix     string
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) OnStartup(ctx context.Context) {
	a.ctx = ctx
	a.exDir = ""
	a.cmdPrefix = ""

	if runtime.GOOS == "darwin" {
		ex, _ := os.Executable()
		a.exDir = filepath.Dir(ex) + "/../../../"
		a.cmdPrefix = "cd " + a.exDir + " && "
	}

	os.Chmod(a.exDir+"backend-rust/webgpu_server", 0777)
	os.Chmod(a.exDir+"backend-rust/web-rwkv-converter", 0777)
	os.Mkdir(a.exDir+"models", os.ModePerm)
	os.Mkdir(a.exDir+"lora-models", os.ModePerm)
	os.Mkdir(a.exDir+"finetune/json2binidx_tool/data", os.ModePerm)
	trainLogPath := a.exDir + "lora-models/train_log.txt"
	if !a.FileExists(trainLogPath) {
		f, err := os.Create(trainLogPath)
		if err == nil {
			f.Close()
		}
	}

	a.downloadLoop()
	a.midiLoop()
	a.watchFs()
	a.monitorHardware()
}

func (a *App) OnBeforeClose(ctx context.Context) bool {
	if monitor != nil {
		monitor.Process.Kill()
	}
	return false
}

func (a *App) watchFs() {
	watcher, err := fsnotify.NewWatcher()
	if err == nil {
		watcher.Add(a.exDir + "./lora-models")
		watcher.Add(a.exDir + "./models")
		go func() {
			for {
				select {
				case event, ok := <-watcher.Events:
					if !ok {
						return
					}
					wruntime.EventsEmit(a.ctx, "fsnotify", event.Name)
				case _, ok := <-watcher.Errors:
					if !ok {
						return
					}
				}
			}
		}()
	}
}

var monitor *exec.Cmd

func (a *App) monitorHardware() {
	if runtime.GOOS != "windows" {
		return
	}

	monitor = exec.Command("./components/LibreHardwareMonitor.Console/LibreHardwareMonitor.Console.exe")
	stdout, err := monitor.StdoutPipe()
	if err != nil {
		monitor = nil
		return
	}

	go func() {
		reader := bufio.NewReader(stdout)
		for {
			line, _, err := reader.ReadLine()
			if err != nil {
				wruntime.EventsEmit(a.ctx, "monitorerr", err.Error())
				break
			}
			wruntime.EventsEmit(a.ctx, "monitor", string(line))
		}
	}()

	monitor.SysProcAttr = &syscall.SysProcAttr{}
	//go:custom_build windows monitor.SysProcAttr.HideWindow = true
	monitor.Start()
}

type ProgressReader struct {
	reader io.Reader
	total  int64
	err    error
}

func (pr *ProgressReader) Read(p []byte) (n int, err error) {
	n, err = pr.reader.Read(p)
	pr.err = err
	pr.total += int64(n)
	return
}

func (a *App) UpdateApp(url string) (broken bool, err error) {
	resp, err := http.Get(url)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	pr := &ProgressReader{reader: resp.Body}

	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()

	go func() {
		for {
			<-ticker.C
			wruntime.EventsEmit(a.ctx, "updateApp", &DownloadStatus{
				Name:        filepath.Base(url),
				Path:        "",
				Url:         url,
				Transferred: pr.total,
				Size:        resp.ContentLength,
				Speed:       0,
				Progress:    100 * (float64(pr.total) / float64(resp.ContentLength)),
				Downloading: pr.err == nil && pr.total < resp.ContentLength,
				Done:        pr.total == resp.ContentLength,
			})
			if pr.err != nil || pr.total == resp.ContentLength {
				break
			}
		}
	}()
	err = selfupdate.Apply(pr, selfupdate.Options{})
	if err != nil {
		if rerr := selfupdate.RollbackError(err); rerr != nil {
			return true, rerr
		}
		return false, err
	}
	if runtime.GOOS == "windows" {
		name, err := os.Executable()
		if err != nil {
			return false, err
		}
		exec.Command(name, os.Args[1:]...).Start()
		wruntime.Quit(a.ctx)
	}
	return false, nil
}

func (a *App) RestartApp() error {
	if runtime.GOOS == "windows" {
		name, err := os.Executable()
		if err != nil {
			return err
		}
		exec.Command(name, os.Args[1:]...).Start()
		wruntime.Quit(a.ctx)
		return nil
	}
	return errors.New("unsupported OS")
}

func (a *App) GetPlatform() string {
	return runtime.GOOS
}
