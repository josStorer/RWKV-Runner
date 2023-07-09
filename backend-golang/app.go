package backend_golang

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

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

	os.Mkdir(a.exDir+"models", os.ModePerm)
	os.Mkdir(a.exDir+"lora-models", os.ModePerm)
	os.Mkdir(a.exDir+"finetune/json2binidx_tool/data", os.ModePerm)
	f, err := os.Create(a.exDir + "lora-models/train_log.txt")
	if err == nil {
		f.Close()
	}

	a.downloadLoop()

	watcher, err := fsnotify.NewWatcher()
	if err == nil {
		watcher.Add("./lora-models")
		watcher.Add("./models")
		go func() {
			for {
				select {
				case event, ok := <-watcher.Events:
					if !ok {
						return
					}
					wruntime.EventsEmit(ctx, "fsnotify", event.Name)
				case _, ok := <-watcher.Errors:
					if !ok {
						return
					}
				}
			}
		}()
	}
}

func (a *App) UpdateApp(url string) (broken bool, err error) {
	resp, err := http.Get(url)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	err = selfupdate.Apply(resp.Body, selfupdate.Options{})
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
