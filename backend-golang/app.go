package backend_golang

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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
	Dev           bool
	proxyPort     int
	exDir         string
	cmdPrefix     string
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

func (a *App) newFetchProxy() {
	go func() {
		handler := func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "OPTIONS" {
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "*")
				w.Header().Set("Access-Control-Allow-Origin", "*")
				return
			}
			proxy := &httputil.ReverseProxy{
				ModifyResponse: func(res *http.Response) error {
					res.Header.Set("Access-Control-Allow-Origin", "*")
					return nil
				},
				Director: func(req *http.Request) {
					realTarget := req.Header.Get("Real-Target")
					if realTarget != "" {
						realTarget, err := url.PathUnescape(realTarget)
						if err != nil {
							log.Printf("Error decoding target URL: %v\n", err)
							return
						}
						target, err := url.Parse(realTarget)
						if err != nil {
							log.Printf("Error parsing target URL: %v\n", err)
							return
						}
						req.Header.Set("Accept", "*/*")
						req.Header.Del("Origin")
						req.Header.Del("Referer")
						req.Header.Del("Real-Target")
						req.Header.Del("Sec-Fetch-Dest")
						req.Header.Del("Sec-Fetch-Mode")
						req.Header.Del("Sec-Fetch-Site")
						req.URL.Scheme = target.Scheme
						req.URL.Host = target.Host
						req.URL.Path = target.Path
						req.URL.RawQuery = url.PathEscape(target.RawQuery)
						log.Println("Proxying to", realTarget)
					} else {
						log.Println("Real-Target header is missing")
					}
				},
			}
			proxy.ServeHTTP(w, r)
		}
		http.HandleFunc("/", handler)
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return
		}
		a.proxyPort = listener.Addr().(*net.TCPAddr).Port

		http.Serve(listener, nil)
	}()
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) OnStartup(ctx context.Context) {
	a.ctx = ctx
	a.exDir = ""
	a.cmdPrefix = ""

	ex, err := os.Executable()
	if err == nil {
		if runtime.GOOS == "darwin" {
			a.exDir = filepath.Dir(ex) + "/../../../"
		} else {
			a.exDir = filepath.Dir(ex) + "/"
		}
		a.cmdPrefix = "cd " + a.exDir + " && "
		if a.Dev {
			a.exDir = a.exDir + "../../"
		} else {
			os.Chdir(a.exDir)
		}
	}

	os.Chmod(a.exDir+"backend-rust/webgpu_server", 0777)
	os.Chmod(a.exDir+"backend-rust/web-rwkv-converter", 0777)
	os.Mkdir(a.exDir+"models", os.ModePerm)
	os.Mkdir(a.exDir+"lora-models", os.ModePerm)
	os.Mkdir(a.exDir+"state-models", os.ModePerm)
	os.Mkdir(a.exDir+"finetune/json2binidx_tool/data", os.ModePerm)
	trainLogPath := "lora-models/train_log.txt"
	if !a.FileExists(trainLogPath) {
		f, err := os.Create(a.exDir + trainLogPath)
		if err == nil {
			f.Close()
		}
	}

	a.downloadLoop()
	a.midiLoop()
	a.watchFs()
	a.monitorHardware()
	a.newFetchProxy()
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
		watcher.Add(a.exDir + "./models")
		watcher.Add(a.exDir + "./lora-models")
		watcher.Add(a.exDir + "./state-models")
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

	// update progress
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

	var updateFile io.Reader = pr
	// extract macos binary from zip
	if strings.HasSuffix(url, ".zip") && runtime.GOOS == "darwin" {
		zipBytes, err := io.ReadAll(pr)
		if err != nil {
			return false, err
		}
		archive, err := zip.NewReader(bytes.NewReader(zipBytes), int64(len(zipBytes)))
		if err != nil {
			return false, err
		}
		file, err := archive.Open("RWKV-Runner.app/Contents/MacOS/RWKV-Runner")
		if err != nil {
			return false, err
		}
		defer file.Close()
		updateFile = file
	}

	// apply update
	err = selfupdate.Apply(updateFile, selfupdate.Options{})
	if err != nil {
		if rerr := selfupdate.RollbackError(err); rerr != nil {
			return true, rerr
		}
		return false, err
	}
	// restart app
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

func (a *App) GetProxyPort() int {
	return a.proxyPort
}
