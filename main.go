package main

import (
	"embed"
	"fmt"
	"net/http"
	"os"
	"runtime/debug"
	"strings"

	backend "rwkv-runner/backend-golang"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options/windows"
)

type FileLoader struct {
	http.Handler
}

func NewFileLoader() *FileLoader {
	return &FileLoader{}
}

func (h *FileLoader) ServeHTTP(res http.ResponseWriter, req *http.Request) {
	var err error
	requestedFilename := strings.TrimPrefix(req.URL.Path, "/")
	println("Requesting file:", requestedFilename)
	fileData, err := os.ReadFile(requestedFilename)
	if err != nil {
		res.WriteHeader(http.StatusBadRequest)
		res.Write([]byte(fmt.Sprintf("Could not load file %s", requestedFilename)))
	}

	res.Write(fileData)
}

//go:embed all:frontend/dist
var assets embed.FS

//go:embed all:py310/Lib/site-packages/cyac
var cyac embed.FS

//go:embed all:py310/Lib/site-packages/cyac-1.7.dist-info
var cyacInfo embed.FS

//go:embed backend-python
var py embed.FS

//go:embed backend-rust
var webgpu embed.FS

//go:embed finetune
var finetune embed.FS

//go:embed midi
var midi embed.FS

//go:embed assets/sound-font
var midiAssets embed.FS

func main() {
	if buildInfo, ok := debug.ReadBuildInfo(); !ok || strings.Contains(buildInfo.String(), "-ldflags") {
		backend.CopyEmbed(cyac)
		backend.CopyEmbed(cyacInfo)
		backend.CopyEmbed(py)
		backend.CopyEmbed(webgpu)
		backend.CopyEmbed(finetune)
		backend.CopyEmbed(midi)
		backend.CopyEmbed(midiAssets)
	}

	// Create an instance of the app structure
	app := backend.NewApp()

	var zoomFactor float64 = 1.0
	data, err := app.ReadJson("config.json")
	if err == nil {
		app.HasConfigData = true
		app.ConfigData = data.(map[string]any)
		if dpiScaling, ok := app.ConfigData["settings"].(map[string]any)["dpiScaling"]; ok {
			zoomFactor = dpiScaling.(float64) / 100
		}
	} else {
		app.HasConfigData = false
	}

	// Create application with options
	err = wails.Run(&options.App{
		Title:     "RWKV-Runner",
		Width:     1024,
		Height:    680,
		MinWidth:  375,
		MinHeight: 640,
		Windows: &windows.Options{
			ZoomFactor:           zoomFactor,
			IsZoomControlEnabled: true,
		},
		AssetServer: &assetserver.Options{
			Assets:  assets,
			Handler: NewFileLoader(),
		},
		OnStartup: app.OnStartup,
		Bind: []any{
			app,
		},
	})

	if err != nil {
		println("Error:", err.Error())
	}
}
