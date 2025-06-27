package main

import (
	"embed"
	"fmt"
	"net/http"
	"os"
	"strings"

	backend "rwkv-runner/backend-golang"

	"github.com/wailsapp/wails/v2"
	wailsLogger "github.com/wailsapp/wails/v2/pkg/logger"
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
	requestedFilename = strings.TrimPrefix(requestedFilename, "=>") // absolute path
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

//go:embed all:py310/Lib/site-packages/cyac-1.9.dist-info
var cyacInfo embed.FS

//go:embed all:py310/Lib/site-packages/llama_cpp
var llamacpp embed.FS

//go:embed all:py310/Lib/site-packages/llama_cpp_python-0.3.9.dist-info
var llamacppInfo embed.FS

//go:embed all:py310/Lib/site-packages/diskcache
var diskcache embed.FS

//go:embed all:py310/Lib/site-packages/diskcache-5.6.3.dist-info
var diskcacheInfo embed.FS

//go:embed backend-python
var py embed.FS

// //go:embed backend-rust
// var webgpu embed.FS

//go:embed all:finetune
var finetune embed.FS

//go:embed midi
var midi embed.FS

//go:embed assets/sound-font
var midiAssets embed.FS

//go:embed components
var components embed.FS

func main() {
	// Create an instance of the app structure
	app := backend.NewApp()
	app.Dev = true

	if !backend.IsDebugMode() {
		app.Dev = false

		backend.CopyEmbed(assets)
		os.RemoveAll("./py310/Lib/site-packages/cyac-1.7.dist-info")
		backend.CopyEmbed(cyac)
		backend.CopyEmbed(cyacInfo)
		backend.CopyEmbed(llamacpp)
		backend.CopyEmbed(llamacppInfo)
		backend.CopyEmbed(diskcache)
		backend.CopyEmbed(diskcacheInfo)
		backend.CopyEmbed(py)
		// backend.CopyEmbed(webgpu)
		backend.CopyEmbed(finetune)
		backend.CopyEmbed(midi)
		backend.CopyEmbed(midiAssets)
		backend.CopyEmbed(components)
	}

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

	var logger wailsLogger.Logger
	if app.Dev {
		logger = wailsLogger.NewDefaultLogger()
	} else {
		logger = wailsLogger.NewFileLogger("crash.log")
	}

	// Create application with options
	err = wails.Run(&options.App{
		Title:                    "RWKV-Runner",
		Width:                    1024,
		Height:                   700,
		MinWidth:                 375,
		MinHeight:                640,
		EnableDefaultContextMenu: true,
		Windows: &windows.Options{
			ZoomFactor:           zoomFactor,
			IsZoomControlEnabled: true,
		},
		AssetServer: &assetserver.Options{
			Assets:  assets,
			Handler: NewFileLoader(),
		},
		OnStartup:     app.OnStartup,
		OnBeforeClose: app.OnBeforeClose,
		Bind: []any{
			app,
		},
		Logger: logger,
	})

	if err != nil {
		println("Error:", err.Error())
	}
}
