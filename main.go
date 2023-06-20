package main

import (
	"embed"
	"os"
	"runtime/debug"
	"strings"

	backend "rwkv-runner/backend-golang"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options/windows"
)

//go:embed all:frontend/dist
var assets embed.FS

//go:embed all:py310/Lib/site-packages/cyac
var cyac embed.FS

//go:embed all:py310/Lib/site-packages/cyac-1.7.dist-info
var cyacInfo embed.FS

//go:embed backend-python
var py embed.FS

func main() {
	if buildInfo, ok := debug.ReadBuildInfo(); !ok || strings.Contains(buildInfo.String(), "-ldflags") {
		backend.CopyEmbed(cyac)
		backend.CopyEmbed(cyacInfo)
		backend.CopyEmbed(py)
		os.Mkdir("models", os.ModePerm)
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
			Assets: assets,
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
