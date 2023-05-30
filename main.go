package main

import (
	"embed"
	"runtime/debug"
	"strings"

	backend "rwkv-runner/backend-golang"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
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
	}

	// Create an instance of the app structure
	app := backend.NewApp()

	// Create application with options
	err := wails.Run(&options.App{
		Title:     "RWKV-Runner",
		Width:     1024,
		Height:    640,
		MinWidth:  375,
		MinHeight: 640,
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
