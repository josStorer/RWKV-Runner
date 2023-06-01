ifeq ($(OS), Windows_NT)
build: build-windows
else
build: build-macos
endif

build-windows:
	@echo ---- build for windows
	wails build -upx -ldflags "-s -w"

build-macos:
	@echo ---- build for macos
	wails build -ldflags "-s -w"

dev:
	wails dev

