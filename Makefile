ifeq ($(OS), Windows_NT)
build: build-windows
else ifeq ($(shell uname -s), Darwin)
build: build-macos
else
build: build-linux
endif

build-windows:
	@echo ---- build for windows
	wails build -upx -ldflags '-s -w -extldflags "-static"' -platform windows/amd64

build-macos:
	@echo ---- build for macos
	wails build -ldflags '-s -w' -platform darwin/universal

build-linux:
	@echo ---- build for linux
	wails build -upx -ldflags '-s -w' -platform linux/amd64

build-web:
	@echo ---- build for web
	cd frontend && npm run build

dev:
	wails dev

dev-web:
	cd frontend && npm run dev

preview:
	cd frontend && npm run preview

