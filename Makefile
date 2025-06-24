ifeq ($(OS), Windows_NT)
build: build-windows
else ifeq ($(shell uname -s), Darwin)
build: build-macos
else
build: build-linux
endif

windows_build = wails build -ldflags '-s -w -extldflags "-static"' -platform windows/amd64 -devtools -upx -upxflags "-9 --lzma"

build-windows:
	@echo ---- build for windows
	$(windows_build) -nsis

debug:
	$(windows_build) -windowsconsole

build-macos:
	@echo ---- build for macos
	wails build -ldflags '-s -w' -platform darwin/universal -devtools

build-linux:
	@echo ---- build for linux
	wails build -ldflags '-s -w' -platform linux/amd64 -devtools -upx -upxflags "-9 --lzma"

build-web:
	@echo ---- build for web
	cd frontend && npm run build

dev:
	wails dev

# go install github.com/josStorer/wails/v2/cmd/wails@v2.9.2x
devq:
	wails dev -s -m -skipembedcreate -skipbindings

devq2:
	wails dev -s -m -skipembedcreate

dev-web:
	cd frontend && npm run dev

preview:
	cd frontend && npm run preview

