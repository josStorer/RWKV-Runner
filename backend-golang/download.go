package backend_golang

import (
	"context"
	"path/filepath"
	"time"

	"github.com/cavaliergopher/grab/v3"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

func (a *App) DownloadFile(path string, url string) error {
	_, err := grab.Get(a.exDir+path, url)
	if err != nil {
		return err
	}
	return nil
}

type DownloadStatus struct {
	resp        *grab.Response
	cancel      context.CancelFunc
	Name        string  `json:"name"`
	Path        string  `json:"path"`
	Url         string  `json:"url"`
	Transferred int64   `json:"transferred"`
	Size        int64   `json:"size"`
	Speed       float64 `json:"speed"`
	Progress    float64 `json:"progress"`
	Downloading bool    `json:"downloading"`
	Done        bool    `json:"done"`
}

var downloadList []*DownloadStatus

func existsInDownloadList(path string, url string) bool {
	for _, ds := range downloadList {
		if ds.Path == path || ds.Url == url {
			return true
		}
	}
	return false
}

func (a *App) PauseDownload(url string) {
	for _, ds := range downloadList {
		if ds.Url == url {
			if ds.cancel != nil {
				ds.cancel()
			}
			ds.resp = nil
			ds.Downloading = false
			ds.Speed = 0
			break
		}
	}
}

func (a *App) ContinueDownload(url string) {
	for _, ds := range downloadList {
		if ds.Url == url {
			if !ds.Downloading && ds.resp == nil && !ds.Done {
				ds.Downloading = true

				req, err := grab.NewRequest(ds.Path, ds.Url)
				if err != nil {
					ds.Downloading = false
					break
				}
				// if PauseDownload() is called before the request finished, ds.Downloading will be false
				// if the user keeps clicking pause and resume, it may result in multiple requests being successfully downloaded at the same time
				// so we have to create a context and cancel it when PauseDownload() is called
				ctx, cancel := context.WithCancel(context.Background())
				ds.cancel = cancel
				req = req.WithContext(ctx)
				resp := grab.DefaultClient.Do(req)

				if resp != nil && resp.HTTPResponse != nil &&
					resp.HTTPResponse.StatusCode >= 200 && resp.HTTPResponse.StatusCode < 300 {
					ds.resp = resp
				} else {
					ds.Downloading = false
				}
			}
			break
		}
	}
}

func (a *App) AddToDownloadList(path string, url string) {
	if !existsInDownloadList(a.exDir+path, url) {
		downloadList = append(downloadList, &DownloadStatus{
			resp:        nil,
			Name:        filepath.Base(path),
			Path:        a.exDir + path,
			Url:         url,
			Downloading: false,
		})
		a.ContinueDownload(url)
	} else {
		a.ContinueDownload(url)
	}
}

func (a *App) downloadLoop() {
	ticker := time.NewTicker(500 * time.Millisecond)
	go func() {
		for {
			<-ticker.C
			for _, ds := range downloadList {
				if ds.resp != nil {
					ds.Transferred = ds.resp.BytesComplete()
					ds.Size = ds.resp.Size()
					ds.Speed = ds.resp.BytesPerSecond()
					ds.Progress = 100 * ds.resp.Progress()
					ds.Downloading = !ds.resp.IsComplete()
					ds.Done = ds.resp.Progress() == 1
					if !ds.Downloading {
						ds.resp = nil
					}
				}
			}
			runtime.EventsEmit(a.ctx, "downloadList", downloadList)
		}
	}()
}
