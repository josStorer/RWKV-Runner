package backend_golang

import (
	"path/filepath"
	"time"

	"github.com/cavaliergopher/grab/v3"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

func (a *App) DownloadFile(path string, url string) error {
	_, err := grab.Get(path, url)
	if err != nil {
		return err
	}
	return nil
}

type DownloadStatus struct {
	resp        *grab.Response
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

var downloadList []DownloadStatus

func existsInDownloadList(url string) bool {
	for _, ds := range downloadList {
		if ds.Url == url {
			return true
		}
	}
	return false
}

func (a *App) PauseDownload(url string) {
	for i, ds := range downloadList {
		if ds.Url == url {
			if ds.resp != nil {
				ds.resp.Cancel()
			}

			downloadList[i] = DownloadStatus{
				resp:        ds.resp,
				Name:        ds.Name,
				Path:        ds.Path,
				Url:         ds.Url,
				Downloading: false,
			}
		}
	}
}

func (a *App) ContinueDownload(url string) {
	for i, ds := range downloadList {
		if ds.Url == url {
			client := grab.NewClient()
			req, _ := grab.NewRequest(ds.Path, ds.Url)
			resp := client.Do(req)

			downloadList[i] = DownloadStatus{
				resp:        resp,
				Name:        ds.Name,
				Path:        ds.Path,
				Url:         ds.Url,
				Downloading: true,
			}
		}
	}
}

func (a *App) AddToDownloadList(path string, url string) {
	if !existsInDownloadList(url) {
		downloadList = append(downloadList, DownloadStatus{
			resp:        nil,
			Name:        filepath.Base(path),
			Path:        path,
			Url:         url,
			Downloading: true,
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
			for i, ds := range downloadList {
				transferred := int64(0)
				size := int64(0)
				speed := float64(0)
				progress := float64(0)
				downloading := ds.Downloading
				done := false
				if ds.resp != nil {
					transferred = ds.resp.BytesComplete()
					size = ds.resp.Size()
					speed = ds.resp.BytesPerSecond()
					progress = 100 * ds.resp.Progress()
					downloading = !ds.resp.IsComplete()
					done = ds.resp.Progress() == 1
				}
				downloadList[i] = DownloadStatus{
					resp:        ds.resp,
					Name:        ds.Name,
					Path:        ds.Path,
					Url:         ds.Url,
					Transferred: transferred,
					Size:        size,
					Speed:       speed,
					Progress:    progress,
					Downloading: downloading,
					Done:        done,
				}
			}
			runtime.EventsEmit(a.ctx, "downloadList", downloadList)
		}
	}()
}
