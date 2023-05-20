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
			ds.resp.Cancel()

			downloadList[i] = DownloadStatus{
				resp:        ds.resp,
				Name:        ds.Name,
				Path:        ds.Path,
				Url:         ds.Url,
				Transferred: ds.Transferred,
				Size:        ds.Size,
				Speed:       0,
				Progress:    ds.Progress,
				Downloading: false,
				Done:        ds.Done,
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
				Transferred: ds.Transferred,
				Size:        ds.Size,
				Speed:       ds.Speed,
				Progress:    ds.Progress,
				Downloading: true,
				Done:        ds.Done,
			}
		}
	}
}

func (a *App) AddToDownloadList(path string, url string) {
	if !existsInDownloadList(url) {
		client := grab.NewClient()
		req, _ := grab.NewRequest(path, url)
		resp := client.Do(req)

		downloadList = append(downloadList, DownloadStatus{
			resp:        resp,
			Name:        filepath.Base(path),
			Path:        path,
			Url:         url,
			Transferred: 0,
			Size:        0,
			Speed:       0,
			Progress:    0,
			Downloading: true,
			Done:        false,
		})
	}
}

func (a *App) downloadLoop() {
	ticker := time.NewTicker(500 * time.Millisecond)
	go func() {
		for {
			<-ticker.C
			for i, ds := range downloadList {
				downloadList[i] = DownloadStatus{
					resp:        ds.resp,
					Name:        ds.Name,
					Path:        ds.Path,
					Url:         ds.Url,
					Transferred: ds.resp.BytesComplete(),
					Size:        ds.resp.Size(),
					Speed:       ds.resp.BytesPerSecond(),
					Progress:    100 * ds.resp.Progress(),
					Downloading: !ds.resp.IsComplete(),
					Done:        ds.resp.Progress() == 1,
				}
			}
			runtime.EventsEmit(a.ctx, "downloadList", downloadList)
		}
	}()
}
