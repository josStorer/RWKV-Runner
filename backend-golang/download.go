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
	Done        bool    `json:"done"`
}

var downloadList []DownloadStatus

func (a *App) AddToDownloadList(path string, url string) {
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
		Done:        false,
	})
}

func (a *App) downloadLoop() {
	ticker := time.NewTicker(500 * time.Millisecond)
	go func() {
		for {
			<-ticker.C
			for i, downloadStatus := range downloadList {
				downloadList[i] = DownloadStatus{
					resp:        downloadStatus.resp,
					Name:        downloadStatus.Name,
					Path:        downloadStatus.Path,
					Url:         downloadStatus.Url,
					Transferred: downloadStatus.resp.BytesComplete(),
					Size:        downloadStatus.resp.Size(),
					Speed:       downloadStatus.resp.BytesPerSecond(),
					Progress:    100 * downloadStatus.resp.Progress(),
					Done:        downloadStatus.resp.IsComplete(),
				}
			}
			runtime.EventsEmit(a.ctx, "downloadList", downloadList)
		}
	}()
}
