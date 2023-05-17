package backend_golang

import (
	"context"
	"net/http"

	"github.com/minio/selfupdate"
)

// App struct
type App struct {
	ctx context.Context
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) OnStartup(ctx context.Context) {
	a.ctx = ctx
}

func (a *App) UpdateApp(url string) (broken bool, err error) {
	resp, err := http.Get(url)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	err = selfupdate.Apply(resp.Body, selfupdate.Options{})
	if err != nil {
		if rerr := selfupdate.RollbackError(err); rerr != nil {
			return true, rerr
		}
		return false, err
	}
	return false, nil
}
