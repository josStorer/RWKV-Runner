package backend_golang

import (
	"errors"
	"os/exec"
	"strconv"
	"strings"
)

func (a *App) GetNvidiaGpuCount() (int, error) {
	// temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used
	// gpu_name,gpu_bus_id,driver_version
	// nvidia-smi --help-query-gpu
	output, err := exec.Command("nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(output)))
}

func (a *App) GetCudaComputeCapability(index int) (string, error) {
	output, err := exec.Command("nvidia-smi", "-i="+strconv.Itoa(index), "--query-gpu=compute_cap", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return "", err
	}

	computeCap := strings.TrimSpace(string(output))
	if computeCap == "" {
		return "", errors.New("compute capability is empty")
	}

	return computeCap, nil
}

func (a *App) GetMaxCudaComputeCapability() (string, error) {
	gpuCount, err := a.GetNvidiaGpuCount()
	if err != nil {
		return "", err
	}
	maxComputeCap := "0.0"
	for i := 0; i < gpuCount; i++ {
		computeCap, err := a.GetCudaComputeCapability(i)
		if err != nil {
			return "", err
		}
		computeCapFloat, err := strconv.ParseFloat(computeCap, 64)
		if err != nil {
			return "", err
		}
		maxComputeCapFloat, err := strconv.ParseFloat(maxComputeCap, 64)
		if err != nil {
			return "", err
		}
		if computeCapFloat > maxComputeCapFloat {
			maxComputeCap = computeCap
		}
	}
	if maxComputeCap == "0.0" {
		return "", errors.New("no cuda compute capability")
	}
	return maxComputeCap, nil
}

func (a *App) GetSupportedCudaVersion() (string, error) {
	output, err := exec.Command("nvidia-smi", "--query").CombinedOutput()
	if err != nil {
		return "", err
	}

	lines := strings.Split(string(output), "\n")

	for _, line := range lines {
		if strings.Contains(line, "CUDA Version") {
			return strings.TrimSpace(strings.Split(line, ":")[1]), nil
		}
	}

	return "", errors.New("cuda version is empty")
}

func (a *App) GetTorchVersion(python string) (string, error) {
	var err error
	if python == "" {
		python, err = a.GetPython()
		if err != nil {
			return "", err
		}
	}

	output, err := exec.Command(python, "-c", "import torch; print(torch.__version__)").CombinedOutput()
	if err != nil {
		return "", err
	}

	version := strings.TrimSpace(string(output))
	if version == "" {
		return "", errors.New("torch version is empty")
	}

	return version, nil
}
