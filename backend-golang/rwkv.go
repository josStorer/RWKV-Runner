// Considering some whitespace and multilingual support, the functions in rwkv.go should always be executed with cwd as RWKV-Runner, and never use a.GetAbsPath() here.
package backend_golang

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

func (a *App) StartServer(python string, port int, host string, webui bool, rwkvBeta bool, rwkvcpp bool, webgpu bool) (string, error) {
	execFile := "./backend-python/main.py"
	_, err := os.Stat(execFile)
	if err != nil {
		return "", err
	}
	if python == "" {
		python, err = GetPython(a)
	}
	if err != nil {
		return "", err
	}
	args := []string{python, execFile}
	if webui {
		args = append(args, "--webui")
	}
	if rwkvBeta {
		// args = append(args, "--rwkv-beta")
	}
	if rwkvcpp {
		args = append(args, "--rwkv.cpp")
	}
	if webgpu {
		args = append(args, "--webgpu")
	}
	args = append(args, "--port", strconv.Itoa(port), "--host", host)
	return Cmd(args...)
}

func (a *App) StartWebGPUServer(port int, host string) (string, error) {
	var execFile string
	execFiles := []string{"./backend-rust/webgpu_server", "./backend-rust/webgpu_server.exe"}
	for _, file := range execFiles {
		_, err := os.Stat(file)
		if err == nil {
			execFile = file
			break
		}
	}
	if execFile == "" {
		return "", errors.New(execFiles[0] + " not found")
	}
	args := []string{execFile}
	args = append(args, "--port", strconv.Itoa(port), "--ip", host)
	return Cmd(args...)
}

func (a *App) ConvertModel(python string, modelPath string, strategy string, outPath string) (string, error) {
	execFile := "./backend-python/convert_model.py"
	_, err := os.Stat(execFile)
	if err != nil {
		return "", err
	}
	if python == "" {
		python, err = GetPython(a)
	}
	if err != nil {
		return "", err
	}
	return Cmd(python, execFile, "--in", modelPath, "--out", outPath, "--strategy", strategy)
}

func (a *App) ConvertSafetensors(modelPath string, outPath string) (string, error) {
	var execFile string
	execFiles := []string{"./backend-rust/web-rwkv-converter", "./backend-rust/web-rwkv-converter.exe"}
	for _, file := range execFiles {
		_, err := os.Stat(file)
		if err == nil {
			execFile = file
			break
		}
	}
	if execFile == "" {
		return "", errors.New(execFiles[0] + " not found")
	}
	args := []string{execFile}
	args = append(args, "--input", modelPath, "--output", outPath)
	return Cmd(args...)
}

func (a *App) ConvertSafetensorsWithPython(python string, modelPath string, outPath string) (string, error) {
	execFile := "./backend-python/convert_safetensors.py"
	_, err := os.Stat(execFile)
	if err != nil {
		return "", err
	}
	if python == "" {
		python, err = GetPython(a)
	}
	if err != nil {
		return "", err
	}
	return Cmd(python, execFile, "--input", modelPath, "--output", outPath)
}

func (a *App) ConvertGGML(python string, modelPath string, outPath string, Q51 bool) (string, error) {
	execFile := "./backend-python/convert_pytorch_to_ggml.py"
	_, err := os.Stat(execFile)
	if err != nil {
		return "", err
	}
	if python == "" {
		python, err = GetPython(a)
	}
	if err != nil {
		return "", err
	}
	dataType := "FP16"
	if Q51 {
		dataType = "Q5_1"
	}
	return Cmd(python, execFile, modelPath, outPath, dataType)
}

func (a *App) ConvertData(python string, input string, outputPrefix string, vocab string) (string, error) {
	execFile := "./finetune/json2binidx_tool/tools/preprocess_data.py"
	_, err := os.Stat(execFile)
	if err != nil {
		return "", err
	}
	if python == "" {
		python, err = GetPython(a)
	}
	if err != nil {
		return "", err
	}
	tokenizerType := "HFTokenizer"
	if strings.Contains(vocab, "rwkv_vocab_v20230424") {
		tokenizerType = "RWKVTokenizer"
	}

	input = strings.TrimSuffix(input, "/")
	if fi, err := os.Stat(input); err == nil && fi.IsDir() {
		files, err := os.ReadDir(input)
		if err != nil {
			return "", err
		}
		jsonlFile, err := os.Create(outputPrefix + ".jsonl")
		if err != nil {
			return "", err
		}
		defer jsonlFile.Close()
		for _, file := range files {
			if file.IsDir() || !strings.HasSuffix(file.Name(), ".txt") {
				continue
			}
			textContent, err := os.ReadFile(input + "/" + file.Name())
			if err != nil {
				return "", err
			}
			textJson, err := json.Marshal(map[string]string{"text": strings.ReplaceAll(strings.ReplaceAll(string(textContent), "\r\n", "\n"), "\r", "\n")})
			if err != nil {
				return "", err
			}
			if _, err := jsonlFile.WriteString(string(textJson) + "\n"); err != nil {
				return "", err
			}
		}
		input = outputPrefix + ".jsonl"
	} else if err != nil {
		return "", err
	}

	return Cmd(python, execFile, "--input", input, "--output-prefix", outputPrefix, "--vocab", vocab,
		"--tokenizer-type", tokenizerType, "--dataset-impl", "mmap", "--append-eod")
}

func (a *App) MergeLora(python string, useGpu bool, loraAlpha int, baseModel string, loraPath string, outputPath string) (string, error) {
	execFile := "./finetune/lora/merge_lora.py"
	_, err := os.Stat(execFile)
	if err != nil {
		return "", err
	}
	if python == "" {
		python, err = GetPython(a)
	}
	if err != nil {
		return "", err
	}
	args := []string{python, execFile}
	if useGpu {
		args = append(args, "--use-gpu")
	}
	args = append(args, strconv.Itoa(loraAlpha), baseModel, loraPath, outputPath)
	return Cmd(args...)
}

func (a *App) DepCheck(python string) error {
	var err error
	if python == "" {
		python, err = GetPython(a)
	}
	if err != nil {
		return err
	}
	out, err := exec.Command(python, a.exDir+"backend-python/dep_check.py").CombinedOutput()
	if err != nil {
		return errors.New("DepCheck Error: " + string(out) + " GError: " + err.Error())
	}
	return nil
}

func (a *App) InstallPyDep(python string, cnMirror bool) (string, error) {
	var err error
	torchWhlUrl := "torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117"
	if python == "" {
		python, err = GetPython(a)
		if cnMirror && python == "py310/python.exe" {
			torchWhlUrl = "https://mirrors.aliyun.com/pytorch-wheels/cu117/torch-1.13.1+cu117-cp310-cp310-win_amd64.whl"
		}
		if runtime.GOOS == "windows" {
			python = `"%CD%/` + python + `"`
		}
	}
	if err != nil {
		return "", err
	}

	if runtime.GOOS == "windows" {
		ChangeFileLine("./py310/python310._pth", 3, "Lib\\site-packages")
		installScript := python + " ./backend-python/get-pip.py -i https://mirrors.aliyun.com/pypi/simple --no-warn-script-location\n" +
			python + " -m pip install " + torchWhlUrl + " --no-warn-script-location\n" +
			python + " -m pip install -r ./backend-python/requirements.txt -i https://mirrors.aliyun.com/pypi/simple --no-warn-script-location\n" +
			"exit"
		if !cnMirror {
			installScript = strings.Replace(installScript, " -i https://mirrors.aliyun.com/pypi/simple", "", -1)
		}
		err = os.WriteFile(a.exDir+"install-py-dep.bat", []byte(installScript), 0644)
		if err != nil {
			return "", err
		}
		return Cmd("install-py-dep.bat")
	}

	if cnMirror {
		return Cmd(python, "-m", "pip", "install", "-r", "./backend-python/requirements_without_cyac.txt", "-i", "https://mirrors.aliyun.com/pypi/simple")
	} else {
		return Cmd(python, "-m", "pip", "install", "-r", "./backend-python/requirements_without_cyac.txt")
	}
}

func (a *App) GetPyError() string {
	content, err := os.ReadFile("./error.txt")
	if err != nil {
		return ""
	}
	return string(content)
}
