package backend_golang

import (
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

func (a *App) StartServer(python string, port int, host string) (string, error) {
	var err error
	if python == "" {
		python, err = GetPython()
	}
	if err != nil {
		return "", err
	}
	return Cmd(python, "./backend-python/main.py", strconv.Itoa(port), host)
}

func (a *App) ConvertModel(python string, modelPath string, strategy string, outPath string) (string, error) {
	var err error
	if python == "" {
		python, err = GetPython()
	}
	if err != nil {
		return "", err
	}
	return Cmd(python, "./backend-python/convert_model.py", "--in", modelPath, "--out", outPath, "--strategy", strategy)
}

func (a *App) ConvertData(python string, input string, outputPrefix string, vocab string) (string, error) {
	var err error
	if python == "" {
		python, err = GetPython()
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
			txtFile, err := os.Open(input + "/" + file.Name())
			if err != nil {
				return "", err
			}
			defer txtFile.Close()
			jsonlFile.WriteString("{\"text\": \"")
			buf := make([]byte, 1024)
			for {
				n, err := txtFile.Read(buf)
				if err != nil {
					break
				}
				// regex replace \r\n \n \r with \\n
				jsonlFile.WriteString(
					strings.ReplaceAll(
						strings.ReplaceAll(
							strings.ReplaceAll(
								strings.ReplaceAll(string(buf[:n]),
									"\r\n", "\\n"),
								"\n", "\\n"),
							"\r", "\\n"),
						"\n\n", "\\n"))
			}
			jsonlFile.WriteString("\"}\n")
		}
		input = outputPrefix + ".jsonl"
	} else if err != nil {
		return "", err
	}

	return Cmd(python, "./finetune/json2binidx_tool/tools/preprocess_data.py", "--input", input, "--output-prefix", outputPrefix, "--vocab", vocab,
		"--tokenizer-type", tokenizerType, "--dataset-impl", "mmap", "--append-eod")
}

func (a *App) MergeLora(python string, useGpu bool, loraAlpha int, baseModel string, loraPath string, outputPath string) (string, error) {
	var err error
	if python == "" {
		python, err = GetPython()
	}
	if err != nil {
		return "", err
	}
	args := []string{python, "./finetune/lora/merge_lora.py"}
	if useGpu {
		args = append(args, "--use-gpu")
	}
	args = append(args, strconv.Itoa(loraAlpha), baseModel, loraPath, outputPath)
	return Cmd(args...)
}

func (a *App) DepCheck(python string) error {
	var err error
	if python == "" {
		python, err = GetPython()
	}
	if err != nil {
		return err
	}
	out, err := exec.Command(python, a.exDir+"./backend-python/dep_check.py").CombinedOutput()
	if err != nil {
		return errors.New("DepCheck Error: " + string(out))
	}
	return nil
}

func (a *App) InstallPyDep(python string, cnMirror bool) (string, error) {
	var err error
	if python == "" {
		python, err = GetPython()
		if runtime.GOOS == "windows" {
			python = `"%CD%/` + python + `"`
		}
	}
	if err != nil {
		return "", err
	}

	if runtime.GOOS == "windows" {
		ChangeFileLine("./py310/python310._pth", 3, "Lib\\site-packages")
		installScript := python + " ./backend-python/get-pip.py -i https://pypi.tuna.tsinghua.edu.cn/simple\n" +
			python + " -m pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117\n" +
			python + " -m pip install -r ./backend-python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple\n" +
			"exit"
		if !cnMirror {
			installScript = strings.Replace(installScript, " -i https://pypi.tuna.tsinghua.edu.cn/simple", "", -1)
			installScript = strings.Replace(installScript, "requirements.txt", "requirements_versions.txt", -1)
		}
		err = os.WriteFile("./install-py-dep.bat", []byte(installScript), 0644)
		if err != nil {
			return "", err
		}
		return Cmd("install-py-dep.bat")
	}

	if cnMirror {
		return Cmd(python, "-m", "pip", "install", "-r", "./backend-python/requirements_without_cyac.txt", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple")
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
