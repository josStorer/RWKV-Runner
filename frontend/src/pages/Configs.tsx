import {
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  DialogTrigger,
  Dropdown,
  Input,
  Label,
  Option,
  Select,
  Switch
} from '@fluentui/react-components';
import {
  AddCircle20Regular,
  ArrowReset20Regular,
  DataUsageSettings20Regular,
  Delete20Regular,
  Save20Regular
} from '@fluentui/react-icons';
import React, { FC } from 'react';
import { Section } from '../components/Section';
import { Labeled } from '../components/Labeled';
import { ToolTipButton } from '../components/ToolTipButton';
import commonStore from '../stores/commonStore';
import { observer } from 'mobx-react-lite';
import { toast } from 'react-toastify';
import { ValuedSlider } from '../components/ValuedSlider';
import { NumberInput } from '../components/NumberInput';
import { Page } from '../components/Page';
import { useNavigate } from 'react-router';
import { RunButton } from '../components/RunButton';
import { updateConfig } from '../apis';
import { ConvertModel, FileExists } from '../../wailsjs/go/backend_golang/App';
import manifest from '../../../manifest.json';
import { getStrategy, refreshLocalModels } from '../utils';
import { useTranslation } from 'react-i18next';
import { WindowShow } from '../../wailsjs/runtime/runtime';

export type ApiParameters = {
  apiPort: number
  maxResponseToken: number;
  temperature: number;
  topP: number;
  presencePenalty: number;
  frequencyPenalty: number;
}

export type Device = 'CPU' | 'CUDA' | 'Custom';
export type Precision = 'fp16' | 'int8' | 'fp32';

export type ModelParameters = {
  // different models can not have the same name
  modelName: string;
  device: Device;
  precision: Precision;
  storedLayers: number;
  maxStoredLayers: number;
  enableHighPrecisionForLastLayer: boolean;
  useCustomCuda?: boolean;
  customStrategy?: string;
}

export type ModelConfig = {
  // different configs can have the same name
  name: string;
  apiParameters: ApiParameters
  modelParameters: ModelParameters
}

export const defaultModelConfigs: ModelConfig[] = [
  {
    name: 'GPU-2G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 4,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 8,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 8,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 18,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 18,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 27,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 27,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-10G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-10G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-12G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-16G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-16G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-16G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 37,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-18G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-32G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'CPU-6G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-12G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-12G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-28G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-28G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  }
];

export const Configs: FC = observer(() => {
  const { t } = useTranslation();
  const [selectedIndex, setSelectedIndex] = React.useState(commonStore.currentModelConfigIndex);
  const [selectedConfig, setSelectedConfig] = React.useState(commonStore.modelConfigs[selectedIndex]);
  const navigate = useNavigate();
  const port = selectedConfig.apiParameters.apiPort;

  const updateSelectedIndex = (newIndex: number) => {
    setSelectedIndex(newIndex);
    setSelectedConfig(commonStore.modelConfigs[newIndex]);

    // if you don't want to update the config used by the current startup in real time, comment out this line
    commonStore.setCurrentConfigIndex(newIndex);
  };

  const setSelectedConfigName = (newName: string) => {
    setSelectedConfig({ ...selectedConfig, name: newName });
  };

  const setSelectedConfigApiParams = (newParams: Partial<ApiParameters>) => {
    setSelectedConfig({
      ...selectedConfig, apiParameters: {
        ...selectedConfig.apiParameters,
        ...newParams
      }
    });
  };

  const setSelectedConfigModelParams = (newParams: Partial<ModelParameters>) => {
    setSelectedConfig({
      ...selectedConfig, modelParameters: {
        ...selectedConfig.modelParameters,
        ...newParams
      }
    });
  };

  const onClickSave = () => {
    commonStore.setModelConfig(selectedIndex, selectedConfig);
    updateConfig({
      max_tokens: selectedConfig.apiParameters.maxResponseToken,
      temperature: selectedConfig.apiParameters.temperature,
      top_p: selectedConfig.apiParameters.topP,
      presence_penalty: selectedConfig.apiParameters.presencePenalty,
      frequency_penalty: selectedConfig.apiParameters.frequencyPenalty
    });
    toast(t('Config Saved'), { autoClose: 300, type: 'success' });
  };

  return (
    <Page title={t('Configs')} content={
      <div className="flex flex-col gap-2 overflow-hidden">
        <div className="flex gap-2 items-center">
          <Dropdown style={{ minWidth: 0 }} className="grow" value={commonStore.modelConfigs[selectedIndex].name}
            selectedOptions={[selectedIndex.toString()]}
            onOptionSelect={(_, data) => {
              if (data.optionValue) {
                updateSelectedIndex(Number(data.optionValue));
              }
            }}>
            {commonStore.modelConfigs.map((config, index) =>
              <Option key={index} value={index.toString()}>{config.name}</Option>
            )}
          </Dropdown>
          <ToolTipButton desc={t('New Config')} icon={<AddCircle20Regular />} onClick={() => {
            commonStore.createModelConfig();
            updateSelectedIndex(commonStore.modelConfigs.length - 1);
          }} />
          <ToolTipButton desc={t('Delete Config')} icon={<Delete20Regular />} onClick={() => {
            commonStore.deleteModelConfig(selectedIndex);
            updateSelectedIndex(Math.min(selectedIndex, commonStore.modelConfigs.length - 1));
          }} />
          <Dialog>
            <DialogTrigger disableButtonEnhancement>
              <ToolTipButton desc={t('Reset All Configs')} icon={<ArrowReset20Regular />} />
            </DialogTrigger>
            <DialogSurface>
              <DialogBody>
                <DialogTitle>{t('Reset All Configs')}</DialogTitle>
                <DialogContent>
                  {t('Are you sure you want to reset all configs? This will obtain the latest preset configs, but will override your custom configs and cannot be undone.')}
                </DialogContent>
                <DialogActions>
                  <DialogTrigger disableButtonEnhancement>
                    <Button appearance="secondary">{t('Cancel')}</Button>
                  </DialogTrigger>
                  <DialogTrigger disableButtonEnhancement>
                    <Button appearance="primary" onClick={() => {
                      commonStore.setModelConfigs(defaultModelConfigs, false); // updateSelectedIndex() will save configs
                      updateSelectedIndex(0);
                    }}>{t('Confirm')}
                    </Button>
                  </DialogTrigger>
                </DialogActions>
              </DialogBody>
            </DialogSurface>
          </Dialog>
          <ToolTipButton desc={t('Save Config')} icon={<Save20Regular />} onClick={onClickSave} />
        </div>
        <div className="flex items-center gap-4">
          <Label>{t('Config Name')}</Label>
          <Input className="grow" value={selectedConfig.name} onChange={(e, data) => {
            setSelectedConfigName(data.value);
          }} />
        </div>
        <div className="flex flex-col gap-2 overflow-y-hidden">
          <Section
            title={t('Default API Parameters')}
            desc={t('Hover your mouse over the text to view a detailed description. Settings marked with * will take effect immediately after being saved.')}
            content={
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <Labeled label={t('API Port')}
                  desc={t('Open the following URL with your browser to view the API documentation') + `: http://127.0.0.1:${port}/docs. ` +
                    t('This tool\'s API is compatible with OpenAI API. It can be used with any ChatGPT tool you like. Go to the settings of some ChatGPT tool, replace the \'https://api.openai.com\' part in the API address with \'') + `http://127.0.0.1:${port}` + '\'.'}
                  content={
                    <NumberInput value={port} min={1} max={65535} step={1}
                      onChange={(e, data) => {
                        setSelectedConfigApiParams({
                          apiPort: data.value
                        });
                      }} />
                  } />
                <Labeled label={t('Max Response Token') + ' *'}
                  desc={t('By default, the maximum number of tokens that can be answered in a single response, it can be changed by the user by specifying API parameters.')}
                  content={
                    <ValuedSlider value={selectedConfig.apiParameters.maxResponseToken} min={100} max={8100}
                      step={400}
                      input
                      onChange={(e, data) => {
                        setSelectedConfigApiParams({
                          maxResponseToken: data.value
                        });
                      }} />
                  } />
                <Labeled label={t('Temperature') + ' *'}
                  desc={t('Sampling temperature, the higher the stronger the randomness and creativity, while the lower, the more focused and deterministic it will be.')}
                  content={
                    <ValuedSlider value={selectedConfig.apiParameters.temperature} min={0} max={2} step={0.1}
                      input
                      onChange={(e, data) => {
                        setSelectedConfigApiParams({
                          temperature: data.value
                        });
                      }} />
                  } />
                <Labeled label={t('Top_P') + ' *'}
                  desc={t('Consider the results of the top n% probability mass, 0.1 considers the top 10%, with higher quality but more conservative, 1 considers all results, with lower quality but more diverse.')}
                  content={
                    <ValuedSlider value={selectedConfig.apiParameters.topP} min={0} max={1} step={0.1} input
                      onChange={(e, data) => {
                        setSelectedConfigApiParams({
                          topP: data.value
                        });
                      }} />
                  } />
                <Labeled label={t('Presence Penalty') + ' *'}
                  desc={t('Positive values penalize new tokens based on whether they appear in the text so far, increasing the model\'s likelihood to talk about new topics.')}
                  content={
                    <ValuedSlider value={selectedConfig.apiParameters.presencePenalty} min={-2} max={2}
                      step={0.1} input
                      onChange={(e, data) => {
                        setSelectedConfigApiParams({
                          presencePenalty: data.value
                        });
                      }} />
                  } />
                <Labeled label={t('Frequency Penalty') + ' *'}
                  desc={t('Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model\'s likelihood to repeat the same line verbatim.')}
                  content={
                    <ValuedSlider value={selectedConfig.apiParameters.frequencyPenalty} min={-2} max={2}
                      step={0.1} input
                      onChange={(e, data) => {
                        setSelectedConfigApiParams({
                          frequencyPenalty: data.value
                        });
                      }} />
                  } />
              </div>
            }
          />
          <Section
            title={t('Model Parameters')}
            content={
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <Labeled label={t('Model')} content={
                  <div className="flex gap-2 grow">
                    <Select style={{ minWidth: 0 }} className="grow"
                      value={selectedConfig.modelParameters.modelName}
                      onChange={(e, data) => {
                        setSelectedConfigModelParams({
                          modelName: data.value
                        });
                      }}>
                      {!commonStore.modelSourceList.find(item => item.name === selectedConfig.modelParameters.modelName)?.isLocal
                        && <option key={-1}
                          value={selectedConfig.modelParameters.modelName}>{selectedConfig.modelParameters.modelName}
                        </option>}
                      {commonStore.modelSourceList.map((modelItem, index) =>
                        modelItem.isLocal && <option key={index} value={modelItem.name}>{modelItem.name}</option>
                      )}
                    </Select>
                    <ToolTipButton desc={t('Manage Models')} icon={<DataUsageSettings20Regular />} onClick={() => {
                      navigate({ pathname: '/models' });
                    }} />
                  </div>
                } />
                <ToolTipButton text={t('Convert')} desc={t('Convert model with these configs')} onClick={async () => {
                  const modelPath = `${manifest.localModelDir}/${selectedConfig.modelParameters.modelName}`;
                  if (await FileExists(modelPath)) {
                    const strategy = getStrategy(selectedConfig);
                    const newModelPath = modelPath + '-' + strategy.replace(/[:> *+]/g, '-');
                    toast(t('Start Converting'), { autoClose: 1000, type: 'info' });
                    ConvertModel(modelPath, strategy, newModelPath).then(() => {
                      toast(`${t('Convert Success')} - ${newModelPath}`, { type: 'success' });
                      refreshLocalModels({ models: commonStore.modelSourceList }, false);
                    }).catch(e => {
                      toast(`${t('Convert Failed')} - ${e.message || e}`, { type: 'error' });
                    });
                    setTimeout(WindowShow, 1000);
                  } else {
                    toast(`${t('Model Not Found')} - ${modelPath}`, { type: 'error' });
                  }
                }} />
                <Labeled label={t('Device')} content={
                  <Dropdown style={{ minWidth: 0 }} className="grow" value={t(selectedConfig.modelParameters.device)!}
                    selectedOptions={[selectedConfig.modelParameters.device]}
                    onOptionSelect={(_, data) => {
                      if (data.optionValue) {
                        setSelectedConfigModelParams({
                          device: data.optionValue as Device
                        });
                      }
                    }}>
                    <Option value="CPU">CPU</Option>
                    <Option value="CUDA">CUDA</Option>
                    <Option value="Custom">{t('Custom')!}</Option>
                  </Dropdown>
                } />
                {
                  selectedConfig.modelParameters.device != 'Custom' && <Labeled label={t('Precision')}
                    desc={t('int8 uses less VRAM, but has slightly lower quality. fp16 has higher quality, and fp32 has the best quality.')}
                    content={
                      <Dropdown style={{ minWidth: 0 }} className="grow"
                        value={selectedConfig.modelParameters.precision}
                        selectedOptions={[selectedConfig.modelParameters.precision]}
                        onOptionSelect={(_, data) => {
                          if (data.optionText) {
                            setSelectedConfigModelParams({
                              precision: data.optionText as Precision
                            });
                          }
                        }}>
                        <Option>fp16</Option>
                        <Option>int8</Option>
                        <Option>fp32</Option>
                      </Dropdown>
                    } />
                }
                {selectedConfig.modelParameters.device == 'CUDA' && <div />}
                {
                  selectedConfig.modelParameters.device == 'CUDA' && <Labeled label={t('Stored Layers')}
                    desc={t('Number of the neural network layers loaded into VRAM, the more you load, the faster the speed, but it consumes more VRAM.')}
                    content={
                      <ValuedSlider value={selectedConfig.modelParameters.storedLayers} min={0}
                        max={selectedConfig.modelParameters.maxStoredLayers} step={1} input
                        onChange={(e, data) => {
                          setSelectedConfigModelParams({
                            storedLayers: data.value
                          });
                        }} />
                    } />
                }
                {
                  selectedConfig.modelParameters.device == 'CUDA' &&
                  <Labeled label={t('Enable High Precision For Last Layer')}
                    desc={t('Whether to use CPU to calculate the last output layer of the neural network with FP32 precision to obtain better quality.')}
                    content={
                      <Switch checked={selectedConfig.modelParameters.enableHighPrecisionForLastLayer}
                        onChange={(e, data) => {
                          setSelectedConfigModelParams({
                            enableHighPrecisionForLastLayer: data.checked
                          });
                        }} />
                    } />
                }
                {
                  selectedConfig.modelParameters.device == 'Custom' &&
                  <Labeled label="Strategy" desc="https://github.com/BlinkDL/ChatRWKV/blob/main/ChatRWKV-strategy.png"
                    content={
                      <Input className="grow" placeholder="cuda:0 fp16 *20 -> cuda:1 fp16"
                        value={selectedConfig.modelParameters.customStrategy}
                        onChange={(e, data) => {
                          setSelectedConfigModelParams({
                            customStrategy: data.value
                          });
                        }} />
                    } />
                }
                {selectedConfig.modelParameters.device == 'Custom' && <div />}
                {
                  selectedConfig.modelParameters.device != 'CPU' &&
                  <Labeled label={t('Use Custom CUDA kernel to Accelerate')}
                    desc={t('Enabling this option can greatly improve inference speed, but there may be compatibility issues. If it fails to start, please turn off this option.')}
                    content={
                      <Switch checked={selectedConfig.modelParameters.useCustomCuda}
                        onChange={(e, data) => {
                          setSelectedConfigModelParams({
                            useCustomCuda: data.checked
                          });
                        }} />
                    } />
                }
              </div>
            }
          />
        </div>
        <div className="flex flex-row-reverse sm:fixed bottom-2 right-2">
          <RunButton onClickRun={onClickSave} />
        </div>
      </div>
    } />
  );
});
