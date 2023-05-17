import {Dropdown, Input, Label, Option, Select, Switch} from '@fluentui/react-components';
import {AddCircle20Regular, DataUsageSettings20Regular, Delete20Regular, Save20Regular} from '@fluentui/react-icons';
import React, {FC} from 'react';
import {Section} from '../components/Section';
import {Labeled} from '../components/Labeled';
import {ToolTipButton} from '../components/ToolTipButton';
import commonStore, {ApiParameters, Device, ModelParameters, Precision} from '../stores/commonStore';
import {observer} from 'mobx-react-lite';
import {toast} from 'react-toastify';
import {ValuedSlider} from '../components/ValuedSlider';
import {NumberInput} from '../components/NumberInput';
import {Page} from '../components/Page';
import {useNavigate} from 'react-router';
import {RunButton} from '../components/RunButton';
import {updateConfig} from '../apis';
import {ConvertModel} from '../../wailsjs/go/backend_golang/App';
import manifest from '../../../manifest.json';
import {getStrategy, refreshLocalModels} from '../utils';

export const Configs: FC = observer(() => {
  const [selectedIndex, setSelectedIndex] = React.useState(commonStore.currentModelConfigIndex);
  const [selectedConfig, setSelectedConfig] = React.useState(commonStore.modelConfigs[selectedIndex]);

  const navigate = useNavigate();

  const updateSelectedIndex = (newIndex: number) => {
    setSelectedIndex(newIndex);
    setSelectedConfig(commonStore.modelConfigs[newIndex]);

    // if you don't want to update the config used by the current startup in real time, comment out this line
    commonStore.setCurrentConfigIndex(newIndex);
  };

  const setSelectedConfigName = (newName: string) => {
    setSelectedConfig({...selectedConfig, name: newName});
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
    toast('Config Saved', {autoClose: 300, type: 'success'});
  };

  return (
    <Page title="Configs" content={
      <div className="flex flex-col gap-2 overflow-hidden">
        <div className="flex gap-2 items-center">
          <Dropdown style={{minWidth: 0}} className="grow" value={commonStore.modelConfigs[selectedIndex].name}
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
          <ToolTipButton desc="New Config" icon={<AddCircle20Regular/>} onClick={() => {
            commonStore.createModelConfig();
            updateSelectedIndex(commonStore.modelConfigs.length - 1);
          }}/>
          <ToolTipButton desc="Delete Config" icon={<Delete20Regular/>} onClick={() => {
            commonStore.deleteModelConfig(selectedIndex);
            updateSelectedIndex(Math.min(selectedIndex, commonStore.modelConfigs.length - 1));
          }}/>
          <ToolTipButton desc="Save Config" icon={<Save20Regular/>} onClick={onClickSave}/>
        </div>
        <div className="flex items-center gap-4">
          <Label>Config Name</Label>
          <Input className="grow" value={selectedConfig.name} onChange={(e, data) => {
            setSelectedConfigName(data.value);
          }}/>
        </div>
        <div className="flex flex-col gap-2 overflow-y-hidden">
          <Section
            title="Default API Parameters"
            desc="Hover your mouse over the text to view a detailed description. Settings marked with * will take effect immediately after being saved."
            content={
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <Labeled label="API Port" desc={`127.0.0.1:${selectedConfig.apiParameters.apiPort}`} content={
                  <NumberInput value={selectedConfig.apiParameters.apiPort} min={1} max={65535} step={1}
                               onChange={(e, data) => {
                                 setSelectedConfigApiParams({
                                   apiPort: data.value
                                 });
                               }}/>
                }/>
                <Labeled label="Max Response Token *" content={
                  <ValuedSlider value={selectedConfig.apiParameters.maxResponseToken} min={100} max={8100} step={400}
                                input
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    maxResponseToken: data.value
                                  });
                                }}/>
                }/>
                <Labeled label="Temperature *" content={
                  <ValuedSlider value={selectedConfig.apiParameters.temperature} min={0} max={2} step={0.1} input
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    temperature: data.value
                                  });
                                }}/>
                }/>
                <Labeled label="Top_P *" content={
                  <ValuedSlider value={selectedConfig.apiParameters.topP} min={0} max={1} step={0.1} input
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    topP: data.value
                                  });
                                }}/>
                }/>
                <Labeled label="Presence Penalty *" content={
                  <ValuedSlider value={selectedConfig.apiParameters.presencePenalty} min={-2} max={2} step={0.1} input
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    presencePenalty: data.value
                                  });
                                }}/>
                }/>
                <Labeled label="Frequency Penalty *" content={
                  <ValuedSlider value={selectedConfig.apiParameters.frequencyPenalty} min={-2} max={2} step={0.1} input
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    frequencyPenalty: data.value
                                  });
                                }}/>
                }/>
              </div>
            }
          />
          <Section
            title="Model Parameters"
            content={
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <Labeled label="Model" content={
                  <div className="flex gap-2 grow">
                    <Select style={{minWidth: 0}} className="grow"
                            value={selectedConfig.modelParameters.modelName}
                            onChange={(e, data) => {
                              setSelectedConfigModelParams({
                                modelName: data.value
                              });
                            }}>
                      {commonStore.modelSourceList.map((modelItem, index) =>
                        modelItem.isLocal && <option key={index} value={modelItem.name}>{modelItem.name}</option>
                      )}
                    </Select>
                    <ToolTipButton desc="Manage Models" icon={<DataUsageSettings20Regular/>} onClick={() => {
                      navigate({pathname: '/models'});
                    }}/>
                  </div>
                }/>
                <ToolTipButton text="Convert" desc="Convert model with these configs" onClick={() => {
                  const modelPath = `${manifest.localModelDir}/${selectedConfig.modelParameters.modelName}`;
                  const strategy = getStrategy(selectedConfig);
                  const newModelPath = modelPath + '-' + strategy.replace(/[> *+]/g, '-');
                  toast('Start Converting', {autoClose: 1000, type: 'info'});
                  ConvertModel(modelPath, strategy, newModelPath).then(() => {
                    toast(`Convert Success - ${newModelPath}`, {type: 'success'});
                    refreshLocalModels({models: commonStore.modelSourceList});
                  }).catch(e => {
                    toast(`Convert Failed - ${e}`, {type: 'error'});
                  });
                }}/>
                <Labeled label="Device" content={
                  <Dropdown style={{minWidth: 0}} className="grow" value={selectedConfig.modelParameters.device}
                            selectedOptions={[selectedConfig.modelParameters.device]}
                            onOptionSelect={(_, data) => {
                              if (data.optionText) {
                                setSelectedConfigModelParams({
                                  device: data.optionText as Device
                                });
                              }
                            }}>
                    <Option>CPU</Option>
                    <Option>CUDA</Option>
                  </Dropdown>
                }/>
                <Labeled label="Precision" content={
                  <Dropdown style={{minWidth: 0}} className="grow" value={selectedConfig.modelParameters.precision}
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
                }/>
                <Labeled label="Stored Layers" content={
                  <ValuedSlider value={selectedConfig.modelParameters.storedLayers} min={0}
                                max={selectedConfig.modelParameters.maxStoredLayers} step={1} input
                                onChange={(e, data) => {
                                  setSelectedConfigModelParams({
                                    storedLayers: data.value
                                  });
                                }}/>
                }/>
                <Labeled label="Enable High Precision For Last Layer" content={
                  <Switch checked={selectedConfig.modelParameters.enableHighPrecisionForLastLayer}
                          onChange={(e, data) => {
                            setSelectedConfigModelParams({
                              enableHighPrecisionForLastLayer: data.checked
                            });
                          }}/>
                }/>
              </div>
            }
          />
        </div>
        <div className="flex flex-row-reverse sm:fixed bottom-2 right-2">
          <RunButton onClickRun={onClickSave}/>
        </div>
      </div>
    }/>
  );
});
