import {Button, Dropdown, Input, Label, Option, Slider, Switch} from '@fluentui/react-components';
import {AddCircle20Regular, DataUsageSettings20Regular, Delete20Regular, Save20Regular} from '@fluentui/react-icons';
import React, {FC} from 'react';
import {Section} from '../components/Section';
import {Labeled} from '../components/Labeled';
import {ToolTipButton} from '../components/ToolTipButton';
import commonStore, {ApiParameters, ModelParameters} from '../stores/commonStore';
import {observer} from 'mobx-react-lite';
import {toast} from 'react-toastify';
import {ValuedSlider} from '../components/ValuedSlider';
import {NumberInput} from '../components/NumberInput';
import {Page} from '../components/Page';

export const Configs: FC = observer(() => {
  const [selectedIndex, setSelectedIndex] = React.useState(commonStore.currentModelConfigIndex);
  const [selectedConfig, setSelectedConfig] = React.useState(commonStore.modelConfigs[selectedIndex]);

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
          <ToolTipButton desc="Save Config" icon={<Save20Regular/>} onClick={() => {
            commonStore.setModelConfig(selectedIndex, selectedConfig);
            toast('Config Saved', {hideProgressBar: true, autoClose: 300, position: 'top-center'});
          }}/>
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
                <Labeled label="API Port" desc="127.0.0.1:8000" content={
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
                <Labeled label="Count Penalty *" content={
                  <ValuedSlider value={selectedConfig.apiParameters.countPenalty} min={-2} max={2} step={0.1} input
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    countPenalty: data.value
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
                    <Dropdown style={{minWidth: 0}} className="grow">
                      {commonStore.modelSourceList.map((modelItem, index) =>
                        <Option key={index} value={index.toString()}>{modelItem.name}</Option>
                      )}
                    </Dropdown>
                    <ToolTipButton desc="Manage Models" icon={<DataUsageSettings20Regular/>}/>
                  </div>
                }/>
                <ToolTipButton text="Convert" desc="Convert model with these configs"/>
                <Labeled label="Device" content={
                  <Dropdown style={{minWidth: 0}} className="grow">
                    <Option>CPU</Option>
                    <Option>CUDA: 0</Option>
                  </Dropdown>
                }/>
                <Labeled label="Precision" content={
                  <Dropdown style={{minWidth: 0}} className="grow">
                    <Option>fp16</Option>
                    <Option>int8</Option>
                    <Option>fp32</Option>
                  </Dropdown>
                }/>
                <Labeled label="Streamed Layers" content={
                  <Slider style={{minWidth: 0}} className="grow"/>
                }/>
                <Labeled label="Enable High Precision For Last Layer" content={
                  <Switch/>
                }/>
              </div>
            }
          />
        </div>
        <div className="flex flex-row-reverse sm:fixed bottom-2 right-2">
          <Button appearance="primary" size="large">Run</Button>
        </div>
      </div>
    }/>
  );
});
