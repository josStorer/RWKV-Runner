import {Button, Divider, Dropdown, Input, Option, Slider, Switch, Text} from '@fluentui/react-components';
import {AddCircle20Regular, DataUsageSettings20Regular, Delete20Regular, Save20Regular} from '@fluentui/react-icons';
import React, {FC} from 'react';
import {Section} from '../components/Section';
import {Labeled} from '../components/Labeled';
import {ToolTipButton} from '../components/ToolTipButton';

export const Configs: FC = () => {
  return (
    <div className="flex flex-col gap-2 p-2 h-full">
      <Text size={600}>Configs</Text>
      <Divider/>
      <div className="flex gap-2 items-center w-full">
        <Dropdown style={{minWidth: 0}} className="grow"/>
        <ToolTipButton desc="New Config" icon={<AddCircle20Regular/>}/>
        <ToolTipButton desc="Delete Config" icon={<Delete20Regular/>}/>
        <ToolTipButton desc="Save Config" icon={<Save20Regular/>}/>
      </div>
      <div className="flex flex-col h-full gap-2 overflow-y-hidden">
        <Section
          title="Default API Parameters"
          desc="Hover your mouse over the text to view a detailed description. Settings marked with * will take effect immediately after being saved."
          content={
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <Labeled label="API Port" desc="127.0.0.1:8000" content={
                <Input type="number" min={1} max={65535} step={1}/>
              }/>
              <Labeled label="Max Response Token *" content={
                <div className="flex items-center grow">
                  <Slider style={{minWidth: 0}} className="grow" step={400} min={100} max={8100}/>
                  <Text>1000</Text>
                </div>
              }/>
              <Labeled label="Temperature *" content={
                <Slider style={{minWidth: 0}} className="grow"/>
              }/>
              <Labeled label="Top_P *" content={
                <Slider style={{minWidth: 0}} className="grow"/>
              }/>
              <Labeled label="Presence Penalty *" content={
                <Slider style={{minWidth: 0}} className="grow"/>
              }/>
              <Labeled label="Count Penalty *" content={
                <Slider style={{minWidth: 0}} className="grow"/>
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
                  <Dropdown style={{minWidth: 0}} className="grow"/>
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
  );
};
