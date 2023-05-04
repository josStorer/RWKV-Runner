import {Checkbox, Input, Text} from '@fluentui/react-components';
import { FC } from "react";
import { Section } from "./Section";

export const Configs: FC = () => {
    return (
        <div className="flex flex-col box-border gap-5 p-2">
            <Text size={600}>Configs</Text>
            <Section
                title="Shapes"
                content={
                    <div className="flex gap-5">
                        <Input/>
                        <Checkbox label="Temp" shape="circular" />
                    </div>
                }
            />
        </div>
    );
};
