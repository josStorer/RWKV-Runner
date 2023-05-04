import { Text } from "@fluentui/react-components";
import { FC } from "react";

export const SectionTitle: FC<{ label: string }> = ({ label }) => <Text weight="medium">{label}</Text>;
