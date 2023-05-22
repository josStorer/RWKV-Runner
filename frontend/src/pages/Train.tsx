import React, { FC } from 'react';
import { Text } from '@fluentui/react-components';
import { useTranslation } from 'react-i18next';

export const Train: FC = () => {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col box-border gap-5 p-2">
      <Text size={600}>{t('In Development')}</Text>
    </div>
  );
};
