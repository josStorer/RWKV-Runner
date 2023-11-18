import { FC, LazyExoticComponent, ReactNode, Suspense } from 'react';
import { useTranslation } from 'react-i18next';
import { Spinner } from '@fluentui/react-components';

interface LazyImportComponentProps {
  lazyChildren: LazyExoticComponent<FC<any>>;
  lazyProps?: any;
  children?: ReactNode;
}

export const LazyImportComponent: FC<LazyImportComponentProps> = (props) => {
  const { t } = useTranslation();

  return (
    <Suspense fallback={
      <div className="flex justify-center items-center h-full w-full">
        <Spinner size="huge" label={t('Loading...')} />
      </div>}>
      <props.lazyChildren {...props.lazyProps}>
        {props.children}
      </props.lazyChildren>
    </Suspense>
  );
};