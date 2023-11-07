import { FC, LazyExoticComponent, ReactNode, Suspense } from 'react';
import { useTranslation } from 'react-i18next';

interface LazyImportComponentProps {
  lazyChildren: LazyExoticComponent<FC<any>>;
  lazyProps?: any;
  children?: ReactNode;
}

export const LazyImportComponent: FC<LazyImportComponentProps> = (props) => {
  const { t } = useTranslation();

  return (
    <Suspense fallback={<div>{t('Loading...')}</div>}>
      <props.lazyChildren {...props.lazyProps}>
        {props.children}
      </props.lazyChildren>
    </Suspense>
  );
};