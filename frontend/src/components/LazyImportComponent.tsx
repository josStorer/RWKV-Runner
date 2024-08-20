import { FC, LazyExoticComponent, ReactNode, Suspense } from 'react'
import { Spinner } from '@fluentui/react-components'
import { useTranslation } from 'react-i18next'

interface LazyImportComponentProps {
  lazyChildren: LazyExoticComponent<FC<any>>
  lazyProps?: any
  children?: ReactNode
}

export const LazyImportComponent: FC<LazyImportComponentProps> = (props) => {
  const { t } = useTranslation()

  return (
    <Suspense
      fallback={
        <div className="flex h-full w-full items-center justify-center">
          <Spinner size="huge" label={t('Loading...')} />
        </div>
      }
    >
      <props.lazyChildren {...props.lazyProps}>
        {props.children}
      </props.lazyChildren>
    </Suspense>
  )
}
