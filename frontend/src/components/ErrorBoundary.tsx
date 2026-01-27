import React from 'react'
import { t } from 'i18next'

type ErrorBoundaryState = {
  hasError: boolean
  error?: Error
  errorInfo?: React.ErrorInfo
}

export class ErrorBoundary extends React.Component<
  React.PropsWithChildren,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { hasError: false }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error', error, info)
    this.setState({ errorInfo: info })
  }

  render() {
    if (!this.state.hasError) return this.props.children
    const message = this.state.error?.message || String(this.state.error || '')
    const stack = this.state.error?.stack || ''
    const componentStack = this.state.errorInfo?.componentStack || ''

    return (
      <div className="flex min-h-screen items-center justify-center bg-neutral-50 p-6 text-neutral-900">
        <div className="max-w-2xl rounded border border-red-200 bg-white p-6 shadow-sm">
          <div className="text-lg font-semibold text-red-700">
            {t('Error')}
          </div>
          <div className="mt-2 text-sm">
            {t('An unexpected error occurred. Please reload the app.')}
          </div>
          {message && (
            <pre className="mt-4 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-red-50 p-3 text-xs text-red-800">
              {message}
            </pre>
          )}
          {(stack || componentStack) && (
            <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-neutral-100 p-3 text-xs text-neutral-700">
              {[stack, componentStack].filter(Boolean).join('\n')}
            </pre>
          )}
        </div>
      </div>
    )
  }
}
