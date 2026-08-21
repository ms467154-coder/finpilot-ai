import { createContext, useCallback, useContext, useMemo, useState } from 'react'

const ApiStatusContext = createContext({
  lastError: null,
  reportError: () => {},
  clearError: () => {},
})

export function ApiStatusProvider({ children }) {
  const [lastError, setLastError] = useState(null)

  const reportError = useCallback((message) => {
    if (message) setLastError(String(message))
  }, [])

  const clearError = useCallback(() => setLastError(null), [])

  const value = useMemo(
    () => ({ lastError, reportError, clearError }),
    [lastError, reportError, clearError],
  )

  return <ApiStatusContext.Provider value={value}>{children}</ApiStatusContext.Provider>
}

export function useApiStatus() {
  return useContext(ApiStatusContext)
}

export function ApiBanner() {
  const { lastError, clearError } = useApiStatus()
  if (!lastError) return null
  return (
    <div className="app-banner" role="alert">
      <span className="app-banner-text">{lastError}</span>
      <button type="button" className="app-banner-dismiss" onClick={clearError}>
        Dismiss
      </button>
    </div>
  )
}

export function LoadingIndicator({ label = 'Loading…' }) {
  return (
    <div className="loading-indicator" role="status">
      <span className="loading-spinner" aria-hidden="true"></span>
      <span>{label}</span>
    </div>
  )
}
