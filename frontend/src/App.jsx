import { useState } from 'react'
import { ApiStatusProvider } from './components/Layout/ApiStatus.jsx'
import { ChatProvider } from './components/Chat/ChatContext.jsx'
import AppShell from './components/Layout/AppShell.jsx'
import ErrorBoundary from './components/Layout/ErrorBoundary.jsx'
import ChatWindow from './components/Chat/ChatWindow.jsx'
import DashboardPage from './pages/Dashboard/DashboardPage.jsx'

const ROUTES = {
  chat: ChatWindow,
  dashboard: DashboardPage,
}

export default function App() {
  const [view, setView] = useState('chat')
  const RouteComponent = ROUTES[view]

  return (
    <ApiStatusProvider>
      <ChatProvider>
        <AppShell view={view} onNavigate={setView}>
          <ErrorBoundary>
            <RouteComponent onNavigate={setView} />
          </ErrorBoundary>
        </AppShell>
      </ChatProvider>
    </ApiStatusProvider>
  )
}
