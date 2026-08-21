import NavBar from './NavBar.jsx'
import { ApiBanner } from './ApiStatus.jsx'
import './layout.css'

export default function AppShell({ view, onNavigate, children }) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">F</span>
          <div className="brand-text">
            <h1>Mohamed Salem</h1>
            <p>AI engineering / financial guidance</p>
          </div>
        </div>
        <div className="header-divider" aria-hidden="true"></div>
        <NavBar view={view} onNavigate={onNavigate} />
        <div className="header-meta">
          <span className="workspace-label">Workspace / FinPilot</span>
          <span className="status-chip" title="Backend status is reported by the chat window">
            <span className="status-dot" aria-hidden="true"></span>
            System online
          </span>
        </div>
      </header>
      <ApiBanner />
      <main className="app-main">{children}</main>
      <footer className="app-footer">
        <span>FinPilot AI</span>
        <span>Clarity before complexity · Mohamed Salem</span>
      </footer>
    </div>
  )
}
