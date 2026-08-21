const NAV_ITEMS = [
  { id: 'chat', label: 'Chat' },
  { id: 'dashboard', label: 'Dashboard' },
]

export default function NavBar({ view, onNavigate }) {
  return (
    <nav className="view-tabs" aria-label="Primary">
      {NAV_ITEMS.map((item) => (
        <button
          key={item.id}
          type="button"
          className={`view-tab${view === item.id ? ' active' : ''}`}
          aria-current={view === item.id ? 'page' : undefined}
          onClick={() => onNavigate(item.id)}
        >
          {item.label}
        </button>
      ))}
    </nav>
  )
}
