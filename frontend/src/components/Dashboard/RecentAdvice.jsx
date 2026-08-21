import AdviceCard from './AdviceCard.jsx'

export default function RecentAdvice({ items, onSave, savingId }) {
  return (
    <section className="dashboard-section">
      <div className="dashboard-section-head">
        <h2>Recent Advice</h2>
        <span className="dash-count">{items.length} latest</span>
      </div>
      {items.length === 0 ? (
        <p className="dash-empty">No advice yet. Start a conversation in the Chat tab.</p>
      ) : (
        <div className="dash-card-grid">
          {items.map((item) => (
            <AdviceCard
              key={item.id}
              advice={item}
              onSave={onSave}
              saving={savingId === item.id}
            />
          ))}
        </div>
      )}
    </section>
  )
}
