import AdviceCard from './AdviceCard.jsx'

export default function SavedAdvice({ items, savingId }) {
  return (
    <section className="dashboard-section">
      <div className="dashboard-section-head">
        <h2>Saved Advice</h2>
        <span className="dash-count">{items.length} saved</span>
      </div>
      {items.length === 0 ? (
        <p className="dash-empty">Nothing saved yet. Tap Save on any advice card.</p>
      ) : (
        <div className="dash-card-grid">
          {items.map((item) => (
            <AdviceCard key={item.id} advice={item} saving={savingId === item.id} showSave={false} />
          ))}
        </div>
      )}
    </section>
  )
}
