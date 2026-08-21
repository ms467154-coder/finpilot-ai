export function formatDate(iso) {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function SaveButton({ id, saved, saving, onSave }) {
  if (saved) {
    return (
      <button type="button" className="dash-save-btn saved" disabled aria-label={`Saved: ${id}`}>
        Saved
      </button>
    )
  }
  return (
    <button
      type="button"
      className="dash-save-btn"
      disabled={saving}
      onClick={() => onSave?.(id)}
      aria-label={`Save advice ${id}`}
    >
      {saving ? 'Saving…' : 'Save'}
    </button>
  )
}

export default function AdviceCard({ advice, onSave, saving, showSave = true }) {
  return (
    <article className="dash-card">
      <div className="dash-card-head">
        <span className="dash-cat">{advice.category}</span>
        {showSave && <SaveButton id={advice.id} saved={advice.saved} saving={saving} onSave={onSave} />}
      </div>
      <h3 className="dash-card-title">{advice.short_title}</h3>
      <p className="dash-key">
        <strong>Key takeaway:</strong> {advice.key_recommendation}
      </p>
      {advice.source_question && (
        <p className="dash-source">Re: “{advice.source_question}”</p>
      )}
      <p className="dash-date">{formatDate(advice.timestamp)}</p>
    </article>
  )
}
