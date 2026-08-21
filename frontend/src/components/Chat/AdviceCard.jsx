export default function AdviceCard({ advice }) {
  if (!advice) return null
  return (
    <div className="advice-card" role="note">
      <span className="advice-category">
        <span className="advice-category-mark" aria-hidden="true">
          ·
        </span>
        {advice.category}
      </span>
      <div className="advice-summary">
        <span className="advice-summary-label">Advice Summary</span>
        {advice.short_title && <p className="advice-title">{advice.short_title}</p>}
        <p className="advice-text">{advice.full_text}</p>
      </div>
      <div className="advice-takeaway">
        <span className="advice-takeaway-label">Key takeaway</span>
        <p className="advice-takeaway-text">{advice.key_recommendation}</p>
      </div>
    </div>
  )
}