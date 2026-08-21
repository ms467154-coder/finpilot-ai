import { useEffect, useState } from 'react'
import { formatDate, SaveButton } from './AdviceCard.jsx'

export default function AdviceHistory({ items, pageSize = 8, onSave, savingId }) {
  const [page, setPage] = useState(1)
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize))

  useEffect(() => {
    if (page > totalPages) setPage(totalPages)
  }, [items.length, pageSize, totalPages, page])

  const start = (page - 1) * pageSize
  const visible = items.slice(start, start + pageSize)

  return (
    <section className="dashboard-section">
      <div className="dashboard-section-head">
        <h2>Advice History</h2>
        <span className="dash-count">{items.length} items</span>
      </div>

      {items.length === 0 ? (
        <p className="dash-empty">No advice history yet.</p>
      ) : (
        <>
          <div className="dash-history">
            {visible.map((item) => (
              <div className="dash-history-row" key={item.id}>
                <span className="dash-cat">{item.category}</span>
                <div className="dash-history-body">
                  <h3 className="dash-card-title">{item.short_title}</h3>
                  <p className="dash-history-key">{item.key_recommendation}</p>
                  <p className="dash-history-meta">
                    <span className="dash-date">{formatDate(item.timestamp)}</span>
                    {item.saved && <span className="dash-badge">Saved</span>}
                  </p>
                </div>
                <SaveButton id={item.id} saved={item.saved} saving={savingId === item.id} onSave={onSave} />
              </div>
            ))}
          </div>

          <div className="dash-pager">
            <button
              type="button"
              className="dash-page-btn"
              disabled={page === 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              Prev
            </button>
            <span className="dash-pager-info">
              Page {page} of {totalPages}
            </span>
            <button
              type="button"
              className="dash-page-btn"
              disabled={page === totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              Next
            </button>
          </div>
        </>
      )}
    </section>
  )
}
