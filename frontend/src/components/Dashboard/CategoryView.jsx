export default function CategoryView({ grouped }) {
  return (
    <section className="dashboard-section">
      <div className="dashboard-section-head">
        <h2>Categories</h2>
        <span className="dash-count">{grouped.length} groups</span>
      </div>
      {grouped.length === 0 ? (
        <p className="dash-empty">No advice to group yet.</p>
      ) : (
        <div className="dash-categories">
          {grouped.map(({ category, items }) => (
            <details className="dash-category" key={category} open>
              <summary className="dash-category-head">
                <span className="dash-cat">{category}</span>
                <span className="dash-count">{items.length}</span>
              </summary>
              <div className="dash-category-items">
                {items.map((item) => (
                  <div className="dash-category-row" key={item.id}>
                    <h4 className="dash-card-title">{item.short_title}</h4>
                    <p className="dash-history-key">{item.key_recommendation}</p>
                  </div>
                ))}
              </div>
            </details>
          ))}
        </div>
      )}
    </section>
  )
}
