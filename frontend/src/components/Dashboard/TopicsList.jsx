export default function TopicsList({ topics }) {
  const max = Math.max(1, ...topics.map((t) => t.count))
  return (
    <section className="dashboard-section">
      <div className="dashboard-section-head">
        <h2>Topics Discussed</h2>
        <span className="dash-count">{topics.length} topics</span>
      </div>
      {topics.length === 0 ? (
        <p className="dash-empty">No topics yet.</p>
      ) : (
        <div className="dash-topic-cloud">
          {topics.map(({ category, count }) => {
            const weight = 0.75 + 0.85 * (count / max)
            return (
              <span
                key={category}
                className="dash-topic"
                style={{ fontSize: `${weight}em` }}
                title={`${count} item${count === 1 ? '' : 's'}`}
              >
                {category}
                <span className="dash-topic-count">{count}</span>
              </span>
            )
          })}
        </div>
      )}
    </section>
  )
}
