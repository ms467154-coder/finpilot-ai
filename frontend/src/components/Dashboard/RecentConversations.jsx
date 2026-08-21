function formatWhen(iso) {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export default function RecentConversations({ conversations, loading, error, onOpen, onRetry, limit = 4 }) {
  const visible = conversations.slice(0, limit)

  return (
    <section className="dashboard-section">
      <div className="dashboard-section-head">
        <h2>Recent Conversations</h2>
        <span className="dash-count">{conversations.length} chats</span>
      </div>

      {error ? (
        <div className="dash-error" role="alert">
          <span>{error}</span>
          <button type="button" className="dash-retry" onClick={onRetry}>
            Retry
          </button>
        </div>
      ) : loading ? (
        <div className="dash-recent-loading" role="status">
          <span className="loading-spinner" aria-hidden="true"></span>
          <span>Loading conversations…</span>
        </div>
      ) : visible.length === 0 ? (
        <p className="dash-empty">No conversations yet. Start one in the Chat tab.</p>
      ) : (
        <div className="dash-recent">
          {visible.map((conversation) => (
            <button
              key={conversation.id}
              type="button"
              className="recent-conversation"
              onClick={() => onOpen(conversation.id)}
            >
              <span className="recent-conversation-title">
                {conversation.title || 'Untitled conversation'}
              </span>
              {conversation.last_message && (
                <span className="recent-conversation-preview">{conversation.last_message}</span>
              )}
              <span className="recent-conversation-date">{formatWhen(conversation.updated_at)}</span>
            </button>
          ))}
        </div>
      )}
    </section>
  )
}