import { useEffect, useMemo, useState } from 'react'
import { listAdvice, saveAdvice } from '../../api/client.js'
import { useApiStatus, LoadingIndicator } from '../../components/Layout/ApiStatus.jsx'
import { useChat } from '../../components/Chat/ChatContext.jsx'
import RecentAdvice from '../../components/Dashboard/RecentAdvice.jsx'
import RecentConversations from '../../components/Dashboard/RecentConversations.jsx'
import AdviceHistory from '../../components/Dashboard/AdviceHistory.jsx'
import CategoryView from '../../components/Dashboard/CategoryView.jsx'
import SavedAdvice from '../../components/Dashboard/SavedAdvice.jsx'
import TopicsList from '../../components/Dashboard/TopicsList.jsx'
import './dashboard.css'

const RECENT_COUNT = 4

export default function DashboardPage({ onNavigate }) {
  const [advice, setAdvice] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [savingId, setSavingId] = useState(null)
  const { reportError } = useApiStatus()
  const { conversations, listError, listLoading, openConversation, refreshList } = useChat()

  async function load() {
    setLoading(true)
    setError(null)
    try {
      setAdvice(await listAdvice())
    } catch (err) {
      const detail = err?.message || 'Could not load advice. Is the backend running?'
      reportError(detail)
      setError(detail)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
    refreshList()
  }, [])

  function openInChat(id) {
    openConversation(id)
    onNavigate?.('chat')
  }

  function handleRetry() {
    load()
    refreshList()
  }

  const sorted = useMemo(() => [...advice].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)), [advice])
  const recent = useMemo(() => sorted.slice(0, RECENT_COUNT), [sorted])
  const grouped = useMemo(() => {
    const map = new Map()
    for (const item of sorted) {
      const category = item.category || 'General'
      if (!map.has(category)) map.set(category, [])
      map.get(category).push(item)
    }
    return [...map.entries()]
      .map(([category, items]) => ({ category, items }))
      .sort((a, b) => b.items.length - a.items.length)
  }, [sorted])
  const topics = useMemo(() => grouped.map(({ category, items }) => ({ category, count: items.length })), [grouped])
  const saved = useMemo(() => sorted.filter((item) => item.saved), [sorted])

  async function handleSave(id) {
    if (savingId) return
    setSavingId(id)
    try {
      await saveAdvice(id)
      setAdvice((prev) => prev.map((item) => (item.id === id ? { ...item, saved: true } : item)))
    } catch (err) {
      const detail = err?.message || 'Could not save advice.'
      reportError(detail)
      setError(detail)
    } finally {
      setSavingId(null)
    }
  }

  return (
    <div className="dashboard-page">
      <div className="dashboard-header">
        <div className="dashboard-heading-block">
          <span className="dashboard-kicker">FinPilot intelligence / 02</span>
          <h1 className="dashboard-title">Advice Dashboard</h1>
          <p className="dashboard-subtitle">A calm, structured view of what your financial conversations are teaching you.</p>
        </div>
        <div className="dashboard-summary" aria-label="Advice summary">
          <div className="summary-lead">Your signal, organized.</div>
          <span className="dash-chip"><strong>{sorted.length}</strong> items</span>
          <span className="dash-chip"><strong>{grouped.length}</strong> categories</span>
          <span className="dash-chip"><strong>{saved.length}</strong> saved</span>
        </div>
      </div>

      {error && (
        <div className="dash-error" role="alert">
          <span>{error}</span>
          <button type="button" className="dash-retry" onClick={handleRetry}>Retry</button>
        </div>
      )}

      {loading ? (
        <LoadingIndicator label="Loading advice…" />
      ) : (
        <div className="dashboard-content">
          <section className="dashboard-feature-row" aria-label="Recent activity">
            <RecentConversations conversations={conversations} loading={listLoading} error={listError} onOpen={openInChat} onRetry={refreshList} />
            <RecentAdvice items={recent} onSave={handleSave} savingId={savingId} />
          </section>

          <section className="dashboard-section dashboard-section-wide">
            <div className="dashboard-section-head">
              <div><span className="section-index">03</span><h2>Key Recommendations</h2></div>
              <span className="dash-count">{sorted.length} highlights</span>
            </div>
            {sorted.length === 0 ? (
              <p className="dash-empty">Highlights will appear here once you have advice.</p>
            ) : (
              <div className="dash-card-grid dash-recommendations">
                {sorted.map((item) => (
                  <article className="dash-card dash-highlight" key={item.id}>
                    <span className="dash-cat">{item.category}</span>
                    <h3 className="dash-card-title">{item.short_title}</h3>
                    <p className="dash-key">{item.key_recommendation}</p>
                  </article>
                ))}
              </div>
            )}
          </section>

          <section className="dashboard-structure-grid">
            <CategoryView grouped={grouped} />
            <TopicsList topics={topics} />
          </section>

          <AdviceHistory items={sorted} onSave={handleSave} savingId={savingId} />
          <SavedAdvice items={saved} savingId={savingId} />
        </div>
      )}
    </div>
  )
}
