import { useEffect, useState } from 'react'

const LEAVE_MS = 220

export default function TypingIndicator({ active }) {
  const [visible, setVisible] = useState(active)

  useEffect(() => {
    if (active) {
      setVisible(true)
      return undefined
    }
    const timer = window.setTimeout(() => setVisible(false), LEAVE_MS)
    return () => window.clearTimeout(timer)
  }, [active])

  if (!visible) return null

  return (
    <div
      className={`typing${active ? '' : ' typing-leaving'}`}
      role="status"
      aria-label="FinAdvise is thinking"
    >
      <span className="avatar avatar-sm" aria-hidden="true">
        F
      </span>
      <span className="typing-dots" aria-hidden="true">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </span>
      <span className="typing-label">Thinking…</span>
    </div>
  )
}