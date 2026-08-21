import { useRef, useState } from 'react'
import AdviceCard from './AdviceCard.jsx'

const DELETE_MS = 220

function prefersReducedMotion() {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  )
}

export default function MessageBubble({ message, onDelete }) {
  const isUser = message.role === 'user'
  const rowRef = useRef(null)
  const [removing, setRemoving] = useState(false)

  function handleDelete() {
    if (removing || !onDelete) return
    const el = rowRef.current
    if (el && !prefersReducedMotion()) {
      setRemoving(true)
      el.style.height = `${el.offsetHeight}px`
      el.style.overflow = 'hidden'
      window.requestAnimationFrame(() => {
        el.style.height = '0px'
        el.style.opacity = '0'
        el.style.transform = 'scale(0.98)'
        el.style.marginBottom = '0px'
      })
      window.setTimeout(() => onDelete(message.id), DELETE_MS)
    } else {
      onDelete(message.id)
    }
  }

  return (
    <div
      ref={rowRef}
      className={`bubble-row ${isUser ? 'user' : 'assistant'}${message.isError ? ' error' : ''}${removing ? ' removing' : ''}`}
    >
      {!isUser && (
        <div className="message-meta">
          <span className="avatar" aria-hidden="true">
            F
          </span>
          <span className="message-name">FinAdvise</span>
        </div>
      )}
      <div className={`bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
        <p className="bubble-text">{message.content}</p>
        {message.advice && <AdviceCard advice={message.advice} />}
      </div>
      <button
        type="button"
        className="message-delete"
        aria-label="Delete message"
        onClick={handleDelete}
      >
        <TrashIcon />
      </button>
    </div>
  )
}

function TrashIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 .9h8a1 1 0 0 0 1-.9L18 7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M10 11v6M14 11v6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}