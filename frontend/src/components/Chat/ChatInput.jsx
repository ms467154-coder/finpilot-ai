import { useRef, useState } from 'react'

export default function ChatInput({ onSend, loading }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  function autoGrow() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }

  function handleChange(e) {
    setValue(e.target.value)
    autoGrow()
  }

  function submit() {
    if (!value.trim() || loading) return
    onSend(value)
    setValue('')
    requestAnimationFrame(() => {
      const el = textareaRef.current
      if (el) el.style.height = 'auto'
    })
  }

  return (
    <div className="input-area">
      <div className="input-context">
        <span className="input-context-label">Financial guidance session</span>
        <span className="input-context-note">Educational information · not personal advice</span>
      </div>
      <div className="input-box">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          placeholder="Ask a financial question…"
          aria-label="Your message"
          onChange={handleChange}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
        />
        <button
          className="send-btn"
          type="button"
          onClick={submit}
          disabled={loading || !value.trim()}
          aria-label="Send message"
        >
          {loading ? <span className="spinner" aria-hidden="true"></span> : <SendIcon />}
        </button>
      </div>
      <p className="input-hint">Enter to send · Shift+Enter for a new line</p>
    </div>
  )
}

function SendIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M3 11.5 21 3l-6.4 18-3.3-6.6L3 11.5Z" fill="currentColor" />
      <path d="M11.3 14.4 21 3" stroke="#0a0a0c" strokeWidth="1.6" />
    </svg>
  )
}