import { useEffect, useRef, useState } from 'react'
import { useChat } from './ChatContext.jsx'
import MessageBubble from './MessageBubble.jsx'
import TypingIndicator from './TypingIndicator.jsx'
import ChatInput from './ChatInput.jsx'
import ConversationSidebar from './ConversationSidebar.jsx'
import './chat.css'

const SUGGESTIONS = [
  'How should I start paying off my debt?',
  'What size emergency fund should I aim for?',
  'How do I begin investing with a small budget?',
]

export default function ChatWindow() {
  const {
    activeId,
    conversations,
    deleteMessage,
    error,
    historyLoading,
    loadConversation,
    messages,
    sending,
    sendMessage,
  } = useChat()
  const endRef = useRef(null)
  const [collapsed, setCollapsed] = useState(
    () =>
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(max-width: 900px)').matches,
  )

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [messages, sending, historyLoading])

  const activeTitle = conversations.find((c) => c.id === activeId)?.title || 'New conversation'

  return (
    <div className={`chat-view${collapsed ? ' sidebar-collapsed' : ''}`}>
      <ConversationSidebar />

      <section className="chat-window" aria-label="Chat conversation">
        <header className="chat-header">
          <button
            type="button"
            className="sidebar-toggle"
            aria-label={collapsed ? 'Show conversations' : 'Hide conversations'}
            onClick={() => setCollapsed((current) => !current)}
          >
            <PanelsIcon />
          </button>
          <div className="chat-header-copy">
            <span className="chat-header-eyebrow">FinPilot / advisory workspace</span>
            <h2 className="chat-header-title">{activeTitle}</h2>
          </div>
          <span className="chat-header-state"><span className="status-dot" /> Context aware</span>
        </header>

        <div className="messages" role="log" aria-live="polite">
          {historyLoading ? (
            <div className="conversation-loading conversation-loading-center" role="status">
              <span className="loading-spinner" aria-hidden="true"></span>
              <span>Restoring your conversation…</span>
            </div>
          ) : messages.length === 0 && !sending ? (
            <div className="empty-state">
              <div className="empty-state-topline"><span>01</span><span>New advisory session</span></div>
              <div className="empty-mark" aria-hidden="true">F</div>
              <h2>Make the next money decision with more clarity.</h2>
              <span className="empty-state-accessible-label">How can I help with your finances today?</span>
              <p>
                Ask about saving, budgeting, investing, debt, retirement, or any financial concept.
                FinPilot turns the question into structured, educational guidance.
              </p>
              <div className="suggestions" role="list">
                {SUGGESTIONS.map((suggestion, index) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="suggestion-chip"
                    role="listitem"
                    onClick={() => sendMessage(suggestion)}
                  >
                    <span className="suggestion-index">0{index + 1}</span>
                    <span>{suggestion}</span>
                    <span className="suggestion-arrow" aria-hidden="true">↗</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-stack">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} onDelete={deleteMessage} />
              ))}
            </div>
          )}

          <TypingIndicator active={sending} />
          <div ref={endRef} />
        </div>

        {error && (
          <div className="error-banner" role="alert">
            <span className="error-banner-text">{error}</span>
            {activeId && messages.length === 0 && !historyLoading && (
              <button type="button" className="error-retry" onClick={() => loadConversation(activeId)}>
                Retry
              </button>
            )}
          </div>
        )}

        <ChatInput onSend={sendMessage} loading={sending} />
      </section>
    </div>
  )
}

function PanelsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="3" y="4" width="12" height="16" rx="2" stroke="currentColor" strokeWidth="1.8" />
      <rect x="15" y="4" width="6" height="16" rx="2" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  )
}
