import { useState } from 'react'
import { useChat } from './ChatContext.jsx'

function formatWhen(iso) {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function matchesQuery(conversation, query) {
  const q = query.trim().toLowerCase()
  if (!q) return true
  return (
    (conversation.title || '').toLowerCase().includes(q) ||
    (conversation.last_message || '').toLowerCase().includes(q)
  )
}

export default function ConversationSidebar() {
  const {
    activeId,
    conversations,
    deleteConversation,
    historyLoading,
    listError,
    listLoading,
    newConversation,
    openConversation,
    refreshList,
    renameConversation,
  } = useChat()
  const [query, setQuery] = useState('')
  const [editingId, setEditingId] = useState(null)
  const [draftTitle, setDraftTitle] = useState('')
  const [pendingDeleteId, setPendingDeleteId] = useState(null)

  const filtered = conversations.filter((c) => matchesQuery(c, query))

  function startRename(conversation) {
    setPendingDeleteId(null)
    setEditingId(conversation.id)
    setDraftTitle(conversation.title || '')
  }

  function commitRename(conversation) {
    const title = draftTitle.trim()
    setEditingId(null)
    setPendingDeleteId(null)
    if (title && title !== conversation.title) renameConversation(conversation.id, title)
  }

  function handleDeleteRequest(id) {
    if (pendingDeleteId === id) {
      setPendingDeleteId(null)
      deleteConversation(id)
    } else {
      setEditingId(null)
      setPendingDeleteId(id)
    }
  }

  return (
    <aside className="conversation-sidebar" aria-label="Conversations">
      <div className="sidebar-head">
        <button type="button" className="new-chat-btn" onClick={newConversation}>
          <span className="new-chat-icon" aria-hidden="true">
            <PlusIcon />
          </span>
          <span>New chat</span>
        </button>
      </div>

      <div className="sidebar-search">
        <span className="search-icon" aria-hidden="true">
          <SearchIcon />
        </span>
        <input
          type="text"
          className="search-input"
          placeholder="Search conversations"
          aria-label="Search conversations"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {query && (
          <button
            type="button"
            className="search-clear"
            aria-label="Clear search"
            onClick={() => setQuery('')}
          >
            ✕
          </button>
        )}
      </div>

      <div className="sidebar-list-wrap">
        <div className="sidebar-list-label">Recent</div>

        {listLoading ? (
          <div className="conversation-loading" role="status">
            <span className="loading-spinner" aria-hidden="true"></span>
            <span>Loading…</span>
          </div>
        ) : listError ? (
          <div className="conversation-list-error" role="alert">
            <span>{listError}</span>
            <button type="button" className="conversation-list-retry" onClick={refreshList}>
              Retry
            </button>
          </div>
        ) : filtered.length === 0 ? (
          <p className="conversation-empty">
            {conversations.length === 0
              ? 'No conversations yet.'
              : 'No conversations match your search.'}
          </p>
        ) : (
          filtered.map((conversation) => {
            const active = conversation.id === activeId
            const editing = editingId === conversation.id
            return (
              <div
                key={conversation.id}
                className={`conversation-item${active ? ' active' : ''}`}
                role="button"
                tabIndex={0}
                aria-current={active ? 'true' : undefined}
                onClick={() => {
                  setPendingDeleteId(null)
                  openConversation(conversation.id)
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    setPendingDeleteId(null)
                    openConversation(conversation.id)
                  }
                }}
              >
                <div className="conversation-item-main">
                  {editing ? (
                    <div className="item-edit">
                      <input
                        type="text"
                        className="item-edit-input"
                        aria-label="Conversation title"
                        value={draftTitle}
                        autoFocus
                        onChange={(e) => setDraftTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') commitRename(conversation)
                          if (e.key === 'Escape') setEditingId(null)
                          e.stopPropagation()
                        }}
                      />
                      <button type="button" className="edit-confirm" aria-label="Save title" onClick={() => commitRename(conversation)}>
                        <CheckIcon />
                      </button>
                      <button type="button" className="edit-cancel" aria-label="Cancel rename" onClick={() => setEditingId(null)}>
                        <CloseIcon />
                      </button>
                    </div>
                  ) : (
                    <>
                      <span className="conversation-item-title">
                        {conversation.title || 'Untitled conversation'}
                      </span>
                      <span className="conversation-item-time">
                        {formatWhen(conversation.updated_at)}
                      </span>
                    </>
                  )}
                </div>

                {!editing && (
                  <div className="item-actions">
                    <button
                      type="button"
                      className="item-action"
                      aria-label="Rename conversation"
                      onClick={(e) => {
                        e.stopPropagation()
                        startRename(conversation)
                      }}
                    >
                      <PencilIcon />
                    </button>
                    <button
                      type="button"
                      className={`item-action delete-action${pendingDeleteId === conversation.id ? ' confirm' : ''}`}
                      aria-label="Delete conversation"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDeleteRequest(conversation.id)
                      }}
                    >
                      {pendingDeleteId === conversation.id ? (
                        <span className="delete-confirm-label">Delete?</span>
                      ) : (
                        <TrashIcon />
                      )}
                    </button>
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </aside>
  )
}

function PlusIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  )
}

function SearchIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
      <path d="m20 20-3.5-3.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

function PencilIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 20h4l10-10-4-4L4 16v4Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path d="m13 7 4 4" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  )
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 .9h8a1 1 0 0 0 1-.9L18 7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M10 11v6M14 11v6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="m5 13 4 4L19 7" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function CloseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  )
}