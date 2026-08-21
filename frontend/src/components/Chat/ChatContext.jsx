import { createContext, useContext, useEffect, useRef, useState } from 'react'
import {
  deleteConversation as apiDeleteConversation,
  getConversation,
  listConversations,
  renameConversation as apiRenameConversation,
  sendMessage as apiSendMessage,
} from '../../api/client.js'
import { useApiStatus } from '../Layout/ApiStatus.jsx'

const ChatContext = createContext(null)

let seq = 0
const nextId = () => `m-${Date.now()}-${seq++}`

const CONVERSATION_PARAM = 'conversation'

function readConversationFromUrl() {
  try {
    return new URL(window.location.href).searchParams.get(CONVERSATION_PARAM)
  } catch {
    return null
  }
}

function updateUrl(id) {
  try {
    const url = new URL(window.location.href)
    if (id) url.searchParams.set(CONVERSATION_PARAM, id)
    else url.searchParams.delete(CONVERSATION_PARAM)
    window.history.replaceState(null, '', url.toString())
  } catch {
    /* jsdom / limited environments; URL persistence is best-effort */
  }
}

export function ChatProvider({ children }) {
  const { reportError } = useApiStatus()
  const [activeId, setActiveId] = useState(() => readConversationFromUrl())
  const [messages, setMessages] = useState([])
  const [loadedId, setLoadedId] = useState(null)
  const [conversations, setConversations] = useState([])
  const [listLoading, setListLoading] = useState(true)
  const [listError, setListError] = useState(null)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState(null)
  const loadingIdRef = useRef(null)

  // Keep the active conversation in the URL so a refresh restores it.
  useEffect(() => {
    updateUrl(activeId)
  }, [activeId])

  async function refreshList() {
    setListError(null)
    try {
      const items = await listConversations()
      setConversations(items)
    } catch (err) {
      // Silent on the initial load so a cold backend does not spam the banner.
      setListError(err?.message || 'Could not load conversations.')
    } finally {
      setListLoading(false)
    }
  }

  async function loadConversation(id) {
    if (!id || loadingIdRef.current === id || loadedId === id) return
    loadingIdRef.current = id
    setHistoryLoading(true)
    setError(null)
    try {
      const data = await getConversation(id)
      setMessages(data.messages.map((m, i) => ({ ...m, id: nextId() })))
      setLoadedId(id)
    } catch (err) {
      const detail = err?.message || 'Could not load this conversation.'
      setError(detail)
      setMessages([])
      setLoadedId(null)
    } finally {
      loadingIdRef.current = null
      setHistoryLoading(false)
    }
  }

  async function openConversation(id) {
    if (id === activeId) return
    setActiveId(id)
    await loadConversation(id)
  }

  function newConversation() {
    setActiveId(null)
    setLoadedId(null)
    setMessages([])
    setError(null)
    setHistoryLoading(false)
  }

  async function renameConversation(id, title) {
    try {
      await apiRenameConversation(id, title)
      setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, title } : c)))
    } catch (err) {
      reportError(err?.message || 'Could not rename this conversation.')
    }
  }

  function deleteMessage(id) {
    setMessages((prev) => prev.filter((m) => m.id !== id))
  }

  async function deleteConversation(id) {
    try {
      await apiDeleteConversation(id)
      setConversations((prev) => prev.filter((c) => c.id !== id))
      if (activeId === id) {
        setActiveId(null)
        setLoadedId(null)
        setMessages([])
        setError(null)
      }
    } catch (err) {
      reportError(err?.message || 'Could not delete this conversation.')
    }
  }

  async function sendMessage(text) {
    if (!text.trim() || sending) return
    setError(null)
    const userMessage = { id: nextId(), role: 'user', content: text }
    setMessages((prev) => [...prev, userMessage])
    setSending(true)
    try {
      const data = await apiSendMessage(text, activeId)
      const assistantMessage = {
        id: nextId(),
        role: 'assistant',
        content: data.reply,
        advice: data.advice || null,
      }
      setMessages((prev) => [...prev, assistantMessage])
      if (data.conversation_id) {
        setActiveId(data.conversation_id)
        setLoadedId(data.conversation_id)
      }
      await refreshList()
    } catch (err) {
      const detail = err?.message || 'Could not reach the advisor backend.'
      reportError(detail)
      setError(detail)
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: 'assistant',
          content: `Error: ${detail}. Make sure the backend is running on port 8000 and try again.`,
          isError: true,
        },
      ])
    } finally {
      setSending(false)
    }
  }

  // Initial load: conversation list + restore a conversation from the URL.
  useEffect(() => {
    refreshList()
    const initial = readConversationFromUrl()
    if (initial) {
      setActiveId(initial)
      loadConversation(initial)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <ChatContext.Provider
      value={{
        activeId,
        conversations,
        deleteConversation,
        deleteMessage,
        error,
        historyLoading,
        listError,
        listLoading,
        loadConversation,
        messages,
        newConversation,
        openConversation,
        refreshList,
        renameConversation,
        sending,
        sendMessage,
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  return useContext(ChatContext)
}