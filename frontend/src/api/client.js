const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

async function parseError(res, fallback) {
  let detail = fallback
  try {
    const body = await res.json()
    if (body?.detail) detail = String(body.detail)
  } catch {
    /* non-JSON error body; keep the fallback */
  }
  return new Error(detail)
}

export async function sendMessage(message, conversationId = null) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
    }),
  })

  if (!res.ok) {
    throw await parseError(res, `Request failed (HTTP ${res.status})`)
  }

  return res.json()
}

export async function listConversations() {
  const res = await fetch(`${API_BASE}/conversations`, {
    headers: { Accept: 'application/json' },
  })
  if (!res.ok) {
    throw await parseError(res, `Failed to load conversations (HTTP ${res.status})`)
  }
  return res.json()
}

export async function getConversation(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}`, {
    headers: { Accept: 'application/json' },
  })
  if (!res.ok) {
    throw await parseError(res, `Failed to load conversation (HTTP ${res.status})`)
  }
  return res.json()
}

export async function renameConversation(conversationId, title) {
  const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ title }),
  })
  if (!res.ok) {
    throw await parseError(res, `Failed to rename conversation (HTTP ${res.status})`)
  }
  return res.json()
}

export async function deleteConversation(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}`, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  })
  if (!res.ok) {
    throw await parseError(res, `Failed to delete conversation (HTTP ${res.status})`)
  }
  return res.json()
}

export async function listAdvice(category) {
  const url = new URL(`${API_BASE}/advice`)
  if (category) url.searchParams.set('category', category)
  const res = await fetch(url, { headers: { Accept: 'application/json' } })
  if (!res.ok) {
    throw await parseError(res, `Failed to load advice (HTTP ${res.status})`)
  }
  return res.json()
}

export async function getAdvice(adviceId) {
  const res = await fetch(`${API_BASE}/advice/${encodeURIComponent(adviceId)}`, {
    headers: { Accept: 'application/json' },
  })
  if (!res.ok) {
    throw await parseError(res, `Failed to load advice (HTTP ${res.status})`)
  }
  return res.json()
}

export async function saveAdvice(adviceId) {
  const res = await fetch(`${API_BASE}/advice/${encodeURIComponent(adviceId)}/save`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  })
  if (!res.ok) {
    throw await parseError(res, `Failed to save advice (HTTP ${res.status})`)
  }
  return res.json()
}
