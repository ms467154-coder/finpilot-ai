import React from 'react'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App.jsx'

function jsonRes(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body }
}

let container
let root

beforeEach(() => {
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes([]))))
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
  vi.unstubAllGlobals()
})

function clickTab(label) {
  act(() => {
    const tab = [...container.querySelectorAll('.view-tab')].find((b) => b.textContent === label)
    tab.click()
  })
}

async function flush() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 20))
  })
}

describe('App navigation and shared layout', () => {
  it('starts on Chat and switches to Dashboard and back', async () => {
    await act(async () => {
      root.render(<App />)
    })

    expect(container.querySelector('.chat-window')).toBeTruthy()
    expect(container.querySelector('.dashboard-page')).toBeNull()

    clickTab('Dashboard')
    await flush()
    expect(container.querySelector('.dashboard-page')).toBeTruthy()
    expect(container.querySelector('.dashboard-title').textContent).toBe('Advice Dashboard')
    expect(container.querySelector('.chat-window')).toBeNull()

    clickTab('Chat')
    expect(container.querySelector('.chat-window')).toBeTruthy()
    expect(container.querySelector('.dashboard-page')).toBeNull()
  })

  it('marks the active tab with the active class', async () => {
    await act(async () => {
      root.render(<App />)
    })

    expect(container.querySelector('.view-tab.active').textContent).toBe('Chat')

    clickTab('Dashboard')
    await flush()
    expect(container.querySelector('.view-tab.active').textContent).toBe('Dashboard')
  })

  it('keeps the shared header visible across views', async () => {
    await act(async () => {
      root.render(<App />)
    })

    expect(container.querySelector('.app-header')).toBeTruthy()
    expect(container.querySelector('.brand-mark').textContent).toBe('F')

    clickTab('Dashboard')
    await flush()
    expect(container.querySelector('.app-header')).toBeTruthy()
    expect(container.querySelector('.brand-mark').textContent).toBe('F')
  })

  it('shows a global error banner when an API call fails and dismisses it', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    await act(async () => {
      root.render(<App />)
    })

    expect(container.querySelector('.app-banner')).toBeNull()

    clickTab('Dashboard')
    await flush()

    expect(container.querySelector('.app-banner')).toBeTruthy()
    expect(container.textContent).toContain('Failed to fetch')

    act(() => {
      container.querySelector('.app-banner-dismiss').click()
    })
    expect(container.querySelector('.app-banner')).toBeNull()
  })

  it('keeps the active conversation when navigating to Dashboard and back', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input) => {
        const url = String(input)
        if (url.includes('/chat')) {
          return jsonRes({
            conversation_id: 'conv-1',
            reply: 'Pay off high-interest debt before investing.',
            advice: null,
          })
        }
        if (url.includes('/conversations')) return jsonRes([])
        return jsonRes([])
      }),
    )
    await act(async () => {
      root.render(<App />)
    })

    typeAndSend('Should I pay off debt?')
    await flush()
    expect(container.querySelector('.user-bubble')).toBeTruthy()

    clickTab('Dashboard')
    await flush()
    expect(container.querySelector('.dashboard-page')).toBeTruthy()

    clickTab('Chat')
    await flush()
    expect(container.querySelector('.chat-window')).toBeTruthy()
    expect(container.querySelector('.user-bubble')).toBeTruthy()
    expect(container.textContent).toContain('Pay off high-interest debt before investing.')
  })

  it('shows recent conversations on the Dashboard and keeps rename/delete consistent with Chat', async () => {
    const serverConversations = [
      {
        id: 'c1',
        title: 'Debt question',
        updated_at: '2026-08-11T10:00:00Z',
        last_message: 'How should I pay off debt?',
      },
    ]
    const details = {
      c1: {
        id: 'c1',
        title: 'Debt question',
        updated_at: '2026-08-11T10:00:00Z',
        messages: [
          { role: 'user', content: 'How should I pay off debt?' },
          { role: 'assistant', content: 'Pay off high-interest debt before investing.' },
        ],
      },
    }

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input, init) => {
        const url = String(input)
        const method = (init && init.method) || 'GET'
        const match = url.match(/\/conversations\/([^/?#]+)/)
        const id = match ? decodeURIComponent(match[1]) : null
        if (id && method === 'PATCH') {
          const title = JSON.parse(init.body).title
          const found = serverConversations.find((c) => c.id === id)
          if (found) found.title = title
          return jsonRes({ id, title })
        }
        if (id && method === 'DELETE') {
          const idx = serverConversations.findIndex((c) => c.id === id)
          if (idx !== -1) serverConversations.splice(idx, 1)
          return jsonRes({ id, deleted: true })
        }
        if (id && method === 'GET') {
          return jsonRes(details[id] ?? { id, messages: [] })
        }
        if (url.includes('/chat')) {
          return jsonRes({ conversation_id: 'c1', reply: 'x', advice: null })
        }
        if (url.includes('/conversations')) return jsonRes([...serverConversations])
        if (url.includes('/advice')) return jsonRes([])
        return jsonRes([])
      }),
    )

    await act(async () => {
      root.render(<App />)
    })
    await flush()

    // Open c1 from the Chat sidebar.
    act(() => {
      const item = [...container.querySelectorAll('.conversation-item')].find((i) =>
        i.textContent.includes('Debt question'),
      )
      item.click()
    })
    await flush()
    expect(container.textContent).toContain('Pay off high-interest debt before investing.')

    // The Dashboard surfaces the recent conversation.
    clickTab('Dashboard')
    await flush()
    expect(container.textContent).toContain('Recent Conversations')
    expect(container.textContent).toContain('Debt question')

    // Renaming it in the Chat sidebar is reflected on the Dashboard.
    clickTab('Chat')
    await flush()
    act(() => {
      const item = [...container.querySelectorAll('.conversation-item')].find((i) =>
        i.textContent.includes('Debt question'),
      )
      item.querySelector('.item-action[aria-label="Rename conversation"]').click()
    })
    await flush()
    setInputValue(container.querySelector('.item-edit-input'), 'Debt plan')
    act(() => container.querySelector('.edit-confirm').click())
    await flush()

    clickTab('Dashboard')
    await flush()
    expect(container.textContent).toContain('Debt plan')
    expect(container.textContent).not.toContain('Debt question')

    // Deleting it in Chat removes it from the Dashboard too.
    clickTab('Chat')
    await flush()
    act(() => {
      const item = [...container.querySelectorAll('.conversation-item')].find((i) =>
        i.textContent.includes('Debt plan'),
      )
      item.querySelector('.item-action.delete-action').click()
    })
    await flush()
    act(() => {
      const item = [...container.querySelectorAll('.conversation-item')].find((i) =>
        i.textContent.includes('Debt plan'),
      )
      item.querySelector('.item-action.delete-action').click()
    })
    await flush()
    expect(container.querySelectorAll('.conversation-item')).toHaveLength(0)

    clickTab('Dashboard')
    await flush()
    expect(container.textContent).toContain('No conversations yet')
  })

  it('opens a conversation from the Dashboard recent list into the Chat view', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input) => {
        const url = String(input)
        if (/\/conversations\/c1$/.test(url)) {
          return jsonRes({
            id: 'c1',
            title: 'Debt question',
            messages: [
              { role: 'user', content: 'How should I pay off debt?' },
              { role: 'assistant', content: 'Pay off high-interest debt before investing.' },
            ],
          })
        }
        if (url.includes('/conversations')) {
          return jsonRes([
            {
              id: 'c1',
              title: 'Debt question',
              updated_at: '2026-08-11T10:00:00Z',
              last_message: 'How should I pay off debt?',
            },
          ])
        }
        if (url.includes('/advice')) return jsonRes([])
        return jsonRes([])
      }),
    )

    await act(async () => {
      root.render(<App />)
    })
    await flush()

    clickTab('Dashboard')
    await flush()
    const openButton = [...container.querySelectorAll('.recent-conversation')].find((b) =>
      b.textContent.includes('Debt question'),
    )
    act(() => openButton.click())
    await flush()

    expect(container.querySelector('.chat-window')).toBeTruthy()
    expect(container.querySelector('.view-tab.active').textContent).toBe('Chat')
    await flush()
    expect(container.textContent).toContain('Pay off high-interest debt before investing.')
  })
})

function typeAndSend(text) {
  const textarea = document.querySelector('textarea')
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype,
    'value',
  ).set
  act(() => {
    setter.call(textarea, text)
    textarea.dispatchEvent(new window.Event('input', { bubbles: true }))
  })
  act(() => {
    document.querySelector('.send-btn').click()
  })
}

function setInputValue(el, value) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    'value',
  ).set
  act(() => {
    setter.call(el, value)
    el.dispatchEvent(new window.Event('input', { bubbles: true }))
  })
}
