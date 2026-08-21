import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ChatWindow from './ChatWindow.jsx'
import { ChatProvider } from './ChatContext.jsx'
import { ApiStatusProvider } from '../Layout/ApiStatus.jsx'

const ADVICE = {
  id: 'adv-1',
  timestamp: '2026-08-11T00:00:00Z',
  category: 'Debt',
  short_title: 'Pay off debt first',
  key_recommendation: 'Pay off high-interest debt before investing.',
  full_text:
    'Pay off high-interest debt before investing. This frees up cash flow and guarantees a return.',
  source_question: 'Should I invest or pay off debt first?',
}

function jsonRes(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body }
}

const okChatResponse = () => ({
  ok: true,
  json: async () => ({
    conversation_id: 'conv-1',
    reply: 'Pay off high-interest debt before investing.',
    advice: ADVICE,
  }),
})

const failChatResponse = () => ({
  ok: false,
  status: 500,
  json: async () => ({ detail: 'Internal server error' }),
})

function stubApi({ conversations = [], details = {}, chat = okChatResponse() } = {}) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input) => {
      const url = String(input)
      if (url.includes('/chat')) return typeof chat === 'function' ? chat() : chat
      const match = url.match(/\/conversations\/([^/?#]+)/)
      if (match) {
        const id = decodeURIComponent(match[1])
        return details[id] || jsonRes({ id, messages: [] }, 404)
      }
      if (url.includes('/conversations')) return jsonRes(conversations)
      return jsonRes([])
    }),
  )
}

function history(id, title, messages) {
  return { id, title, created_at: '2026-08-11T00:00:00Z', updated_at: '2026-08-11T00:00:00Z', messages }
}

let container
let root

beforeEach(() => {
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
  vi.unstubAllGlobals()
  window.history.replaceState(null, '', '/')
})

async function renderChat() {
  await act(async () => {
    root.render(
      <ApiStatusProvider>
        <ChatProvider>
          <ChatWindow />
        </ChatProvider>
      </ApiStatusProvider>,
    )
  })
  await flush()
}

async function flush() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 30))
  })
}

function typeAndSend(text) {
  const textarea = container.querySelector('textarea')
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype,
    'value',
  ).set
  act(() => {
    setter.call(textarea, text)
    textarea.dispatchEvent(new window.Event('input', { bubbles: true }))
  })
  act(() => {
    container.querySelector('.send-btn').click()
  })
}

function chatCalls() {
  return global.fetch.mock.calls.filter(([url]) => String(url).includes('/chat'))
}

function clickConversation(title) {
  act(() => {
    const item = [...container.querySelectorAll('.conversation-item')].find((b) =>
      b.textContent.includes(title),
    )
    item.click()
  })
}

describe('ChatWindow', () => {
  it('renders the empty state, sends a message, and shows one clean reply with one advice card', async () => {
    stubApi()
    await renderChat()
    expect(container.textContent).toContain('How can I help with your finances today?')

    typeAndSend('Should I invest or pay off debt first?')
    await flush()

    expect(container.querySelector('.user-bubble')).toBeTruthy()
    // The reply body is shown exactly once in the message bubble; nothing echoed.
    const bubbleText = container.querySelector('.assistant-bubble .bubble-text').textContent
    expect(bubbleText).toBe('Pay off high-interest debt before investing.')
    expect(container.textContent).not.toContain('Re:')
    // The structured advice card renders once inside the assistant message.
    expect(container.querySelectorAll('.advice-card')).toHaveLength(1)
    const card = container.querySelector('.advice-card')
    expect(card.textContent).toContain('Key takeaway')
    expect(card.textContent).toContain('This frees up cash flow')
    expect(card.textContent).toContain('Debt')

    const sentBody = JSON.parse(chatCalls()[0][1].body)
    expect(sentBody.message).toBe('Should I invest or pay off debt first?')
    expect(sentBody.conversation_id).toBeNull()
  })

  it('sends the conversation_id on follow-up messages', async () => {
    stubApi()
    await renderChat()
    typeAndSend('First question')
    await flush()
    typeAndSend('Follow-up')
    await flush()

    expect(chatCalls()).toHaveLength(2)
    const secondBody = JSON.parse(chatCalls()[1][1].body)
    expect(secondBody.conversation_id).toBe('conv-1')
  })

  it('shows an error bubble when the backend call fails', async () => {
    stubApi({ chat: failChatResponse() })
    await renderChat()

    typeAndSend('Hello')
    await flush()

    expect(container.textContent).toContain('Internal server error')
    expect(container.querySelector('.bubble-row.error')).toBeTruthy()
  })

  it('shows a fallback message when the fetch itself rejects', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    await renderChat()

    typeAndSend('Hello')
    await flush()

    expect(container.textContent).toContain('Failed to fetch')
    expect(container.querySelector('.bubble-row.error')).toBeTruthy()
  })

  it('starts a fresh chat with New chat without losing prior conversations', async () => {
    stubApi({
      conversations: [
        { id: 'conv-1', title: 'Debt question', updated_at: '2026-08-11T10:00:00Z', last_message: 'x' },
      ],
    })
    await renderChat()
    typeAndSend('Hello')
    await flush()
    expect(container.querySelector('.user-bubble')).toBeTruthy()

    act(() => container.querySelector('.new-chat-btn').click())
    await flush()

    expect(container.querySelector('.user-bubble')).toBeNull()
    expect(container.textContent).toContain('How can I help with your finances today?')
    expect(container.querySelector('.conversation-item')).toBeTruthy()
    expect(container.querySelector('.conversation-item.active')).toBeNull()
  })

  it('opens a saved conversation and loads its history', async () => {
    stubApi({
      conversations: [
        { id: 'conv-1', title: 'Debt question', updated_at: '2026-08-11T10:00:00Z', last_message: 'x' },
        { id: 'conv-2', title: 'Savings question', updated_at: '2026-08-11T11:00:00Z', last_message: 'y' },
      ],
      details: {
        'conv-2': jsonRes(
          history('conv-2', 'Savings question', [
            { role: 'user', content: 'How should I save for a house?' },
            { role: 'assistant', content: 'Try a separate high-yield savings account.' },
          ]),
        ),
      },
    })
    await renderChat()
    expect(container.querySelector('.user-bubble')).toBeNull()

    clickConversation('Savings question')
    await flush()

    expect(container.textContent).toContain('How should I save for a house?')
    expect(container.textContent).toContain('Try a separate high-yield savings account.')
    expect(container.querySelector('.conversation-item.active').textContent).toContain('Savings question')
  })

  it('switches between conversations without bleed-over', async () => {
    stubApi({
      conversations: [
        { id: 'conv-1', title: 'Debt question', updated_at: '2026-08-11T10:00:00Z', last_message: 'x' },
        { id: 'conv-2', title: 'Savings question', updated_at: '2026-08-11T11:00:00Z', last_message: 'y' },
      ],
      details: {
        'conv-1': jsonRes(
          history('conv-1', 'Debt question', [
            { role: 'user', content: 'Q1 debt' },
            { role: 'assistant', content: 'A1 debt answer' },
          ]),
        ),
        'conv-2': jsonRes(
          history('conv-2', 'Savings question', [
            { role: 'user', content: 'Q2 savings' },
            { role: 'assistant', content: 'A2 savings answer' },
          ]),
        ),
      },
    })
    await renderChat()

    clickConversation('Debt question')
    await flush()
    expect(container.textContent).toContain('A1 debt answer')

    clickConversation('Savings question')
    await flush()
    expect(container.textContent).toContain('Q2 savings')
    expect(container.textContent).toContain('A2 savings answer')
    expect(container.textContent).not.toContain('A1 debt answer')
    expect(container.textContent).not.toContain('Q1 debt')
  })

  it('restores the active conversation on refresh from the URL', async () => {
    stubApi({
      conversations: [
        { id: 'conv-1', title: 'Debt question', updated_at: '2026-08-11T10:00:00Z', last_message: 'x' },
      ],
      details: {
        'conv-1': jsonRes(
          history('conv-1', 'Debt question', [
            { role: 'user', content: 'Q1 debt' },
            { role: 'assistant', content: 'A1 debt answer' },
          ]),
        ),
      },
    })
    window.history.replaceState(null, '', '/?conversation=conv-1')
    await renderChat()

    expect(container.textContent).toContain('Q1 debt')
    expect(container.textContent).toContain('A1 debt answer')
    expect(container.querySelector('.conversation-item.active').textContent).toContain('Debt question')
  })

  it('sends a suggested question from the empty state', async () => {
    stubApi()
    await renderChat()

    const chip = container.querySelector('.suggestion-chip')
    expect(chip).toBeTruthy()
    act(() => chip.click())
    await flush()

    const sentBody = JSON.parse(chatCalls()[0][1].body)
    expect(sentBody.message).toBe('How should I start paying off my debt?')
    expect(container.querySelector('.user-bubble')).toBeTruthy()
  })

  it('collapses and expands the conversation sidebar', async () => {
    stubApi()
    await renderChat()

    expect(container.querySelector('.chat-view.sidebar-collapsed')).toBeNull()

    act(() => container.querySelector('.sidebar-toggle').click())
    await flush()
    expect(container.querySelector('.chat-view.sidebar-collapsed')).toBeTruthy()

    act(() => container.querySelector('.sidebar-toggle').click())
    await flush()
    expect(container.querySelector('.chat-view.sidebar-collapsed')).toBeNull()
  })

  it('shows the AI typing indicator while waiting and fades it out when the reply arrives', async () => {
    let resolveChat
    stubApi({
      chat: () =>
        new Promise((resolve) => {
          resolveChat = resolve
        }),
    })
    await renderChat()

    typeAndSend('Hello')
    await flush()

    const typing = container.querySelector('.typing')
    expect(typing).toBeTruthy()
    expect(typing.querySelector('.avatar')).toBeTruthy()
    expect(typing.querySelectorAll('.typing-dots .dot')).toHaveLength(3)
    expect(container.textContent).toContain('Thinking')

    await act(async () => {
      resolveChat(okChatResponse())
    })
    await flush()

    // Leaves smoothly: still mounted while fading out, then unmounts.
    expect(container.querySelector('.typing.typing-leaving')).toBeTruthy()
    expect(container.querySelector('.assistant-bubble')).toBeTruthy()

    await act(async () => {
      await new Promise((r) => setTimeout(r, 260))
    })
    expect(container.querySelector('.typing')).toBeNull()
  })

  it('animates message removal and keeps the remaining messages', async () => {
    stubApi()
    await renderChat()
    typeAndSend('Hello')
    await flush()

    expect(container.querySelectorAll('.user-bubble')).toHaveLength(1)
    expect(container.querySelectorAll('.assistant-bubble')).toHaveLength(1)

    const deleteButtons = container.querySelectorAll('.message-delete')
    expect(deleteButtons).toHaveLength(2)

    act(() => deleteButtons[0].click())
    expect(container.querySelector('.bubble-row.removing')).toBeTruthy()

    await act(async () => {
      await new Promise((r) => setTimeout(r, 260))
    })

    expect(container.querySelectorAll('.user-bubble')).toHaveLength(0)
    expect(container.querySelectorAll('.assistant-bubble')).toHaveLength(1)
    expect(container.textContent).toContain('Pay off high-interest debt before investing.')
  })

  it('clears the active conversation when it is deleted from the sidebar', async () => {
    stubApi({
      conversations: [
        { id: 'conv-1', title: 'Debt question', updated_at: '2026-08-11T10:00:00Z', last_message: 'x' },
      ],
      details: {
        'conv-1': jsonRes(
          history('conv-1', 'Debt question', [
            { role: 'user', content: 'Q1 debt' },
            { role: 'assistant', content: 'A1 debt answer' },
          ]),
        ),
      },
    })
    await renderChat()

    clickConversation('Debt question')
    await flush()
    expect(container.textContent).toContain('A1 debt answer')

    act(() => container.querySelector('.item-action.delete-action').click())
    await flush()
    act(() => container.querySelector('.item-action.delete-action').click())
    await flush()

    expect(container.querySelector('.user-bubble')).toBeNull()
    expect(container.querySelector('.assistant-bubble')).toBeNull()
    expect(container.textContent).toContain('How can I help with your finances today?')
  })
})

describe('ChatWindow conversation persistence', () => {
  function NavHarness() {
    const [showChat, setShowChat] = useState(true)
    return (
      <div>
        <button type="button" className="toggle" onClick={() => setShowChat((s) => !s)}>
          toggle
        </button>
        {showChat && <ChatWindow />}
      </div>
    )
  }

  it('keeps the conversation when the chat view unmounts and remounts', async () => {
    stubApi()
    await act(async () => {
      root.render(
        <ApiStatusProvider>
          <ChatProvider>
            <NavHarness />
          </ChatProvider>
        </ApiStatusProvider>,
      )
    })
    await flush()

    typeAndSend('Hello')
    await flush()
    expect(container.querySelector('.user-bubble')).toBeTruthy()

    act(() => container.querySelector('.toggle').click())
    await flush()
    expect(container.querySelector('.user-bubble')).toBeNull()

    act(() => container.querySelector('.toggle').click())
    await flush()
    expect(container.querySelector('.user-bubble')).toBeTruthy()
    expect(container.textContent).toContain('Pay off high-interest debt before investing.')
  })
})