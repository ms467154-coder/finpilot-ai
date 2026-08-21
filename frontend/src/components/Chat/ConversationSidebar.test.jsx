import React from 'react'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ConversationSidebar from './ConversationSidebar.jsx'
import { ChatProvider } from './ChatContext.jsx'
import { ApiStatusProvider } from '../Layout/ApiStatus.jsx'

function jsonRes(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body }
}

const CONVERSATIONS = [
  {
    id: 'conv-1',
    title: 'Debt question',
    updated_at: '2026-08-11T10:00:00Z',
    last_message: 'Pay off credit cards first',
  },
  {
    id: 'conv-2',
    title: 'Savings goal',
    updated_at: '2026-08-11T11:00:00Z',
    last_message: 'Build a three month emergency fund',
  },
]

function makeStub(conversations) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input, opts = {}) => {
      const url = String(input)
      const method = (opts && opts.method) || 'GET'
      const match = url.match(/\/conversations\/([^/?#]+)/)
      const id = match ? decodeURIComponent(match[1]) : null
      if (id && method === 'PATCH') {
        const title = JSON.parse(opts.body).title
        const found = conversations.find((c) => c.id === id)
        if (found) found.title = title
        return jsonRes({ id, title })
      }
      if (id && method === 'DELETE') {
        const idx = conversations.findIndex((c) => c.id === id)
        if (idx !== -1) conversations.splice(idx, 1)
        return jsonRes({ id, deleted: true })
      }
      if (url.includes('/conversations')) return jsonRes([...conversations])
      return jsonRes([])
    }),
  )
}

function freshConversations() {
  return CONVERSATIONS.map((c) => ({ ...c }))
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

async function renderSidebar() {
  await act(async () => {
    root.render(
      <ApiStatusProvider>
        <ChatProvider>
          <ConversationSidebar />
        </ChatProvider>
      </ApiStatusProvider>,
    )
  })
  await act(async () => {
    await new Promise((r) => setTimeout(r, 30))
  })
}

async function flush() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 30))
  })
}

function items() {
  return [...container.querySelectorAll('.conversation-item')]
}

function findItem(title) {
  return items().find((item) => item.textContent.includes(title))
}

describe('ConversationSidebar', () => {
  it('shows an empty state when there are no conversations', async () => {
    makeStub([])
    await renderSidebar()
    expect(container.textContent).toContain('No conversations yet.')
  })

  it('filters conversations by title and message content', async () => {
    makeStub(CONVERSATIONS)
    await renderSidebar()
    expect(items()).toHaveLength(2)

    setInputValue(container.querySelector('.search-input'), 'sav')
    await flush()
    expect(items()).toHaveLength(1)
    expect(items()[0].textContent).toContain('Savings goal')

    setInputValue(container.querySelector('.search-input'), 'credit cards')
    await flush()
    expect(items()).toHaveLength(1)
    expect(items()[0].textContent).toContain('Debt question')

    setInputValue(container.querySelector('.search-input'), 'zzz')
    await flush()
    expect(items()).toHaveLength(0)
    expect(container.textContent).toContain('No conversations match your search.')

    setInputValue(container.querySelector('.search-input'), '')
    await flush()
    expect(items()).toHaveLength(2)
  })

  it('renames a conversation inline and persists it on the next load', async () => {
    makeStub(freshConversations())
    await renderSidebar()

    const item = findItem('Debt question')
    act(() => {
      item.querySelector('.item-action[aria-label="Rename conversation"]').click()
    })
    await flush()

    const input = container.querySelector('.item-edit-input')
    expect(input).toBeTruthy()
    expect(input.value).toBe('Debt question')

    setInputValue(input, 'My retirement plan')
    act(() => container.querySelector('.edit-confirm').click())
    await flush()

    expect(items()[0].textContent).toContain('My retirement plan')
    const patchCall = global.fetch.mock.calls.find(
      ([url, opts]) => String(url).includes('/conversations/') && opts && opts.method === 'PATCH',
    )
    expect(JSON.parse(patchCall[1].body)).toEqual({ title: 'My retirement plan' })

    // Simulate a refresh: fresh provider fetches the (renamed) list again.
    act(() => root.unmount())
    container.textContent = ''
    root = createRoot(container)
    await renderSidebar()
    expect(items()[0].textContent).toContain('My retirement plan')
  })

  it('cancels rename with Escape without a network call', async () => {
    makeStub(freshConversations())
    await renderSidebar()

    const item = findItem('Debt question')
    act(() => {
      item.querySelector('.item-action[aria-label="Rename conversation"]').click()
    })
    await flush()

    const input = container.querySelector('.item-edit-input')
    setInputValue(input, 'Discarded title')
    act(() => {
      input.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
    })
    await flush()

    expect(container.querySelector('.item-edit-input')).toBeNull()
    expect(findItem('Debt question')).toBeTruthy()
    const patchCalls = global.fetch.mock.calls.filter(
      ([url, opts]) => String(url).includes('/conversations/') && opts && opts.method === 'PATCH',
    )
    expect(patchCalls).toHaveLength(0)
  })

  it('deletes a conversation after a two-step confirmation', async () => {
    makeStub(freshConversations())
    await renderSidebar()
    expect(items()).toHaveLength(2)

    const item = findItem('Debt question')
    const delButton = () => item.querySelector('.item-action.delete-action')

    act(() => delButton().click())
    await flush()
    expect(delButton().textContent).toContain('Delete?')
    expect(items()).toHaveLength(2)

    act(() => delButton().click())
    await flush()

    expect(items()).toHaveLength(1)
    expect(items()[0].textContent).toContain('Savings goal')
    const delCall = global.fetch.mock.calls.find(
      ([url, opts]) => String(url).includes('/conversations/conv-1') && opts && opts.method === 'DELETE',
    )
    expect(delCall).toBeTruthy()
  })
})