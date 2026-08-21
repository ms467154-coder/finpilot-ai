import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  deleteConversation,
  getAdvice,
  getConversation,
  listAdvice,
  listConversations,
  renameConversation,
  saveAdvice,
  sendMessage,
} from './client.js'

function jsonRes(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body }
}

describe('advice API client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('listAdvice fetches GET /advice and returns the items', async () => {
    const items = [{ id: 'a1', category: 'Saving' }]
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes(items))))
    const result = await listAdvice()
    expect(result).toEqual(items)
    const [url, opts] = vi.mocked(fetch).mock.calls[0]
    expect(String(url)).toBe('http://127.0.0.1:8000/advice')
    expect((opts && opts.method) || 'GET').toBe('GET')
  })

  it('listAdvice appends the category query param when given', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes([]))))
    await listAdvice('Debt')
    expect(String(vi.mocked(fetch).mock.calls[0][0])).toBe('http://127.0.0.1:8000/advice?category=Debt')
  })

  it('getAdvice fetches GET /advice/{id}', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes({ id: 'a1' }))))
    const result = await getAdvice('a1')
    expect(result.id).toBe('a1')
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe('http://127.0.0.1:8000/advice/a1')
  })

  it('saveAdvice POSTs to /advice/{id}/save', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes({ id: 'a1', saved: true }))))
    const result = await saveAdvice('a1')
    expect(result.saved).toBe(true)
    const [url, opts] = vi.mocked(fetch).mock.calls[0]
    expect(url).toBe('http://127.0.0.1:8000/advice/a1/save')
    expect(opts.method).toBe('POST')
  })

  it('throws the backend detail message on failure', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes({ detail: 'Advice not found' }, 404))))
    await expect(getAdvice('missing')).rejects.toThrow('Advice not found')
  })
})

describe('conversation API client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('sendMessage POSTs to /chat with an explicit conversation_id', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes({ conversation_id: 'c1' }))))
    await sendMessage('Hello', 'c1')
    const [url, opts] = vi.mocked(fetch).mock.calls[0]
    expect(String(url)).toBe('http://127.0.0.1:8000/chat')
    expect(opts.method).toBe('POST')
    expect(JSON.parse(opts.body).conversation_id).toBe('c1')
  })

  it('sendMessage omits a conversation id for a fresh chat', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes({ conversation_id: 'c1' }))))
    await sendMessage('Hello')
    const body = JSON.parse(vi.mocked(fetch).mock.calls[0][1].body)
    expect(body.conversation_id).toBeNull()
  })

  it('listConversations fetches GET /conversations', async () => {
    const items = [{ id: 'c1', title: 'Budgeting' }]
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes(items))))
    const result = await listConversations()
    expect(result).toEqual(items)
    expect(String(vi.mocked(fetch).mock.calls[0][0])).toBe('http://127.0.0.1:8000/conversations')
  })

  it('getConversation fetches GET /conversations/{id}', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes({ id: 'c1', messages: [] }))))
    const result = await getConversation('c1')
    expect(result.id).toBe('c1')
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe('http://127.0.0.1:8000/conversations/c1')
  })

  it('renameConversation PATCHes /conversations/{id} with the new title', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes({ id: 'c1', title: 'New' }))))
    const result = await renameConversation('c1', 'New')
    expect(result.title).toBe('New')
    const [url, opts] = vi.mocked(fetch).mock.calls[0]
    expect(String(url)).toBe('http://127.0.0.1:8000/conversations/c1')
    expect(opts.method).toBe('PATCH')
    expect(JSON.parse(opts.body)).toEqual({ title: 'New' })
  })

  it('deleteConversation DELETEs /conversations/{id}', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonRes({ id: 'c1', deleted: true }))))
    const result = await deleteConversation('c1')
    expect(result.deleted).toBe(true)
    const [url, opts] = vi.mocked(fetch).mock.calls[0]
    expect(String(url)).toBe('http://127.0.0.1:8000/conversations/c1')
    expect(opts.method).toBe('DELETE')
  })
})
