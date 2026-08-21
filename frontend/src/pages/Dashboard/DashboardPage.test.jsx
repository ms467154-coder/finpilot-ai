import React from 'react'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import DashboardPage from './DashboardPage.jsx'
import { ApiStatusProvider } from '../../components/Layout/ApiStatus.jsx'
import { ChatProvider } from '../../components/Chat/ChatContext.jsx'

const ADVISORY = [
  { id: 'a1', timestamp: '2026-08-10T10:00:00Z', category: 'Debt', short_title: 'Pay off debt first', key_recommendation: 'Pay off high-interest debt before investing.', full_text: 'x', source_question: 'Invest or pay off debt?', saved: false },
  { id: 'a2', timestamp: '2026-08-10T09:00:00Z', category: 'Saving', short_title: 'Automate your savings', key_recommendation: 'Automate a transfer to savings on payday.', full_text: 'x', source_question: 'How can I save more?', saved: true },
  { id: 'a3', timestamp: '2026-08-09T08:00:00Z', category: 'Investing', short_title: 'Start with index funds', key_recommendation: 'Start with low-cost index funds.', full_text: 'x', source_question: 'How do I start investing?', saved: false },
  { id: 'a4', timestamp: '2026-08-08T07:00:00Z', category: 'Budgeting', short_title: 'Track every expense', key_recommendation: 'Track every expense for a month.', full_text: 'x', source_question: 'Budgeting tips?', saved: false },
  { id: 'a5', timestamp: '2026-08-07T06:00:00Z', category: 'Concepts', short_title: 'What is compound interest?', key_recommendation: 'Interest on interest grows your balance faster.', full_text: 'x', source_question: 'What is compound interest?', saved: false },
  { id: 'a6', timestamp: '2026-08-06T05:00:00Z', category: 'General', short_title: 'Emergency fund basics', key_recommendation: 'Keep three to six months of expenses.', full_text: 'x', source_question: 'How much should I keep liquid?', saved: false },
  { id: 'a7', timestamp: '2026-08-05T04:00:00Z', category: 'Investing', short_title: 'Rebalance your portfolio', key_recommendation: 'Rebalance your portfolio annually.', full_text: 'x', source_question: 'Portfolio rebalancing?', saved: false },
  { id: 'a8', timestamp: '2026-08-04T03:00:00Z', category: 'Debt', short_title: 'Avoid new credit card debt', key_recommendation: 'Avoid adding new credit card debt.', full_text: 'x', source_question: 'Credit card debt?', saved: false },
  { id: 'a9', timestamp: '2026-08-03T02:00:00Z', category: 'Saving', short_title: 'Cut back on subscriptions', key_recommendation: 'Cut back on unused subscriptions.', full_text: 'x', source_question: 'Easy ways to save?', saved: false },
]

const CHATS = [
  { id: 'c1', title: 'Debt convo', updated_at: '2026-08-11T10:00:00Z', last_message: 'Should I pay off debt first?' },
  { id: 'c2', title: 'Savings convo', updated_at: '2026-08-10T09:00:00Z', last_message: 'Automate a savings transfer.' },
]

function jsonRes(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body }
}

function mockBackend({ advice = [], conversations = [], details = {} } = {}) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input, init) => {
      const url = String(input)
      const method = (init && init.method) || 'GET'
      if (url.endsWith('/save') && method === 'POST') {
        const id = url.split('/').slice(-2)[0]
        return jsonRes({ id, saved: true })
      }
      const detail = url.match(/\/conversations\/([^/?#]+)/)
      if (detail) {
        const id = decodeURIComponent(detail[1])
        return jsonRes(details[id] ?? { id, messages: [] })
      }
      if (url.includes('/conversations')) {
        return jsonRes(conversations)
      }
      if (url.includes('/advice')) {
        return jsonRes(advice)
      }
      throw new Error(`Unexpected request: ${method} ${url}`)
    }),
  )
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

async function renderPage({ onNavigate = vi.fn() } = {}) {
  await act(async () => {
    root.render(
      <ApiStatusProvider>
        <ChatProvider>
          <DashboardPage onNavigate={onNavigate} />
        </ChatProvider>
      </ApiStatusProvider>,
    )
  })
  await act(async () => {
    await new Promise((r) => setTimeout(r, 20))
  })
  return onNavigate
}

async function renderDashboard({
  advice = ADVISORY,
  conversations = [],
  details = {},
  onNavigate = vi.fn(),
} = {}) {
  mockBackend({ advice, conversations, details })
  return renderPage({ onNavigate })
}

function sectionText(sectionClass) {
  const el = [...container.querySelectorAll('.dashboard-section')].find((s) =>
    s.querySelector('.dashboard-section-head h2')?.textContent.includes(sectionClass),
  )
  return el ? el.textContent : ''
}

async function flush() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 20))
  })
}

describe('DashboardPage', () => {
  it('renders all sections grouped from the fetched advice', async () => {
    await renderDashboard({ conversations: CHATS })

    expect(container.textContent).toContain('Advice Dashboard')
    expect(container.querySelector('.dash-chip').textContent).toContain('9 items')

    const recent = sectionText('Recent Advice')
    expect(recent).toContain('Pay off debt first')
    expect(recent).toContain('Automate your savings')

    expect(sectionText('Key Recommendations')).toContain('Pay off high-interest debt before investing.')

    expect(sectionText('Categories')).toContain('Investing')
    expect(sectionText('Categories')).toContain('Budgeting')

    expect(sectionText('Topics Discussed')).toContain('Debt')
    expect(sectionText('Topics Discussed')).toContain('Saving')

    expect(sectionText('Saved Advice')).toContain('Automate your savings')
  })

  it('paginates the Advice History list', async () => {
    await renderDashboard({ conversations: CHATS })

    const history = sectionText('Advice History')
    expect(history).toContain('Pay off debt first')
    expect(history).toContain('Avoid new credit card debt')
    expect(history).not.toContain('Cut back on subscriptions')
    expect(container.querySelector('.dash-pager-info').textContent).toContain('Page 1 of 2')

    act(() => {
      [...container.querySelectorAll('.dash-page-btn')].find((b) => b.textContent === 'Next').click()
    })

    expect(container.querySelector('.dash-pager-info').textContent).toContain('Page 2 of 2')
    expect(sectionText('Advice History')).toContain('Cut back on subscriptions')
  })

  it('saves an item via POST /advice/{id}/save and moves it into Saved Advice', async () => {
    await renderDashboard({ conversations: CHATS })

    const recent = [...container.querySelectorAll('.dashboard-section')].find((s) =>
      s.querySelector('.dashboard-section-head h2')?.textContent.includes('Recent Advice'),
    )
    const card = [...recent.querySelectorAll('.dash-card')].find((c) =>
      c.textContent.includes('Start with index funds'),
    )
    const saveBtn = card.querySelector('.dash-save-btn')
    expect(saveBtn.textContent).toBe('Save')

    act(() => saveBtn.click())
    await flush()

    const saveCalls = vi.mocked(fetch).mock.calls.filter(([input]) =>
      String(input).endsWith('/a3/save'),
    )
    expect(saveCalls).toHaveLength(1)
    expect(saveCalls[0][1].method).toBe('POST')

    expect(sectionText('Saved Advice')).toContain('Start with index funds')
    expect(sectionText('Saved Advice')).not.toContain('Pay off debt first')
  })

  it('shows empty states when there is no advice or conversations', async () => {
    await renderDashboard({ advice: [], conversations: [] })

    expect(container.querySelector('.dash-chip').textContent).toContain('0 items')
    expect(sectionText('Recent Advice')).toContain('No advice yet')
    expect(sectionText('Saved Advice')).toContain('Nothing saved yet')
    expect(sectionText('Topics Discussed')).toContain('No topics yet')
    expect(sectionText('Recent Conversations')).toContain('No conversations yet')
  })

  it('shows an error banner and recovers via Retry', async () => {
    let shouldFail = true
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input, init) => {
        const url = String(input)
        const method = (init && init.method) || 'GET'
        if (shouldFail) {
          throw new TypeError('Failed to fetch')
        }
        if (url.includes('/conversations')) {
          return jsonRes([])
        }
        if (url.includes('/advice') && method === 'GET') {
          return jsonRes([ADVISORY[0]])
        }
        throw new Error(`Unexpected request: ${method} ${url}`)
      }),
    )

    await renderPage()
    expect(container.querySelector('.dash-error')).toBeTruthy()
    expect(container.textContent).toContain('Failed to fetch')

    shouldFail = false
    act(() => {
      container.querySelector('.dash-retry').click()
    })
    await flush()

    expect(container.querySelector('.dash-error')).toBeNull()
    expect(container.textContent).toContain('Pay off debt first')
  })

  it('lists recent conversations from the shared conversation list', async () => {
    await renderDashboard({ conversations: CHATS })

    const section = sectionText('Recent Conversations')
    expect(section).toContain('2 chats')
    expect(section).toContain('Debt convo')
    expect(section).toContain('Savings convo')
    expect(section).toContain('Should I pay off debt first?')
  })

  it('opens a recent conversation by navigating to the Chat view', async () => {
    const onNavigate = vi.fn()
    await renderDashboard({
      conversations: [CHATS[0]],
      details: {
        c1: {
          id: 'c1',
          title: 'Debt convo',
          messages: [
            { role: 'user', content: 'Should I pay off debt first?' },
            { role: 'assistant', content: 'Pay off high-interest debt before investing.' },
          ],
        },
      },
      onNavigate,
    })

    const openButton = [...container.querySelectorAll('.recent-conversation')].find((b) =>
      b.textContent.includes('Debt convo'),
    )
    expect(openButton).toBeTruthy()
    act(() => openButton.click())
    await flush()

    expect(onNavigate).toHaveBeenCalledWith('chat')
    const detailCall = vi.mocked(fetch).mock.calls.some(([input]) =>
      String(input).includes('/conversations/c1'),
    )
    expect(detailCall).toBe(true)
  })
})