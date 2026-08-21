import React from 'react'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { ApiBanner, ApiStatusProvider, LoadingIndicator, useApiStatus } from './ApiStatus.jsx'

function Trigger({ message }) {
  const { reportError } = useApiStatus()
  return (
    <button type="button" onClick={() => reportError(message)}>
      fail
    </button>
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
})

describe('ApiStatus', () => {
  it('renders no banner until an error is reported, then dismisses it', async () => {
    await act(async () => {
      root.render(
        <ApiStatusProvider>
          <Trigger message="backend unreachable" />
          <ApiBanner />
        </ApiStatusProvider>,
      )
    })

    expect(container.querySelector('.app-banner')).toBeNull()

    act(() => {
      container.querySelector('button').click()
    })
    expect(container.querySelector('.app-banner').textContent).toContain('backend unreachable')

    act(() => {
      container.querySelector('.app-banner-dismiss').click()
    })
    expect(container.querySelector('.app-banner')).toBeNull()
  })

  it('exposes a shared loading indicator with a label', async () => {
    await act(async () => {
      root.render(<LoadingIndicator label="Loading stuff…" />)
    })
    expect(container.querySelector('.loading-indicator')).toBeTruthy()
    expect(container.textContent).toContain('Loading stuff…')
  })
})
