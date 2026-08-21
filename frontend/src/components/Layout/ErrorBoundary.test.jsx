import React from 'react'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import ErrorBoundary from './ErrorBoundary.jsx'

function Boom() {
  throw new Error('kaboom')
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

describe('ErrorBoundary', () => {
  it('renders children when nothing throws', async () => {
    await act(async () => {
      root.render(
        <ErrorBoundary>
          <p>healthy content</p>
        </ErrorBoundary>,
      )
    })
    expect(container.textContent).toContain('healthy content')
    expect(container.querySelector('.app-error-boundary')).toBeNull()
  })

  it('shows a themed fallback when a child throws', async () => {
    await act(async () => {
      root.render(
        <ErrorBoundary>
          <Boom />
        </ErrorBoundary>,
      )
    })
    expect(container.querySelector('.app-error-boundary')).toBeTruthy()
    expect(container.textContent).toContain('Something went wrong')
    expect(container.textContent).toContain('kaboom')
  })
})
