import React from 'react'

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Something went wrong.' }
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary]', error, info)
  }

  handleReset = () => {
    this.setState({ hasError: false, message: '' })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="app-error-boundary" role="alert">
          <h2>Something went wrong</h2>
          <p>{this.state.message}</p>
          <button type="button" onClick={this.handleReset}>
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
