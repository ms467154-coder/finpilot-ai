# Contributing to FinPilot AI

Thank you for contributing to FinPilot AI. The project values clear engineering, small reviewable changes, reliable tests, and a calm product experience.

## Development workflow

Create a focused branch for each change. Keep commits small and descriptive, and avoid mixing unrelated frontend, backend, and documentation changes unless the work is intentionally cross-cutting.

Before opening a pull request, run the relevant backend tests, frontend tests, and production build. For changes that affect both services, run the full regression workflow described in [`RUNBOOK.md`](./RUNBOOK.md).

## Pull requests

A pull request should explain the problem, summarize the implementation, identify any API or data changes, and include screenshots for meaningful frontend changes. It should also mention the verification commands that were run.

## Frontend standards

Preserve the existing state behavior when changing presentation. Loading, empty, error, retry, active, saving, deleting, and validation states are part of the product contract. Keep the brand system consistent: warm cream and ivory surfaces, parchment layers, charcoal/cocoa text, and muted soft red accents. Avoid grey dashboards, rainbow status colors, decorative gradients, and invented metrics.

## Backend standards

Keep API contracts explicit and backward-compatible where possible. Validate inputs at the boundary, return useful error messages, and add regression coverage for new behavior. Never commit credentials, private databases, model weights, or user conversation data.
