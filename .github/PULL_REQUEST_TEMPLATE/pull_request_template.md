## Summary

Describe the change and the user or engineering problem it solves.

## Scope

- [ ] Backend
- [ ] Frontend
- [ ] Documentation
- [ ] Tests
- [ ] Configuration

## State and API impact

Describe any changes to application states, API contracts, persistence, or model behavior. If there is no impact, state that explicitly.

## Verification

- [ ] `python -m pytest tests -q`
- [ ] `cd frontend; npm test -- --run`
- [ ] `cd frontend; npm run build`
- [ ] Manual smoke test completed where relevant

## Screenshots

Include screenshots or recordings for meaningful visual changes.

## Checklist

- [ ] No secrets, private databases, model weights, or user data are committed.
- [ ] Existing loading, empty, error, retry, save, and delete states remain functional.
- [ ] Documentation was updated if setup or behavior changed.
