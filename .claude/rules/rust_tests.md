---
paths:
  - "**/tests/*.rs"
---

# Rust Integration Tests Standards

- test files must be named after what is being tested in a readable, descriptive form
- tests themselves need to named after what is being tested, and after the intention behind the test as well
- `tests/` directory must not contain any test fixtures, or any test helpers (private modules used in a single test are allowed)
- if you need a test library or a fixture shared between multiple integration tests, place it in `src/`
- if there are no libraries needed, and `src/` is empty, remove the sibling crate, move the tests into the base crate

