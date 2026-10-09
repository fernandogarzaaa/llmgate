Drop optional wheels here before `docker build` (for example `costroute-0.1.0-py3-none-any.whl`,
`injectguard-0.1.0-py3-none-any.whl`). They are installed into the image if present; the image
works without them (vendored heuristic router, guardrails disabled). `make wheels` fills this
directory from sibling checkouts (`../costroute`, `../injectguard`).
