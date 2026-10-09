# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately to **fernandogarzaaa@gmail.com** (or via GitHub's
"Report a vulnerability" on the repository) rather than in a public issue. You should get an
answer within 7 days. Please include a reproduction and the affected version.

## Supported versions

Only the latest released minor version receives security fixes.

## Security model, in brief

- Virtual API keys are random 256-bit tokens (`lg-` prefix). Only their SHA-256 hash is stored;
  the plaintext is shown once at creation.
- The admin API is disabled unless `server.admin_key` is set; it is compared in constant time.
- Upstream provider keys are read from the environment / a Kubernetes Secret and never returned
  by any endpoint (`/healthz` reports component names only).
- The exact and semantic caches are scoped per tenant by default (`cache.scope: tenant`). A
  global cache can return one tenant's answer to another tenant: only enable it for shared,
  non-sensitive workloads.
- Guardrails (injectguard) reduce but do not eliminate prompt-injection and PII-leak risk. In
  `stream_output: post_scan` mode, flagged streamed output has already reached the client;
  use `buffer` mode where that matters.
- The container runs as UID 10001 with a read-only root filesystem in the Helm chart.
