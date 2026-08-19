# Public Demo Security

The public repository uses a separate sanitized demo dataset.

## Demo release

- 114 CSV files
- 17,790 rows
- 15 anonymized page identities
- secret/privacy scan: PASS

## Sanitization

The public demo replaces or removes:

- real client domains
- real search queries
- product identifiers
- category identifiers
- organization identifiers
- exact production business values
- local machine paths where applicable

Business values are deterministically transformed so that the UI preserves realistic relationships without publishing exact private production performance.

## Excluded

The public repository must not contain:

- `.env`
- service-account JSON
- API keys
- refresh tokens
- private keys
- production database dumps
- raw private extracts
- credentials folders
- local backup folders

The production project and the public demo are intentionally separated.
