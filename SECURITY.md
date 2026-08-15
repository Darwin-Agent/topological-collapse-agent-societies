# Security Policy

## Credentials

This repository must never contain live credentials, private endpoints, access
tokens, cookies, authorization headers, or private-key material.

Use environment variables or an untracked `.env` file. Before publishing a
change, run:

```bash
python scripts/verify_release.py
```

If a credential is ever committed, remove it from the repository, rotate it at
the provider, and purge it from Git history. Deleting only the visible line is
not sufficient.

## Reporting

Report a suspected credential, privacy, or data-release issue privately to the
repository maintainers. Do not include the secret itself in a public issue.
