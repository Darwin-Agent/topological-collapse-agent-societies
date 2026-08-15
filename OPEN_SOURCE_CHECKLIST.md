# Open-Source Release Checklist

- [x] Release assembled from an explicit allowlist.
- [x] `.env`, credentials, private keys, and internal endpoints excluded.
- [x] Raw platform data, databases, identifier lists, and model traces excluded.
- [x] Absolute private filesystem paths removed from executable code and docs.
- [x] API callers require explicit environment configuration.
- [x] Aggregate results and compact figure inputs retained.
- [x] License, security policy, data boundary, and usage documentation added.
- [x] Offline secret/privacy/integrity verifier added.
- [x] Offline unit tests added.

Run before every publication or archive:

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```
