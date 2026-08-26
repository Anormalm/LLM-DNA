# Contributing

Contributions are welcome, especially new discovery sources, probe packs, fingerprint backends, and reproducible drift cases.

1. Create a focused branch.
2. Add or update offline tests.
3. Run `python -m unittest discover -s tests -v`.
4. Do not commit API keys, raw private prompts, generated databases, or provider-retained user data.
5. In reports and UI, describe matches as behavioral proximity rather than proven identity or lineage.

Keep network access injectable so every feature can be tested without paid requests.

