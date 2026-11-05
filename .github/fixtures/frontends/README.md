# NLPOPTNet integration fixture

KKTHN and DAEHN integration tests install their published 2.0.0 wheels from PyPI.
Their frozen copies have been retired. The integration report records the
consumer distribution and version and rejects imports from a source checkout.

The updated NLPOPTNet frontend is not published yet, so its reviewed Apache-2.0
snapshot remains test-only. It is not an OHAGI runtime dependency or wheel content.
`manifest.json` records its source/packaging SHA-256 hashes with CRLF normalized
to LF. Refresh deliberately from reviewed sources; never refresh automatically.
Preserve LICENSE and NOTICE when refreshing. Tests cover fixed and accelerated
local QPs, short training, constraints and metadata-only inference.
