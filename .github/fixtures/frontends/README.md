# Frozen frontend integration fixtures

These Apache-2.0 frontend adapters are test-only snapshots from the local HardNet
stack, included with the maintainer's approval. They are not part of OHAGI's
installed package or runtime dependencies. PURIN checks out this fixture suite
from an immutable OHAGI commit and installs its own candidate wheel.

`manifest.json` records the SHA-256 of every frozen source and packaging file,
with CRLF normalized to LF so Windows and Linux checkouts agree.
Refresh deliberately from reviewed frontend sources; never refresh automatically
in CI. Preserve each package's LICENSE and NOTICE. The integration job checks
real training, differential processing, local QPs and metadata-only inference.
