# Flower FL client (PV-19 protocol)

Scored client for the public PV-19 artefact (24 August 2026 protocol).
Build from this directory:

```bash
docker build -t fl-client:ml .
python -m app.fl_client
```

Locked matrix jobs use 128 px inputs, batch size 16, a frozen
`model.features` block, and Flower 1.32.1. See the repository root README.
