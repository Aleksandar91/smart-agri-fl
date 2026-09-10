# Flower FL server (PV-19 protocol)

Scored aggregator process (`python -m app.fl_server`). Strategies used in
the paper: FedAvg, FedProx, FedMedian, FedTrimmedAvg, Krum / MultiKrum
(Flower 1.32.1). Build from this directory:

```bash
docker build -t fl-server:pv19 .
```
