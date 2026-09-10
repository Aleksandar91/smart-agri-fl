# Phase 5 — locked confirmatory slice (PV-19-full)

Lock date: 28 August 2026.

**Status: 30 August 2026 — training 45/45, audit 0 issues, locked test scored** (`phase5_full_test_summary.json`).

Basis: capped phases 3, 4.1, and 4.2a. Test numbers were not used to choose algorithms.

Only key patterns are confirmed, not the full matrix.

## Dataset and partitions

- Dataset: `pv19-full-774007483a1d` (20,597 images, same 19 classes).
- Partitions: `docs/experiment_protocol/partitions/pv19-full/{a01,a05,iid}/seed-{101,211,307,401,503}`
- Same seeds and α as capped. The apple-pair attacker from `attack_attacker_clients.json` is **not** automatically the same on full partitions; the attacker is computed from the full `partitions_summary.json` (max `Apple___healthy` count, tie: smallest index).

## Jobs (45, E=1, 10 rounds, 5 clients)

Clean:

- α=0.1 FedAvg; α=0.1 FedProx; IID FedAvg × 5 seeds (15)

Label-flip 1.0, apple pair:

- FedAvg at α=0.1, 0.5, IID × 5 seeds (15)

Robust aggregation, same apple flip:

- median at α=0.1, 0.5, IID × 5 seeds (15)

This covers: FedProx on strong non-IID, the apple visibility gap, and median as the cheapest geometric aggregator from capped 4.2a.

Not included: the full 4.2b matrix on full, Krum, model-update, E=5, cherry/potato.
