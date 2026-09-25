# Pre-locked attacks for phase 4

Lock date: 24 August 2026.

Basis: class monopoly and class ownership on the locked PV-19-capped train partitions. **The final test set was not inspected.**

The earlier pair `Tomato___healthy → Tomato___Late_blight` is not valid on PV-19: neither class entered the leaf-safe set. The three pairs below are same-crop, exist in PV-19, and cover high and lower concentration.

## Three primary label-flip pairs

1. `Apple___healthy` → `Apple___Apple_scab`  
   high concentration, field-relevant crop.
2. `Cherry___healthy` → `Cherry___Powdery_mildew`  
   high to mid concentration, field-relevant crop.
3. `Potato___healthy` → `Potato___Late_blight`  
   lower concentration and smaller train support (`n=104` capped), contrast pair.

Mean source monopoly, five seeds:

- Apple healthy: α=0.1 ≈ 0.661; α=0.5 ≈ 0.569; IID ≈ 0.209
- Cherry healthy: α=0.1 ≈ 0.762; α=0.5 ≈ 0.536; IID ≈ 0.211
- Potato healthy: α=0.1 ≈ 0.638; α=0.5 ≈ 0.523; IID ≈ 0.231

## Locked attacker role

The attacker is not fixed as `client-0`. For each seed the attacker is the client with the most images of the **source** class on that partition. Source: `partitions_summary.json`; machine-readable table: `attack_attacker_clients.json`. The final test was not inspected.

Apple healthy → Apple scab, attacker by seed:

- α=0.1: client-0, 0, 1, 2, 1
- α=0.5: client-4, 1, 2, 1, 1
- IID: client-1, 1, 1, 2, 1

Cherry healthy → Cherry powdery mildew:

- α=0.1: client-3, 3, 1, 1, 3
- α=0.5: client-4, 4, 2, 0, 3
- IID: client-3, 0, 4, 4, 2

Potato healthy → Potato late blight:

- α=0.1: client-4, 1, 4, 3, 1
- α=0.5: client-3, 3, 4, 4, 4
- IID: client-0, 0, 0, 0, 0

Seed order in each row: 101, 211, 307, 401, 503.

## Strength and algorithms (not run in phase 3)

- primary flip fraction: `1.0`;
- sensitivity: `0.25` and `0.50`;
- model-update: `s = −0.5` and `s = −1`;
- clean and attacked conditions on the same partitions;
- phase 4 aggregators: FedAvg, FedProx, median, trimmed mean, Krum/MultiKrum.

Phase 4 does not start until the clean matrix, failures, and cost are audited.

**25 August 2026:** that gate passed. Stage 4.1 runs only flip `1.0` × FedAvg/FedProx × E=1 on all locked partitions.

**26 August 2026:** stage 4.1 audited (90/90, 0 failures) and the locked test scored. Stage 4.2a, locked before those test numbers were used to choose aggregators, repeats the same flip-`1.0` attacks with robust aggregators: median, trimmed mean (`β=0.25`), Krum (`f=1`, keep=0) and MultiKrum (`f=1`, keep=`n−f=4`). Local training stays E=1. Flip 0.25/0.50, model-update, and E=5 remain 4.2b+.

**30 August 2026:** phase 5 audited (45/45) and the test scored. E=5 under attack was not run.
