# Faza 5 — unaprijed zaključani potvrdni slice (PV-19-full)

Datum zaključavanja: 28. avgust 2026.  
**Status: 30. avgust 2026 — trening 45/45, audit 0 problema, zaključani test evaluiran** (`phase5_full_test_summary.json`).  
Osnova: capped faze 3, 4.1 i 4.2a. Test brojevi nisu korišteni za izbor algoritama.  
Raspberry Pi nije dio ovog slice-a.

Potvrđuju se samo ključni obrasci, ne cijela matrica.

## Dataset i particije

- Dataset: `pv19-full-774007483a1d` (20.597 slika, istih 19 klasa).
- Particije: `docs/experiment_protocol/partitions/pv19-full/{a01,a05,iid}/seed-{101,211,307,401,503}`
- Isti seedovi i α kao capped. Napadač za apple par iz `attack_attacker_clients.json` **nije** automatski isti na full particijama; napadač se računa iz full `partitions_summary.json` (max Apple___healthy count, tie: najmanji indeks).

## Poslovi (45, E=1, 10 rundi, 5 klijenata)

Clean:

- α=0,1 FedAvg; α=0,1 FedProx; IID FedAvg × 5 seedova (15)

Label-flip 1,0, par apple:

- FedAvg na α=0,1; 0,5; IID × 5 seedova (15)

Robusna agregacija, isti apple flip:

- mediana na α=0,1; 0,5; IID × 5 seedova (15)

Ovo pokriva: FedProx na jakom non-IID, visibility gap na apple, i median kao najmanje skup geometrijski aggregator sa capped 4.2a.

Ne ulazi: cijela 4.2b matrica na full, Krum, model-update, E=5, cherry/potato, Pi.
