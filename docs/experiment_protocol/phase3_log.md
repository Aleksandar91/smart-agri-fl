# Faza 3 — dnevnik clean utility matrice

## 24. avgust 2026.

### Gate prije pokretanja

Faze 0–2 su završene. Smoke run-ovi nisu naučni. Prije prvog 10-round
run-a zaključano je:

- dataset `pv19-capped-62b5b2119fb2`;
- 15 group-safe particija (α=0,1; 0,5; IID × 5 seedova);
- 60 clean poslova: FedAvg/FedProx × E=1/E=5 × 10 rundi;
- FedProx `μ = 0,01`;
- development-validation tokom rundi, jedan evaluator po rundi;
- finalni test se **ne evaluira** dok svih 60 run-ova ne završi
  (`PV19_MATRIX_MODE=eval_test`);
- tri faza-4 label-flip para, iz monopoly mjera, bez pregleda testa.

Pi nije potreban.

### Pokretanje

Prvo se izvršava jedan kanonski posao
`pv19-capped-a01-s101-fedavg-e1` kao provjera 10-round naučnog toka.
Ako artefakti i validation evaluator prođu, pokreće se ostatak matrice.

### Kanonski posao — tehnički audit, ne izbor algoritma

`docs/experiment_protocol/runs/pv19-capped/pv19-capped-a01-s101-fedavg-e1`

- trajanje: 17:23:35–17:30:56 UTC (7 min 21 s);
- 10/10 rundi sa zaključanom validation evaluacijom;
- fit i evaluation failures: 0;
- finalni test nije evaluiran;
- development-validation accuracy raste kroz runde i u rundi 10 iznosi
  56,20% (macro-F1 49,52% na nezavisnom evaluatoru).

Ovi brojevi služe samo provjeri toka i procjeni troška. Ne koriste se za
izbor α, E, agregatora ili napada. Procjena ostatka matrice: oko 22 sata
na istom CPU Docker toku (29×E=1 + 30×E=5).

Ostatak 59 poslova pokrenut je odmah nakon ove provjere. Skripta
preskače već završene poslove, pa se može prekinuti i nastaviti.

## 25. avgust 2026.

### Stanje i prekid roditeljskog launcher-a

Do 05:34 UTC završeno je 41/60 poslova: cijeli α=0,1, cijeli α=0,5 i
`iid-s101-fedavg-e1`. Finalni test nije evaluiran ni na jednom run-u.

Roditeljski `run_pv19_utility_matrix.sh` proces se prekinuo dok je Docker
nastavio `pv19-capped-iid-s101-fedavg-e5`. Taj posao se ne prekida i ne
pokreće iznova. Nakon njegovog `docker wait` slijedi samo validation
evaluator i `run_manifest`, zatim nastavak preostalih 18 poslova istom
skriptom.

Procjena ostatka: oko 5 sati (9×E=1 + 9×E=5 + dovršetak trenutnog E=5).

Trenutni E=5 posao je završio u Dockeru (10/10 rundi, 0 failures).
Validation evaluator je pokrenut naknadno; test nije diran. Launcher je
nastavljen u 06:52 UTC i preskočio 42 završena posla. Sljedeći je
`pv19-capped-iid-s101-fedprox-e1`.

### Završetak clean matrice i audit gate

Svih 60 pre-registrovanih poslova je završeno. Artefakt audita:

`docs/experiment_protocol/runs/pv19-capped/phase3_audit.json`

Provjere:

- 60/60 imaju 10/10 rundi, `global_latest.npz` i `validation_evaluation.json`;
- ukupno fit failures: 0;
- ukupno evaluation failures: 0;
- `test_evaluation.json` datoteka: 0;
- wall-clock: 2026-08-24 17:23 UTC do 2026-08-25 12:23 UTC (~19 h).

Tipično trajanje (ne računajući dva Docker zastoja):

- E=1: srednje 6,5 min (5,5–10);
- E=5: srednje 31,4 min, ali bez dva zastoja oko 23–29 min.

Dva wall-clock odstupanja nisu extra trening:

- `a01-s101-fedprox-e5` ~100 min (noćni zastoj);
- `iid-s101-fedavg-e5` ~73 min (prekid launcher-a + pauze rundi 5 i 8).

Development-validation, samo kao sanity check toka, prati očekivani
red heterogenosti (IID > α=0,5 > α=0,1). Ovi brojevi nisu osnov za
izbor algoritma, α, E ili napada. Primarne metrike ostaju na zaključanom
testu, koji se sada tek evaluira jer je matrica kompletna.

Faza 4 i dalje čeka završetak test evaluacije i ovaj audit. Napadački
parovi ostaju oni zaključani iz particija.

### Finalni test — batch evaluacija

`PV19_MATRIX_MODE=eval_test` je završio 25. avgusta 12:33 UTC. Svih 60
checkpointova je evaluirano na zaključanom testu (850 slika). Napadački
parovi nisu mijenjani.

Sažetak primarnih test metrika (mean ± sd, 5 seedova):

- α=0,1 FedAvg E=1: accuracy 0,636 ± 0,061; macro-F1 0,591 ± 0,076
- α=0,1 FedAvg E=5: 0,702 ± 0,050; 0,665 ± 0,068
- α=0,1 FedProx E=1: 0,678 ± 0,045; 0,649 ± 0,055
- α=0,1 FedProx E=5: 0,753 ± 0,042; 0,737 ± 0,049
- α=0,5 FedAvg E=1: 0,809 ± 0,032; 0,805 ± 0,036
- α=0,5 FedAvg E=5: 0,842 ± 0,028; 0,839 ± 0,030
- α=0,5 FedProx E=1: 0,821 ± 0,021; 0,820 ± 0,023
- α=0,5 FedProx E=5: 0,852 ± 0,009; 0,851 ± 0,010
- IID FedAvg E=1: 0,861 ± 0,005; 0,861 ± 0,006
- IID FedAvg E=5: 0,879 ± 0,005; 0,878 ± 0,005
- IID FedProx E=1: 0,857 ± 0,005; 0,856 ± 0,005
- IID FedProx E=5: 0,874 ± 0,002; 0,874 ± 0,003

Detalj: `docs/experiment_protocol/runs/pv19-capped/phase3_test_summary.json`.

Faza 4 počinje na zaključanim parovima iz particija, ne na osnovu ovih
brojeva.
