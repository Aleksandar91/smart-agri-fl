# Faza 4 — dnevnik attack/defense matrice

## 25. avgust 2026.

### Gate prije pokretanja

Faza 3 je završena: 60/60 clean run-ova, 0 failures, zaključani test
evaluiran. Napadački parovi ostaju oni zaključani 24. avgusta iz
particija, ne iz test rezultata.

### Stage 4.1 — primarni label-flip slice

Pokreće se samo ovaj unaprijed definisan podskup:

- tri zaključana para (apple, cherry, potato);
- flip fraction `1,0`;
- FedAvg i FedProx, da budu upareni sa clean E=1 run-ovima;
- E=1, 10 rundi, 5 klijenata;
- α=0,1; 0,5; IID i seedovi 101, 211, 307, 401, 503;
- napadač = klijent sa najviše izvornih train slika na toj particiji.

To je 90 poslova. Finalni test se ne evaluira dok 4.1 ne završi.

Ne ulazi u 4.1:

- flip 0,25 i 0,50 (konfigi su već zapisani, ne pokreću se);
- model-update `s = −0,5` i `s = −1`;
- mediana, trimmed mean, Krum/MultiKrum;
- E=5.

Ti uslovi čekaju audit 4.1.

### Kanonski gate

Prvi posao: `pv19-capped-a05-s101-fedavg-e1-flip1-apple`.  
Napadač na toj particiji je `client-4`.

Gate je prošao 25. avgusta 2026. 12:41–12:48 UTC (~6,4 min):

- 10/10 rundi, `server_exit=0`;
- label-flip aktivan samo na `client-4` u svih 10 rundi;
- `label_flip_source_count=124` i `label_flipped_samples=124` po rundi
  (ukupno 1.240); ostali klijenti `source_count=0`;
- zaključana validation evaluacija: 879 slika, accuracy 0,807,
  macro-F1 0,804;
- `test_evaluated=false`.

Ostatak 4.1 (89 poslova) pokreće se odmah. Finalni test ostaje zapečaćen.

## 26. avgust 2026.

### Prekid nadzora i noćni zastoj

Cursor je izgubio nadzor launcher-a 25. avgusta 21:49 UTC, u trenutku
kad je Docker već krenuo `pv19-capped-iid-s503-fedavg-e1-flip1-apple`.
Sam launcher nije ugašen: Docker je ostao na tom poslu ~8 h (isti tip
zastoja kao u fazi 3), zatim završio 10/10 rundi i zapečatio validation
26. avgusta 05:56 UTC. `test_evaluated=false`.

Do tog trenutka završeno je 85/90 poslova. Launcher je odmah nastavio
preostalih pet (`iid-s503` cherry/potato × FedAvg, zatim FedProx × 3).
Drugi launcher se ne pokreće da ne bi sudario kontejnere.

### Završetak stage 4.1 treninga

Svih 90 poslova ima 10/10 rundi, `global_latest.npz` i
`validation_evaluation.json`. Posljednji:
`pv19-capped-iid-s503-fedprox-e1-flip1-potato` u 06:29 UTC.
Docker je idle. `test_evaluation.json`: 0 datoteka.

Finalni test ostaje zapečaćen dok se ne uradi audit 4.1.
Faza 4.2 se ne pokreće.

### Audit 4.1 i zaključani test

Artefakt: `docs/experiment_protocol/runs/pv19-capped-attack/phase4_audit.json`

- 90/90 imaju 10/10 rundi, checkpoint i validation;
- fit failures: 0; eval failures: 0;
- label-flip samo na zaključanom napadaču u svih 90 poslova;
- tipično trajanje 6,5 min (5,5–7,5), jedan zastoj
  `iid-s503-fedavg-e1-flip1-apple` ~8,1 h (nadzor/Docker, ne extra trening);
- wall-clock treninga: 25. avgust 12:41 UTC do 26. avgust 06:29 UTC.

`PV19_ATTACK_MODE=eval_test` je završio 26. avgusta. Svih 90 ima
`test_evaluation.json` (850 slika, `split=test`). Sažetak:

`docs/experiment_protocol/runs/pv19-capped-attack/phase4_test_summary.json`

Ti brojevi ne mijenjaju parove, napadače ni 4.2a aggregatore.

### Stage 4.2a — robusni aggregatori na flip 1,0

Isti particije, parovi, napadači, E=1 i 10 rundi kao 4.1. Tretman je
server aggregator: median, trimmed mean, Krum, MultiKrum. To je 180
poslova. Test se ne evaluira dok 4.2a ne završi.

Kanonski gate: `pv19-capped-a05-s101-median-e1-flip1-apple`
(napadač `client-4`). Gate je prošao 26. avgusta 07:31–07:38 UTC:
10/10 rundi, flip samo na `client-4` (1.240 flipped), validation 879
slika. Ostatak 4.2a (179 poslova) pokreće se odmah.

### Prekid zbog restarta laptopa

26. avgusta ~14:11 UTC laptop restart je prekinuo Docker na
`pv19-capped-a05-s101-median-e1-flip1-cherry` (3/10 rundi, bez
validation). Zapečaćeno je 61/180; nepotpuni artefakt je obrisan.
Launcher se nastavlja od tog posla i preskače kompletne. Test ostaje
zapečaćen.

Ne ulazi u 4.2a: flip 0,25/0,50, model-update, E=5, clean robust runs.

### Završetak stage 4.2a

Svih 180 poslova ima 10/10 rundi, checkpoint i validation. Posljednji:
`pv19-capped-iid-s503-multikrum-e1-flip1-potato` u 14:07 UTC 27. avgusta.
Audit: `docs/experiment_protocol/runs/pv19-capped-defense/phase4_defense_audit.json`

- fit/eval failures: 0;
- label-flip samo na zaključanom napadaču;
- tipično ~7 min; dva zastoja (`iid-s101-krum-apple` ~8,4 h,
  `iid-s401-krum-apple` ~80 min);
- wall-clock: 26. avgust 07:31 UTC do 27. avgust 14:07 UTC.

Zaključani test se evaluira odmah. Brojevi ne mijenjaju 4.2b.

### Audit 4.2a locked test

`PV19_DEFENSE_MODE=eval_test` je završio 27. avgusta 14:35 UTC. Svih 180
ima `test_evaluation.json` (850 slika). Sažetak:

`docs/experiment_protocol/runs/pv19-capped-defense/phase4_defense_test_summary.json`

Ti brojevi ne mijenjaju 4.2b.

### Stage 4.2b — osjetljivost flip 0,25 i 0,50

Isti particije, parovi, napadači, FedAvg/FedProx, E=1 kao 4.1. To je 180
poslova. Test ostaje zapečaćen dok 4.2b ne završi.

Ne ulazi u 4.2b: model-update, E=5, clean robust runs.

### Prekid Docker daemon-a

27. avgusta 19:02 UTC Docker se ugasio tokom
`pv19-capped-a01-s401-fedprox-e1-flip050-apple` (launcher exit 127, bez
artefakta). Zapečaćeno je 43/180. Daemon je ponovo podignut; launcher se
nastavlja i preskače kompletne. Test ostaje zapečaćen.

### Pauza 27. avgust naveče i nastavak 28. avgusta

Korisnik je pauzirao launcher na 55/180. 28. avgusta 4.2b opet radi u
lokalnom terminalu; drugi launcher se ne pokreće. Pi ostaje za povratak.
Nakon 4.2b slijede audit, zaključani test, dopuna rukopisa, 4.2c
(model-update i clean robust) i faza 5 (45 full poslova). E=5 pod napadom
samo ako ostane vremena. Na kraju se gasi Docker.

## 28. avgust 2026.

### Završetak stage 4.2b treninga

Svih 180 poslova ima 10/10 rundi, `global_latest.npz` i
`validation_evaluation.json`. Posljednji:
`pv19-capped-iid-s503-fedprox-e1-flip050-potato` u 20:58 UTC.
Audit: `docs/experiment_protocol/runs/pv19-capped-flip-sensitivity/phase4_flip_sensitivity_audit.json`

- fit/eval failures: 0;
- label-flip samo na zaključanom napadaču;
- tipično ~6,4 min (5,3–7,8); training stalls: 0
  (prekidi 27–28. avgusta oporavljeni skip-complete nastavkom);
- wall-clock treninga: 27. avgust 14:37 UTC do 28. avgust 20:58 UTC.

### Audit 4.2b locked test

`PV19_FLIP_SENS_MODE=eval_test` je završio 28. avgusta 21:27 UTC. Svih 180
ima `test_evaluation.json` (850 slika). Sažetak:

`docs/experiment_protocol/runs/pv19-capped-flip-sensitivity/phase4_flip_sensitivity_test_summary.json`

Ti brojevi ne mijenjaju 4.2c (model-update i clean robust već su
zaključani prije pregleda testa).

### Stage 4.2c — model-update, zatim clean robust

Prvo 60 poslova: scale `s = −0,5` i `s = −1` × FedAvg/FedProx × 3 α × 5
seedova. Napadač = zaključani monopol klijent za `Apple___healthy`.
Zatim 60 čistih poslova robusnih agregatora (bez napada). Test ostaje
zapečaćen dok svaki slice ne završi. E=5 pod napadom samo ako ostane
vremena prije nedjelje uveče.

## 29. avgust 2026.

### Završetak 4.2c model-update treninga

Svih 60 poslova ima 10/10 rundi, checkpoint i validation. Posljednji:
`pv19-capped-iid-s503-fedprox-e1-muminus1` u 03:52 UTC.
Audit: `docs/experiment_protocol/runs/pv19-capped-model-update/phase4_model_update_audit.json`

- fit/eval failures: 0;
- model-update samo na zaključanom Apple napadaču, skala tačna u 10/10 rundi;
- tipično ~6,3 min (5,4–7,2); stalls: 0;
- wall-clock: 28. avgust 21:32 UTC do 29. avgust 03:53 UTC.

### Audit 4.2c model-update locked test

`PV19_MU_MODE=eval_test` je završio 29. avgusta 04:03 UTC. Svih 60 ima
`test_evaluation.json` (850 slika). Sažetak:

`docs/experiment_protocol/runs/pv19-capped-model-update/phase4_model_update_test_summary.json`

Ti brojevi ne mijenjaju čiste robusne agregatore.

### Stage 4.2c — clean robust

60 poslova: median / trimmed mean / Krum / MultiKrum × 3 α × 5 seedova,
E=1, bez napada. Test ostaje zapečaćen dok slice ne završi.

### Završetak 4.2c clean robust

Svih 60 poslova ima 10/10 rundi, checkpoint i validation. Posljednji:
`pv19-capped-iid-s503-multikrum-e1-clean` u 10:28 UTC.
Audit: `docs/experiment_protocol/runs/pv19-capped-clean-robust/phase4_clean_robust_audit.json`

- fit/eval failures: 0; unexpected attacks: 0;
- tipično ~6,3 min (5,3–7,0); stalls: 0;
- wall-clock: 29. avgust 04:08 UTC do 10:28 UTC.

`PV19_CLEAN_ROBUST_MODE=eval_test` je završio 29. avgusta ~10:40 UTC.
Svih 60 ima `test_evaluation.json` (850 slika). Sažetak:

`docs/experiment_protocol/runs/pv19-capped-clean-robust/phase4_clean_robust_test_summary.json`

Ti brojevi ne mijenjaju fazu 5 (PV-19-full već je zaključan).

### Faza 5 — PV-19-full potvrda

45 poslova, E=1, 10 rundi. Prvi: `pv19-full-a01-s101-fedavg-e1`.
Test ostaje zapečaćen dok slice ne završi. E=5 pod napadom ostaje
opciono ako ostane vremena poslije faze 5.

## 30. avgust 2026.

### Završetak faze 5

Svih 45 poslova ima 10/10 rundi, checkpoint i validation. Posljednji:
`pv19-full-iid-s503-median-e1-flip1-apple` u 03:18 UTC.
Audit: `docs/experiment_protocol/runs/pv19-full-confirm/phase5_full_audit.json`

- fit/eval failures: 0;
- flip samo na full Apple napadaču; čisti poslovi bez napada;
- tipično ~22 min (18–25); stalls: 0;
- wall-clock: 29. avgust 10:42 UTC do 30. avgust 03:18 UTC.

`PV19_FULL_MODE=eval_test` je završio 30. avgusta 03:35 UTC. Svih 45 ima
`test_evaluation.json` (3.110 slika). Sažetak:

`docs/experiment_protocol/runs/pv19-full-confirm/phase5_full_test_summary.json`

E=5 pod napadom nije pokrenut. Docker se gasi da laptop odmori.
Pi ostaje za povratak.
