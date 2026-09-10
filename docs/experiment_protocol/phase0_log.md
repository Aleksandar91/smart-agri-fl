# Faza 0 — dnevnik metodološkog audita

## 24. avgust 2026.

### Korak 0.1 — potvrda lokalnog PlantVillage izvora

Potvrđen je lokalni originalni PlantVillage repozitorijum:

`iot-edge/sensors/data/PlantVillage/PlantVillage-Dataset-master/PlantVillage-Dataset-master`

Inventar varijanti:

- `raw/color`: 54.305 RGB datoteka, 38 klasa;
- `raw/grayscale`: 54.305 datoteka, 38 klasa;
- `raw/segmented`: 54.306 datoteka, 38 klasa;
- `leaf-map.json`: 40.328 ključeva;
- `leaf_grouping/filtered_leafmaps`: class-level CSV mape fizičkih listova.

Za novi protokol koristi se samo `raw/color`. Nije potrebno ponovno
preuzimanje PlantVillage skupa. Razlika od jedne datoteke između lokalne color
i segmented varijante ostaje zabilježena i ne popunjava se iz druge varijante.

### Korak 0.2 — audit postojećeg pipelinea

Postojeći tok je imao dvije metodološke slabosti:

1. `partition_dataset.py` je particionisao sve slike direktno po klijentima,
   bez prethodno zaključanog globalnog test skupa;
2. `fl_task.py` je zatim pravio per-image lokalni train/validation split, bez
   korištenja `leaf_id` grupa.

Zbog toga različiti snimci istog fizičkog lista mogu završiti u train i
validation dijelu. Postojeći rezultati ostaju engineering evidence, ali se ne
koriste kao finalni confirmatory rezultati novog rada.

### Korak 0.3 — pokrivenost leaf metapodataka

Utvrđeno je da `leaf-map.json` nije potpun za svih 38 klasa. Posebno:

- više klasa ima potpunu class-aware pokrivenost;
- neke klase nemaju leaf mapu;
- neke tomato klase imaju samo djelimičnu mapu;
- ista originalna oznaka datoteke može postojati u više klasa, pa lookup samo
  po filenameu nije bezbjedan;
- Apple black rot u leaf mapi koristi patološki sinonim `Apple_Frogeye Spot`.

Implementiran je class-aware lookup koji prihvata samo leaf ID iz odgovarajuće
klase. Za Apple black rot eksplicitno je dokumentovan navedeni sinonim.

### Korak 0.4 — novi generator protokolskih manifesta

Dodan je:

`fl-client/app/prepare_pv_protocol.py`

Generator:

- ne kopira i ne mijenja slike;
- normalizuje 27 postojećih klasa;
- računa SHA-256 svake izvorne slike;
- povezuje class-aware leaf siblings i exact duplikate;
- bira capped varijantu na nivou cijelih grupa;
- pravi jedan globalni 70/15/15 train/validation/test split prije FL
  particionisanja;
- provjerava group i SHA-256 presjeke između splitova;
- zapisuje kompatibilne `samples` i detaljne per-image `records`;
- ostavlja dataset gate zatvorenim dok ne završi perceptual audit.

Dodan je test:

`fl-client/tests/test_prepare_pv_protocol.py`

Pet unit testova prolazi. Pokrivaju class-aware lookup, Apple sinonim,
leaf/exact povezivanje, determinističku podjelu, group-safe capped izbor i
detekciju split curenja. IDE linter ne prijavljuje greške.

### Korak 0.5 — prvi dry-run i korekcija source aliasa

Prvi stvarni run je namjerno prekinut bez izlaznog manifesta jer puni original
koristi drugačija imena deset tomato direktorijuma od ranijeg Kaggle mirror-a.

Nakon toga:

- dodana je eksplicitna source alias mapa;
- dodan je fail-fast audit svih izabranih direktorijuma prije hashiranja;
- nije korišteno tiho preskakanje nedostajućih klasa.

### Korak 0.6 — prvi reproduktivan PV-27-capped manifest

Artefakti:

`docs/experiment_protocol/datasets/pv27-capped-v1`

Rezultat:

- dataset ID: `pv27-capped-d9563fc4632b`;
- manifest SHA-256:
  `d9563fc4632b31a5a42da002dae5be0c510dd18efb4984f28bd44c64c098b690`;
- slike: 7.926;
- povezane grupe: 2.493;
- train: 5.469 slika;
- development validation: 1.251 slika;
- final test: 1.206 slika;
- slike sa poznatim leaf ID-om: 6.726;
- slike bez poznatog leaf ID-a: 1.200;
- exact duplicate skupovi u izabranoj varijanti: 7, ukupno 14 slika;
- group-boundary provjera: prolazi;
- SHA-256 boundary provjera: prolazi;
- reproduktivnost: nezavisno ponavljanje dalo je identičan manifest hash.

Stari per-image capped skup imao je 7.927 slika. Novi ima jednu sliku manje jer
generator nikada ne presijeca grupu fizičkog lista samo da bi dostigao tačno
300 slika u klasi. Ova razlika je očekivana metodološka korekcija, a ne gubitak
ili greška u izvoru.

### Trenutni phase gate

Strukturni dio gatea prolazi, ali kompletan dataset gate još ne prolazi.
Otvorena stavka je perceptual near-duplicate audit za 1.200 izabranih slika bez
poznatog leaf ID-a, kao i provjera kandidata između poznatih i nepoznatih
grupa. Nijedan novi naučni FL run neće biti pokrenut prije zatvaranja ovog
gatea.

Raspberry Pi nije potreban u fazi 0.

### Korak 0.7 — perceptual audit exploratory PV-27 varijante

Dodan je:

`fl-client/app/audit_pv_perceptual.py`

Alat računa 64-bitni pHash i dHash, koristi BK-tree za candidate retrieval i
nikada automatski ne spaja grupe. Unit test paket sada ima osam testova i svi
prolaze.

Na PV-27-capped manifestu:

- analizirano je 7.926 slika;
- početni pHash candidate skup sadržao je 96 parova iz različitih postojećih
  grupa;
- 76 parova ima različite, poznate leaf ID-e i zato se tretiraju kao
  metadata-resolved različiti listovi;
- 20 parova ostaje neriješeno jer najmanje jedna strana nema leaf ID;
- 11 od 20 neriješenih parova prelazi split granicu.

Vizuelno su provjereni najbolje rangirani kandidati. Parovi `Grape___healthy`
9127/9129 i `Tomato___Tomato_mosaic_virus` 2064/2065 prikazuju isti fizički
list u blago različitom položaju, a bili su raspoređeni u različite splitove.

Važno ograničenje: među 21.352 poznata sibling para median pHash udaljenost je
26, a samo 207 parova ulazi u radius 8 candidate skup. PHash zato ima dobru
vrijednost za pronalaženje nekih očiglednih near-duplicate parova, ali nema
dovoljan recall da rekonstruiše nedostajuće fizičke leaf grupe. Automatsko
spajanje pragom je odbijeno.

Exploratory PV-27 dataset gate ostaje zatvoren.

### Korak 0.8 — strict primarni PV-19-capped dataset

Prije novih FL run-ova protokol je metodološki sužen:

- klasa mora imati najmanje 99% class-aware leaf-ID pokrivenosti;
- nakon class filtera svaka preostala slika mora imati leaf ID;
- klase sa nepotpunom zvaničnom grupacijom ostaju samo u exploratory auditu.

Primarni artefakti:

`docs/experiment_protocol/datasets/pv19-capped-primary-v1`

Rezultat:

- dataset ID: `pv19-capped-62b5b2119fb2`;
- manifest SHA-256:
  `62b5b2119fb2e12b004e59c5fdfdef29765e2e4bb9c1b6cf79aeb3155ba8f872`;
- 19 klasa;
- 5.526 slika;
- 993 fizičke leaf grupe;
- 0 slika bez leaf ID-a;
- 7 exact-duplicate setova, obuhvaćenih postojećim grupama;
- perceptual kandidati između različitih grupa: 72, svi razriješeni različitim
  zvaničnim leaf ID-ima;
- neriješeni perceptual kandidati: 0;
- structural validation: prolazi;
- perceptual audit: prolazi;
- dataset gate: **prolazi**.

Promjena PV-27 → PV-19 napravljena je prije pregleda bilo kojeg novog modelskog
rezultata. Razlog je kontrola curenja podataka, ne optimizacija accuracy.

Sljedeći korak je izrada PV-19-full potvrdnog manifesta i prilagođavanje
`partition_dataset.py` da klijente particioniše isključivo iz zaključanog
`train.json` poola.

### Korak 0.9 — PV-19-full confirmatory dataset

Generisan je puni potvrdni skup sa identičnim 19-klasnim label spaceom:

`docs/experiment_protocol/datasets/pv19-full-confirmatory-v1`

Rezultat:

- dataset ID: `pv19-full-774007483a1d`;
- manifest SHA-256:
  `774007483a1d71af339ae430d2e41dfd90fb966363c18c238dc0dc97004626d2`;
- slike: 20.597;
- fizičke leaf grupe: 4.064;
- train: 14.372;
- development validation: 3.115;
- final test: 3.110;
- slike bez leaf ID-a: 0;
- neriješeni perceptual kandidati: 0;
- dataset gate: **prolazi**.

Kompletna utility/attack matrica neće se izvršavati na ovom skupu. Njegova
namjena je unaprijed ograničena potvrda ključnih nalaza iz PV-19-capped studije.

### Korak 0.10 — group-safe FL particionisanje train poola

`fl-client/app/partition_dataset.py` je proširen opcijom
`--source-manifest`. Novi protokolski način:

- prihvata isključivo manifest čiji je split `train`;
- prenosi dataset ID i manifest SHA-256 u svaki klijentski manifest;
- dodjeljuje cijele leaf grupe, ne pojedinačne slike;
- provjerava da nijedna grupa ne prelazi granicu klijenata;
- provjerava da je svaka izvorna grupa dodijeljena tačno jednom;
- zapisuje class-level monopoly, normalized entropy, effective-client i
  client sample-share mjere.

Legacy `--data-root` način ostaje dostupan samo radi reprodukcije starijih
run-ova. Ne koristi se u novoj confirmatory studiji.

Prva provjera:

`docs/experiment_protocol/partitions/pv19-capped/a05/seed-101`

Konfiguracija:

- dataset: `pv19-capped-62b5b2119fb2`;
- izvor: isključivo zaključani train pool;
- pet klijenata;
- Dirichlet alpha = 0,5;
- partition seed = 101;
- 3.797 train slika;
- 685 leaf grupa, svih 685 dodijeljeno tačno jednom.

Dobijene veličine klijenata su 556, 551, 757, 729 i 1.204 slike.
Srednji class monopoly iznosi 0,611, a srednja normalizovana class entropy
0,629. Ovo potvrđuje da alpha sam po sebi ne opisuje realizovanu class-level
heterogenost.

Test paket sada sadrži 13 prolaznih testova. Novi testovi pokrivaju train-only
source gate, IID i Dirichlet group assignment, determinističnost, zabranu
cross-client group curenja i računanje monopoly/entropy mjera.

## Faza 1 — evaluation infrastruktura i tehnički smoke

### Korak 1.1 — odvajanje training i development-validation podataka

`fl_task.load_data` i `fl_client.py` sada podržavaju zaključani
`FL_EVAL_MANIFEST`.

U novom protokolskom načinu:

- klijent trenira na svim slikama iz svoje group-safe train particije;
- više se ne oduzima slučajnih 15% klijentskih train slika;
- round evaluacija koristi isključivo globalni `validation.json`;
- `test.json` je eksplicitno odbijen u FL round evaluatoru;
- provjeravaju se dataset ID, class order i path presjek train/validation
  manifesta.

Flower server podržava `FL_EVAL_CLIENTS`. Pošto svi klijenti imaju identičan
zaključani validation manifest, po rundi se bira jedan evaluator. Time se
izbjegava petostruko ponavljanje istih predikcija i lažno petostruko uvećanje
class support vrijednosti.

### Korak 1.2 — nezavisni post-run evaluator

Dodan je:

`fl-client/app/evaluate_global.py`

Evaluator:

- prihvata samo zaključani validation ili test manifest;
- učitava tačno određeni Flower `.npz` checkpoint;
- koristi deterministički evaluation transform;
- računa accuracy, balanced accuracy, macro-F1, weighted-F1, worst-class
  recall, 10. percentil recall-a, per-class metrike i confusion matrix;
- arhivira per-image target, prediction, confidence i top-3;
- zapisuje checkpoint SHA-256, dataset manifest SHA-256 i verzije okruženja.

Tehnički test na inicijalnom 19-klasnom checkpointu uspješno je obradio svih
879 validation slika. Njegove metrike nisu istraživački rezultat.

### Korak 1.3 — prvi FL smoke i otkrivena Flower konfiguraciona zamka

Prvi 1-round run je završio treniranje svih pet klijenata bez greške, ali
server log je pokazao:

`configure_evaluate: no clients selected, skipping evaluation`

Uzrok: Flower preskače evaluaciju kada je `fraction_evaluate=0`, čak i kada je
`min_evaluate_clients=1`. Konfiguracija je ispravljena na
`fraction_evaluate = eval_clients / min_clients`. Launcher sada poslije svakog
run-a provjerava da postoji uspješna zaključana validation evaluacija za svaku
rundu; odsustvo evaluacije ubuduće završava run greškom.

### Korak 1.4 — uspješan protokolski smoke run

Artefakti:

`docs/experiment_protocol/smoke/fl-a05-s101-r1-v2`

Konfiguracija:

- eksperiment: `smoke-pv19-a05-s101-fedavg-r1-v2`;
- dataset: `pv19-capped-62b5b2119fb2`;
- partition: alpha 0,5, seed 101;
- pet train klijenata, jedan validation evaluator;
- FedAvg, jedna runda, jedna lokalna epoha, image size 128;
- train slike: 3.797;
- global validation slike: 879;
- fit failures: 0;
- evaluation failures: 0.

Round-1 development-validation rezultat:

- loss: 1,8311;
- accuracy: 53,47%;
- balanced accuracy: 53,52%;
- macro-F1: 48,92%;
- worst-class recall: 0%;
- 10. percentil class recall-a: 6,96%.

Nezavisni post-run evaluator dao je identičnu accuracy vrijednost
`0.534698521046644`, čime je potvrđena konzistentnost Flower i centralnog
evaluation puta. Checkpoint SHA-256 je:

`5784131ed4d263a6e3567b4e66a739c5f95367c4204d9489f1ef28293df6987d`

Ovo je isključivo tehnički smoke rezultat. Ne ulazi u multi-seed statistiku,
ne koristi se za izbor algoritma i finalni test skup nije evaluiran.

Test paket sada sadrži 17 prolaznih testova; shell syntax i IDE lint provjere
takođe prolaze.

### Korak 1.5 — FedProx i priprema clean utility matrice

Prethodna sesija se prekinula tokom umetanja FedProx-a zbog limita modela.
Nastavak je završio tu infrastrukturu bez pokretanja naučne 10-round matrice.

FedProx:

- Flower strategija `FedProx` sa obaveznim `proximal_mu`;
- server šalje `μ` i kroz `on_fit_config_fn` i kroz `configure_fit`;
- klijent dodaje `0.5 μ ||w − w_global||²` samo na trainable parametre;
- launcher prima `FL_PROXIMAL_MU`;
- zaključana vrijednost za primarnu matricu: `μ = 0,01`.

Tehnički smoke:

`docs/experiment_protocol/smoke/fl-a05-s101-fedprox-r1`

- ista particija kao FedAvg smoke (`a05`, seed 101);
- jedna runda, jedna lokalna epoha;
- svih pet klijenata je primilo `proximal_mu = 0.01`;
- fit/evaluation failures: 0;
- development-validation accuracy: 54,15%.

Ovo nije naučni rezultat i ne ulazi u statistiku.

Group-safe particije za cijelu clean matricu:

`docs/experiment_protocol/partitions/pv19-capped/{a01,a05,iid}/seed-{101,211,307,401,503}`

Srednji class monopoly (pet seedova) grubo razdvaja uslove:

- α=0,1: oko 0,75–0,81;
- α=0,5: oko 0,50–0,61;
- IID: oko 0,22, uz normalizovanu entropiju ≈ 0,998.

Launcher i evidencija poslova:

- `infra/generate_pv19_partitions.sh`
- `infra/run_pv19_utility_matrix.sh` (default je samo lista; izvršavanje
  zahtijeva `PV19_MATRIX_MODE=run`)
- `docs/experiment_protocol/runs/pv19-capped/utility_matrix_jobs.json`

Pre-registrovano je 60 clean run-ova: 3 raspodjele × 2 algoritma × 2
lokalne epohe × 5 seedova, po 10 rundi. Nijedan naučni run još nije
pokrenut. Test paket sada ima 18 prolaznih testova.
