# Zaključani protokol novog istraživačkog zadatka

Datum početka: 24. avgust 2026.  
Status: faza 5 auditovana i testirana 30. avgusta (45/45). Zaključani
FL sliceovi na laptopu su zatvoreni. E=5 pod napadom nije pokrenut
(nema lock-a, nema vremena prije povratka). Pi čeka korisnika.  
Primarni cilj: ispitati kako class-concentrated non-IID raspodjela utiče na
korisnost, ciljano trovanje i robusnu agregaciju u reproduktivnom small-silo FL
testbedu, uz odvojenu potvrdu na fizičkom Raspberry Pi klijentu.

Ovaj dokument je operativni protokol. Svaka promjena unaprijed definisanih
odluka mora biti zabilježena u odjeljku „Odstupanja od protokola” prije analize
rezultata na koju ta promjena utiče.

## 1. Granice primarnog rada

Primarni naučni doprinos nije novi FL algoritam niti novi model za prepoznavanje
biljnih bolesti. PlantVillage i pametna poljoprivreda služe kao praktičan,
kontrolisan slučaj za:

1. mjerenje class-level heterogenosti, a ne samo globalnog Dirichlet parametra;
2. povezivanje koncentracije klase sa clean utility i target-class štetom;
3. upareno poređenje agregatora na identičnim particijama;
4. provjeru da isti protokol može uključiti fizički edge klijent, uz mjerljive
   sistemske troškove.

DP nije dio primarnih tvrdnji dok mehanizam, susjedstvo, sensitivity bound,
randomness i accounting ne budu zasebno metodološki popravljeni.

## 2. Istraživačka pitanja

**RQ1.** Kako Dirichlet heterogenost i koncentracija vlasništva nad klasom
utiču na globalni macro-F1, balanced accuracy, worst-class recall i per-client
korisnost pri jednakom optimization budgetu?

**RQ2.** U kojoj mjeri class ownership concentration predviđa štetu ciljanog
label-flip napada i jaz između ukupne i target-class metrike?

**RQ3.** Kako FedAvg, FedProx i odabrani robusni agregatori mijenjaju clean
utility–attack robustness kompromis na identičnim dataset particijama?

**RQ4.** Može li se zaključani protokol izvršiti sa Raspberry Pi 5 klijentom
uz kvantifikovane round latency, local train time, komunikaciju, CPU, RAM,
temperaturu, throttling i mTLS overhead?

RQ4 ostaje systems doprinos samo ako se navedena mjerenja zaista prikupe.
U suprotnom se Pi/mTLS prikazuje kao implementation validation.

## 3. Dataset uloge

### 3.1 Primarni kontrolisani benchmark: PV-19-capped

- Izvor: originalne RGB slike iz lokalnog PlantVillage repozitorijuma.
- Klase: 19 klasa sa najmanje 99% class-aware `leaf_id` pokrivenosti; nakon
  class-level filtera izbacuju se i pojedinačne slike bez `leaf_id`.
- Ograničenje: najviše 300 slika po klasi.
- Veličina prije train/validation/test podjele: 5.526 slika u 993 fizička
  leaf ID-a.
- Namjena: kompletna multi-seed utility, heterogeneity i attack/defense
  matrica uz prihvatljiv računarski trošak.

Postojeći `infra/fl-data-pv-v2` nije automatski prihvaćen kao zaključani
benchmark. Prvo mora biti rekonstruisan iz deklarisanog izvora uz group-safe
podjelu i manifest sa hash vrijednostima.

### 3.2 Potvrdni benchmark: PV-19-full

- Isti izvor, iste 19 klase i ista kanonska imena kao PV-19-capped.
- Bez ograničenja od 300 slika po klasi.
- Namjena: potvrda samo unaprijed odabranih ključnih nalaza, bez ponavljanja
  cijele široke matrice.

PV-19-full je primarni potvrdni skup jer zadržava istu label-space definiciju.
PV-27 ostaje exploratory audit varijanta: u capped verziji sadrži 1.200 slika
bez poznatog leaf ID-a i potvrđene neriješene cross-split perceptual kandidate,
pa nije prihvatljiv za primarne confirmatory tvrdnje. PV-38 može kasnije
služiti kao odvojen scalability test, ali se njegovi
rezultati ne porede direktno sa PV-19 kao da je riječ o istom zadatku.

### 3.3 Terenski dokazi

Vlastite terenske fotografije se ne zanemaruju i ne miješaju u zaključani
PlantVillage test skup. Njihove uloge su:

- eksterni domain-shift/OOD test ako postoje stručne ground-truth oznake;
- personalization studija ako postoje dovoljne oznake i odvojene
  biljka/list/sesija/date grupe;
- kvalitativna demonstracija samo za fotografije bez pouzdane oznake.

Train i test fotografije ne smiju biti iz istog burst-a, istog fizičkog lista
ili iste capture sesije. Finalni protokol će koristiti grupisanje po biljci,
listu, sesiji i datumu, u mjeri u kojoj metapodaci to omogućavaju.

### 3.4 Dodatni javni terenski skup

U primarni rad se uključuje najviše jedan javni terenski benchmark:

- Plant Pathology 2021 ako je cilj nezavisna replikacija na velikom realnom
  skupu, kao zaseban multi-label FL zadatak; ili
- PlantDoc ako je cilj direktnija, ali manja domain-shift provjera prema
  PlantVillage klasama.

RoCoLe nije primarni izbor zbog male veličine, jednog polja, uskog coffee
domena i jake neravnoteže. Može biti naknadni stress-test, ne ključni dokaz.
Konačan izbor javnog skupa donosi se tek nakon inventara i kvaliteta oznaka
vlastitih terenskih slika.

## 4. Obavezna group-safe podjela

Jedan zaključani globalni split nastaje prije FL particionisanja:

- train pool: 70%;
- development validation: 15%;
- final test: 15%.

Procenti se primjenjuju približno po klasi na nivou grupa, ne pojedinačnih
slika. Sve slike istog fizičkog lista moraju pripasti samo jednom splitu.
Finalni test se ne koristi za izbor hiperparametara, runde, agregatora,
attack para niti model checkpointa.

Hijerarhija group ID izvora:

1. class-aware PlantVillage `leaf_id`, gdje je dostupan;
2. identičan SHA-256 sadržaj slike;
3. potvrđena near-duplicate/perceptual grupa;
4. jedinstven image ID samo kada prethodne veze nisu pronađene.

Perceptualni kandidati se ne spajaju automatski samo na osnovu labavog praga.
Audit mora prijaviti prag, udaljenost, veličinu komponente i reprezentativne
parove za ručnu ili strogu automatsku potvrdu.

Početni audit lokalnog punog RGB skupa utvrdio je:

- 54.305 lokalnih RGB datoteka u 38 klasa;
- 40.328 ključeva u postojećem `leaf-map.json`;
- potpuna `leaf_id` pokrivenost za većinu, ali ne za sve klase;
- nultu trenutnu pokrivenost, između ostalog, za kukuruz, squash, tomato
  target spot i tomato mosaic virus;
- djelimičnu pokrivenost za nekoliko tomato klasa.

Zato samo prisustvo `leaf-map.json` nije dovoljno da se split proglasi
group-safe. Mora se arhivirati izvještaj pokrivenosti i unresolved grupa.

## 5. Dataset manifest i provjere

Za svaku dataset varijantu arhiviraju se:

- naziv i verzija izvora;
- relativna putanja, kanonska klasa, veličina datoteke i SHA-256;
- group ID i dokaz porijekla group ID-a;
- split (`train`, `validation`, `test`);
- subset selection seed i split seed;
- lista klasa i class-to-index mapiranje;
- broj slika i grupa po klasi i splitu;
- exact-duplicate i perceptual-audit izvještaj;
- provjera da nema group ID ili SHA-256 presjeka između splitova;
- hash samog manifesta i korištena verzija skripte.

Dataset gate prolazi samo ako:

1. nijedna poznata grupa ne prelazi split granicu;
2. nijedan SHA-256 ne prelazi split granicu;
3. svaka klasa ima train uzorke i, gdje broj grupa dopušta, validation i test;
4. svi manifest pathovi postoje;
5. ponovno pokretanje sa istim parametrima daje identičan manifest hash;
6. nepoznata ili nepotpuna `leaf_id` pokrivenost je eksplicitno kvantifikovana.

## 6. FL particije i seedovi

Klijentske particije nastaju isključivo iz zaključanog train poola.
Validation i test slike se nikada ne upisuju u klijentske training manifeste.

Primarna postavka:

- pet klijenata;
- alpha: 0,1; 0,5; i IID kontrola;
- FedAvg i FedProx;
- lokalne epohe: 1 i 5;
- jednak broj rundi i unaprijed definisan local-work budget;
- najmanje pet uparenih partition/training seedova.

Zaključani komunikacioni budžet za primarnu clean matricu je **10 rundi**.
Lokalne epohe E=1 i E=5 porede se pri tom istom broju rundi. Jednak
local-work budžet (npr. E=5 × 2 runde naspram E=1 × 10 rundi) ostaje
sekundarna, unaprijed najavljena analiza, ne uslov za početak matrice.

FedProx koristi `μ = 0,01` dok se ne uvede zasebna, unaprijed zaključana
osjetljivost na μ.

Početna unaprijed definisana seed lista je:

`101, 211, 307, 401, 503`

Dataset subset i globalni split imaju zaseban, fiksiran seed i ne mijenjaju se
između ovih run-ova. Isti partition seed koristi se kroz poređene algoritme,
clean/attack parove i lokalne epohe.

Prije pune matrice radi se samo tehnički smoke test. Njegov rezultat se ne
uključuje u naučnu statistiku.

## 7. Primarne metrike i vrijeme evaluacije

Primarne finalne metrike računaju se na zaključanom test skupu u unaprijed
fiksiranoj finalnoj rundi:

- macro-F1;
- balanced accuracy;
- weighted/global accuracy;
- per-class recall;
- worst-class recall;
- 10. percentil class recall distribucije.

Sekundarne metrike:

- validation AULC;
- rounds-to-target, ako je target definisan prije run-ova;
- per-client utility;
- komunikacioni i vremenski trošak;
- peak validation rezultat, jasno označen kao sekundaran.

Za napade su primarne:

- target-class recall harm;
- attacked macro-F1;
- clean utility penalty;
- attack recovery;
- attack success/visibility gap.

Per-image predikcije, targeti, confidence vrijednosti i checkpoint identitet
moraju se arhivirati kako bi bili mogući upareni i class-level testovi.

## 8. Statistički plan

Za svaku konfiguraciju izvještavaju se mean, standardna devijacija i 95%
bootstrap interval kroz nezavisne seed run-ove. Algoritmi i clean/attack
uslovi porede se upareno:

- paired permutation test kao primarni test;
- Wilcoxon signed-rank kao sensitivity analiza;
- uparena razlika i interval efekta;
- Holm korekcija unutar unaprijed definisane porodice poređenja.

Class-level analiza koristi entropy i monopoly/concentration mjere. Planirani
model je:

`class_recall ~ entropy + monopoly + algorithm + attack +`
`monopoly:attack + algorithm:attack + (1 | seed) + (1 | class)`

Ako se arhiviraju per-image predikcije, prednost ima binomial mixed model nad
regresijom agregiranih stopa.

## 9. Redoslijed izvođenja i phase gates

### Faza 0 — dataset audit

1. Implementirati class normalization bez kopiranja datoteka.
2. Izmjeriti `leaf_id` pokrivenost po klasi.
3. Auditovati exact duplikate i napraviti perceptual candidate izvještaj.
4. Generisati primarne PV-19-capped/PV-19-full i exploratory PV-27 manifeste.
5. Verifikovati dataset gate i reproduktivnost.

### Faza 1 — evaluation i experiment infrastruktura

1. Odvojiti train particije, development validation i final test.
2. Dodati centralni post-run evaluator za sačuvane globalne checkpointove.
3. Arhivirati per-image predikcije i kompletan run/environment manifest.
4. Omogućiti stvarne `FL_LOCAL_EPOCHS`, seed i FedProx parametre kroz launcher.
5. Dodati automatske testove protokola.

### Faza 2 — lokalni smoke test

Jedna mala konfiguracija, smanjen broj rundi i slika, služi samo provjeri da
manifeste, treniranje, agregaciju i evaluator povezuje isti class mapping.

### Faza 3 — PV-19-capped utility matrica

Prvo clean FedAvg/FedProx matrica. Attack/defense matrica ne počinje dok clean
rezultati, kvarovi i procijenjeni troškovi nisu auditovani.

### Faza 4 — attack/defense matrica

Najmanje tri source/target para biraju se unaprijed na osnovu class
concentration profila, ne na osnovu pregledanja finalnih test rezultata.

### Faza 5 — PV-19-full potvrda

Ponavljaju se samo ključne konfiguracije zaključane nakon capped razvojnog
protokola, uz jasno označavanje confirmatory i exploratory analiza.

### Faza 6 — field i fizički Pi

Terenska anotacija/split i systems instrumentacija moraju biti spremni prije
Pi run-a. Pi se ne uključuje samo radi još jednog accuracy rezultata.

## 10. Kada je potreban Raspberry Pi

Pi nije potreban za faze 0–5 dok se dataset, lokalna infrastruktura i primarni
modeli ne stabilizuju. Korisnik će biti eksplicitno obaviješten prije faze 6.
Tada će zahtjev sadržati:

- tačnu granu/verziju koda ili arhivski hash;
- komandu koju treba pokrenuti;
- očekivani izlaz i health check;
- potrebne senzore/mjerače;
- procijenjeno trajanje;
- lokaciju artefakata koji se vraćaju u analizu.

## 11. Odstupanja od protokola

**24. avgust 2026, prije bilo kojeg novog FL run-a.** Početna PV-27 namjera
zamijenjena je primarnim PV-19 protokolom. Audit je pokazao da zvanični lokalni
leaf metadata nije potpun za svih 27 izabranih klasa. Exploratory PV-27-capped
manifest imao je 1.200 slika bez leaf ID-a; perceptual audit je zatim pronašao
neriješene slične parove koji prelaze split granicu, uključujući vizuelno
potvrđene snimke istog fizičkog lista. Prag perceptualnog hasha istovremeno
propušta većinu poznatih sibling slika, pa nije korišten za automatsko spajanje.

Primarni skup zato zadržava samo klase sa najmanje 99% class-aware leaf
pokrivenosti i izbacuje preostale slike bez leaf ID-a. Dobijeni
`pv19-capped-62b5b2119fb2` ima 5.526 slika, 993 grupe, nula nepoznatih leaf
ID-a i prolazi dataset gate. Promjena je metodološka korekcija donesena prije
pregleda bilo kojeg novog modelskog rezultata, ne post hoc izbor prema accuracy.

**31. avgust 2026, nakon zatvaranja FL sliceova, prije izračuna p-vrijednosti.**
Inferencijalni plan za RQ1–RQ3 zaključan je u
`docs/experiment_protocol/locked_inference_plan.md` i izvršen na već
arhiviranim locked-test artefaktima (`docs/manuscript/locked_inference.py`).
Nema novog treninga. U mixed modelu protokolni faktor `attack` kodiran je kao
`targeted` na nivou klase unutar posla (1 samo ako je ta klasa zaključani
izvor label-flip-a). Job-level napad bi pomiješao ciljanu štetu s kolateralom
na ostalih 18 klasa i ne bi odgovorio na RQ2. Analiza je confirmatory za
porodice F1–F3 i za Gaussian LMM; binomni GEE je osjetljivost (pet seed
klastera).

Svako naredno odstupanje bilježi datum, razlog, zahvaćene run-ove i da li je
analiza confirmatory ili exploratory.
