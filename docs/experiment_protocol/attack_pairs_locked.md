# Unaprijed zaključani napadi za fazu 4

Datum zaključavanja: 24. avgust 2026.  
Osnova: class monopoly i vlasništvo nad klasom u zaključanim PV-19-capped
train particijama. **Finalni test skup nije pregledan.**

Stari par `Tomato___healthy → Tomato___Late_blight` nije valjan na PV-19:
nijedna od tih klasa nije ušla u leaf-safe skup. Novi parovi su isti crop,
postoje u PV-19, i pokrivaju visoku i nižu koncentraciju.

## Tri primarna label-flip para

1. `Apple___healthy` → `Apple___Apple_scab`  
   visoka koncentracija, terenski relevantna kultura.
2. `Cherry___healthy` → `Cherry___Powdery_mildew`  
   visoka do srednja koncentracija, terenski relevantna kultura.
3. `Potato___healthy` → `Potato___Late_blight`  
   niža koncentracija i manji train support (`n=104` capped), kontrastni par.

Srednji monopoly izvora, pet seedova:

- Apple healthy: α=0,1 ≈ 0,661; α=0,5 ≈ 0,569; IID ≈ 0,209
- Cherry healthy: α=0,1 ≈ 0,762; α=0,5 ≈ 0,536; IID ≈ 0,211
- Potato healthy: α=0,1 ≈ 0,638; α=0,5 ≈ 0,523; IID ≈ 0,231

## Zaključana napadačka uloga

Napadač nije fiksno `client-0`. Za svaki seed napadač je klijent sa
najvećim brojem slika **izvorne** klase na toj particiji. Izvor je
`partitions_summary.json`; mašina čitljiva tabela je
`attack_attacker_clients.json`. Finalni test nije pregledan.

Apple healthy → Apple scab, napadač po seedu:

- α=0,1: client-0, 0, 1, 2, 1
- α=0,5: client-4, 1, 2, 1, 1
- IID: client-1, 1, 1, 2, 1

Cherry healthy → Cherry powdery mildew:

- α=0,1: client-3, 3, 1, 1, 3
- α=0,5: client-4, 4, 2, 0, 3
- IID: client-3, 0, 4, 4, 2

Potato healthy → Potato late blight:

- α=0,1: client-4, 1, 4, 3, 1
- α=0,5: client-3, 3, 4, 4, 4
- IID: client-0, 0, 0, 0, 0

Redoslijed seedova u svakom redu: 101, 211, 307, 401, 503.

## Jačina i algoritmi (ne pokreću se u fazi 3)

- primarni flip fraction: `1,0`;
- osjetljivost: `0,25` i `0,50`;
- model-update: `s = −0,5` i `s = −1`;
- clean i napadnuti uslovi na istim particijama;
- agregatori faze 4: FedAvg, FedProx, mediana, trimmed mean, Krum/MultiKrum.

Faza 4 ne počinje dok clean matrica, kvarovi i trošak nisu auditovani.
**25. avgust 2026:** taj gate je ispunjen. Stage 4.1 pokreće samo flip
`1,0` × FedAvg/FedProx × E=1 na svim zaključanim particijama.

**26. avgust 2026:** stage 4.1 je auditovan (90/90, 0 failures) i
zaključani test je evaluiran. Stage 4.2a, zaključan prije pregleda tih
test brojeva iz izbora aggregatora, pokreće iste flip-`1,0` napade sa
robusnim aggregatorima: mediana, trimmed mean (`β=0,25`), Krum (`f=1`,
keep=0) i MultiKrum (`f=1`, keep=`n−f=4`). Lokalni trening ostaje E=1.
Flip 0,25/0,50, model-update i E=5 ostaju 4.2b+.

**30. avgust 2026:** faza 5 auditovana (45/45) i test evaluiran.
E=5 pod napadom nije pokrenut. Pi čeka povratak.
