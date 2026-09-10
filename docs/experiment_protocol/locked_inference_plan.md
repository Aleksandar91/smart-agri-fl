# Zaključani inferencijalni plan (faze 3–4.2a)

Datum zaključavanja plana: 31. avgust 2026.  
Status: porodice i kodiranje zaključani **prije** izračuna p-vrijednosti.  
Ulaz: već arhivirani locked-test sažeci i `test_evaluation.json` (matrica zabune / per-class recall). Nema novog FL treninga. Nema izbora testova prema pregledu p-vrijednosti.

Osnova: `docs/research_protocol_20260824.md` §8 i RQ1–RQ3 u rukopisu.

## Jedinica uparivanja

- Seed lista: `101, 211, 307, 401, 503`.
- Algoritmi i clean/attack uslovi na istoj particiji uparuju se po seedu.
- Gdje porodica nalaže, tri zaključana para (jabuka, trešnja, krompir) **ulaze u isti seed-blok**: usrednjavaju se unutar seeda, pa je confirmatory n = 5 (amandman 8. septembar 2026.). Stari n = 15 (pari kao razmjenjivi) nije confirmatory.

## Testovi

Primarni test: upareni permutation test na sredini razlika (sve 2^n dodjele predznaka kada n ≤ 15).  
Osjetljivost: Wilcoxon signed-rank na istim razlikama.  
Interval: percentilni bootstrap 95% CI srednje uparene razlike, 10 000 ponavljanja, RNG seed 20260824.  
Holm korekcija unutar svake porodice ispod, ne preko porodica.

Sa n = 5 seedova tačan dvostrani permutation p ne može biti manji od 2/32 = 0,0625. To nije razlog da se porodica proširi ili da se nivo α naknadno mijenja.

## Porodica F1 — čista korisnost (RQ1), E = 1

Metrika: macro-F1 na zaključanom testu.

| ID | Poređenje | n |
|----|-----------|---|
| F1.1 | FedProx − FedAvg, α = 0,1 | 5 seedova |
| F1.2 | FedProx − FedAvg, α = 0,5 | 5 |
| F1.3 | FedProx − FedAvg, IID | 5 |

Holm preko F1.1–F1.3.

## Porodica F1b — heterogenost (RQ1), FedAvg E = 1

Isti seed, različite particije (uparivanje po RNG listi, ne po identičnom rasporedu slika).

| ID | Poređenje | n |
|----|-----------|---|
| F1b.1 | α = 0,1 − IID, macro-F1 | 5 |
| F1b.2 | α = 0,5 − IID, macro-F1 | 5 |

Holm preko F1b.1–F1b.2.

## Porodica F2a — šteta na izvoru (RQ2), FedAvg flip 1,0

| ID | H0 | n |
|----|----|---|
| F2a.1 | srednja source-recall harm = 0, α = 0,1 | 5 seedova (3 para usrednjena unutar seeda) |
| F2a.2 | isto, α = 0,5 | 5 |
| F2a.3 | isto, IID | 5 |

Holm preko F2a.1–F2a.3.

## Porodica F2b — jaz vidljivosti (RQ2), FedAvg flip 1,0

Razlika: source-recall harm minus pad tačnosti, ista skala.

| ID | H0 | n |
|----|----|---|
| F2b.1 | mean(harm − acc_drop) = 0, α = 0,1 | 5 seedova (3 para usrednjena unutar seeda) |
| F2b.2 | isto, α = 0,5 | 5 |
| F2b.3 | isto, IID | 5 |

Holm preko F2b.1–F2b.3.

## Porodica F2c — FedProx nije odbrana (RQ2), flip 1,0

| ID | Poređenje | n |
|----|-----------|---|
| F2c.1 | FedProx − FedAvg, source harm, α = 0,1 | 5 seedova (3 para usrednjena unutar seeda) |
| F2c.2 | isto, α = 0,5 | 5 |
| F2c.3 | isto, IID | 5 |

Holm preko F2c.1–F2c.3.

## Porodica F3 — robusni agregatori pri α = 0,1 (RQ3)

Confirmatory samo α = 0,1 (naslovna tvrdnja rukopisa). Ostali α ostaju opisni.

| ID | Poređenje | n |
|----|-----------|---|
| F3.1 | median − napadnuti FedAvg, source-recall recovery, α = 0,1 | 5 seedova (3 para usrednjena unutar seeda) |
| F3.2 | trimmed mean − napadnuti FedAvg, isto | 5 |
| F3.3 | MultiKrum − napadnuti FedAvg, isto | 5 |
| F3.4 | Krum − napadnuti FedAvg, isto | 5 |
| F3.5 | čisti Krum − čisti FedAvg, macro-F1, α = 0,1 | 5 seedova |

Holm preko F3.1–F3.5.

## Mixed model (RQ2)

Protokol:

`class_recall ~ monopoly + entropy + algorithm + attack + monopoly:attack + algorithm:attack + (1 | seed) + (1 | class)`

**Kodiranje `attack` (dokumentovano odstupanje).** Indikator je na nivou klase unutar posla, ne na nivou cijelog posla: `targeted = 1` samo ako je ta klasa zaključani izvor label-flip-a u tom poslu. Job-level `attack` bi pomiješao ciljanu štetu s kolateralom na ostalih 18 klasa i ne bi odgovorio na RQ2.

Obuhvat: faza 3 čisti E = 1 FedAvg/FedProx (30 poslova) i faza 4.1 flip 1,0 (90 poslova). Sve 19 klasa. Ishod: recall na zaključanom testu iz arhivirane `test_evaluation.json`. `monopoly` i `normalized_entropy` sa train particije.

Gaussian LMM je confirmatory model iz protokola. Binomni GEE na (tačni, support) klasterovan po seedu je osjetljivost, jer per-image predikcije postoje (agregacija po klasi daje isti binomni broj kao Bernoulli po slici).

Exploratory (nije Holm, nije confirmatory): isti LMM plus `log(test support)`.

Spearman ρ monopol–šteta na 45 FedAvg poslova ostaje opisna tačka uz seed-klasterovani bootstrap CI; nije treća p-porodica (poslovi unutar seeda nisu nezavisni).

## Šta nije u ovom planu

- E = 5 pod napadom (nije rađeno).
- Izbor dodatnih poređenja nakon p-vrijednosti.
- Faza 5 PV-19-full kao nova test porodica (potvrda ostaje opisna, slice je manji i nije pun par).
## Amendment — 8 September 2026 (inferential unit)

**Status:** dated amendment after isolated manuscript reviews. It does **not** add or drop comparisons, change Holm families, or use new training jobs. It corrects the exchangeability assumption for families that previously listed n = 15.

**What each seed controls.** The integers `101, 211, 307, 401, 503` draw the client partition from the locked train pool and are also passed to each client as `FL_SEED` (`seed_everything`: Python, NumPy, PyTorch). Attack JSON files use a separate flip-mask seed (`1337`); eligible source examples still depend on that partition. A confirmatory seed is therefore a joint replicate of ownership pattern and training randomness.

**Confirmatory unit for F2a, F2b, F2c, F3.1–F3.4.** Average the three locked pairs (apple, cherry, potato) inside each seed, then run the exact sign-flip and the percentile bootstrap on the five seed-level means (n = 5). That is equivalent to flipping all three pair gaps inside a seed together. The mean of the five seed-means equals the unweighted 15-job mean; only the p-value, bootstrap interval, and Wilcoxon on the clustered units change.

**What this implies.** The two-sided permutation floor is 0.0625 for those families as well. They cannot reject at 0.05. A bootstrap interval that excludes zero when all five seed-means have the same sign does not rescue the exact test.

**Diagnostic only.** The old n = 15 sign-flip (pairs treated as independently flippable) is stored in `locked_inference_results.json` under `legacy_job_level` and is not a confirmatory claim.

**Script:** `docs/manuscript/locked_inference.py`.

## Amendment — 8 September 2026 (α vs monopoly and dose qMc)

**Status:** secondary analysis of archived phase-4.1 and 4.2b jobs. Not a new Holm family. Table 10 is unchanged.

**Question.** The abstract asked whether monopoly explains harm better than α. Table 10 has no α term. Under the provisioned-owner rule, flip fraction q implies expected global corrupted share qMc.

**What was computed.** (1) Flip 1.0, n=90: Spearman Mc–harm pooled and within α; partial Spearman given α; leave-one-seed-out RMSE of harm ~ α vs α+Mc. (2) Fractions 0.25/0.50/1.0, n=270: harm vs qMc and vs realized flipped/N_c; OLS/LOSO with qMc and Mc together.

**Result used in the manuscript.** At flip 1.0, realized flipped/N_c equals Mc. Adding Mc to α improves seed-blocked prediction of harm, but the within-α=0.1 association is weak. Across fractions, harm tracks qMc; Mc given qMc is compatible with zero. No dose-matched extra experiment.

## Amendment — 8 September 2026 (LMM specification audit)

**Status:** secondary. Table 10 protocol formula and coefficients are unchanged.

**What was computed.** Named variance components; residual/fitted summaries; Pearson/VIF for Mc vs normalised entropy; monopoly-only and entropy-only LMMs; protocol formula with (1|job); protocol plus job-level `attack_job` alongside targeting; paired source-recall harm ~ Mc + α + algorithm + pair with (1|seed).

**Result used in the manuscript.** Class intercept is the boundary component. Monopoly×targeted survives job intercept (−0.692) and attack_job (−0.720); attack_job ≈ 0. Monopoly main effect is not interpretable (sign flips without entropy). Interaction is three source classes. Paired-harm Mc = 0.85 aligns with the Table 11 dose reading. GEE does not validate Table 10.

## Amendment — 9 September 2026 (preregistration / sealed / confirmatory wording)

**Status:** wording and chronology only. No new Holm family, no new training, no change to Table 9 p-values or Table 10 coefficients.

**What this plan is.** Dataset, partitions, pairs, and slices were locked on 24 August 2026 (`docs/research_protocol_20260824.md`) before phase 3–5 training. This inference document was written on 31 August 2026 on archived `test_evaluation.json` files and slice summaries, before p-values were computed. That is a locked analysis of archived predictions. It is **not** a public preregistration (OSF or similar). Blindness to means, plots, or per-class recalls is **not** documented.

**Sealed test.** The 850-image capped test is scored once per completed matrix slice and reused on later slices. It is not a once-only holdout kept unseen until the whole study ended.

**Confirmatory.** Family IDs F1–F3 are comparison bundles in this document. PV-19-full is a locked follow-on on more images of the same 19 classes. Validation learning-curve summaries exist in job histories; reporting them would be exploratory, not forbidden by a preregistration rule.

**Manuscript.** Table 13 lists dates, artefacts (`dataset_id` and summary JSON paths), and what was already on disk. Git SHAs are omitted because they were not part of the laboratory lock record.

## Amendment — 9 September 2026 (own-baseline source harm)

**Status:** secondary metric on archived 4.2a and 4.2c jobs. Not a Holm family. Table 9 and Table 10 unchanged.

**Question.** Table 4 source harm for aggregator \(a\) is versus clean FedAvg, so it mixes clean source-class cost with attack-induced damage.

**What was computed.** For each locked source class, clean source recall on the matching clean-robust job minus attacked source recall on the 4.2a job (same seed, α, aggregator). Pair-averaged like Table 4. Identity check: harm vs clean FedAvg = clean source cost + own-baseline harm.

**Result used in the manuscript.** At α = 0.1, Krum own-baseline harm is 0.250 (not zero); median / trimmed mean remain 0.536 / 0.559. Table 7 reports both columns. Figure 5 uses own-baseline harm on the vertical axis.

## Amendment — 10 September 2026 (visibility-gap arithmetic)

**Status:** secondary. Not a Holm family. Table 9 F2b formula unchanged; its interpretation is operational.

**Question.** \(H_s-\Delta\mathrm{Accuracy}=0\) compares a class-level change with a prevalence-weighted change. Report \(\pi_s\), \(\pi_s H_s\), collateral \(\Delta\mathrm{Accuracy}-\pi_s H_s\), and \(P(\hat y=\mathrm{target}\mid y=\mathrm{source})\).

**Result used in the manuscript.** Locked test supports: Apple 50, Cherry 46, Potato 24 of 850. At α = 0.1 FedAvg, \(\pi_s H_s\) is 0.035 / 0.034 / 0.016 against accuracy drops 0.016 / 0.042 / 0.004. Pair-mean source term 0.028 vs drop 0.021. Source-to-target rates 0.584 / 0.691 / 0.425; Potato errors are more dispersed. No detector was evaluated.

## Amendment — 10 September 2026 (executable specification)

**Status:** methods dump. Not a Holm family. No new training. No new table.

**Question.** Sections 3 and 8 must specify Dirichlet allocation, leaf-group integrity among clients, leaf-ID origin and the 99% rule, scored-run optimiser and freeze, BN buffers, attack-mask timing, aggregation tensors and Krum \(n>2f+2\), and what a public artefact would contain.

**Result used in the manuscript.** Group-safe Dirichlet (class shares across clients; groups not split across silos; min 20 images/client). Scored freeze is `features` only; AdamW \(10^{-3}\), batch 16, BN running stats still aggregated. Flip mask seed 1337 drawn once; fractions nested. Flower 1.32.1; no public OSF/DOI claimed.
