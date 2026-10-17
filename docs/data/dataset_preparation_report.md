# Phase 4 dataset inspection and preparation report

Date: 2026-10-17  
Pipeline version: 1.0.0  
Random seed: 20261017  
Status: complete for the supplied sources

## Scope and outcome

Eight user-supplied sources were downloaded and inspected. Seven contain labeled text or claims; BS Detector contains source-category metadata and unlabeled posts, so it was prepared only as an auxiliary source-credibility resource. The arXiv link describes BharatGather; the released dataset referenced by that work was obtained from the authors' Hugging Face organization.

Raw downloads remain unchanged under `data/raw/_archives/`. Extracted working copies, exclusions, and processed datasets are ignored by Git. No dataset was merged into a single training corpus because task granularity, label meaning, provenance, and licenses differ.

## Received sources

| Supplied link | Prepared as | License/status used |
| --- | --- | --- |
| [ISOT on Kaggle](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset/data) | ISOT article dataset | Kaggle metadata: MIT; underlying article rights noted |
| [Indian Fake News on Kaggle](https://www.kaggle.com/datasets/imbikramsaha/fake-real-news) | Indian Fake News text dataset | CC0 |
| [BS Detector dataset](https://github.com/thiagovas/bs-detector-dataset) | Auxiliary source categories and unlabeled posts | LGPL-3.0 repository license |
| [Fake-News-Classification](https://github.com/shivani-tufchi/Fake-News-Classification) | FRI article dataset | No license found |
| [LIAR mirror](https://github.com/tfs4/liar_dataset) | LIAR claim dataset | Research only; original-source copyright |
| [BharatFakeNewsKosh on Kaggle](https://www.kaggle.com/datasets/man2191989/bharatfakenewskosh) | BharatFakeNewsKosh workbook | CC BY-NC 4.0 |
| [BharatGather paper](https://arxiv.org/html/2609.02895v1#Sx4.SS1) | [BharatGather release](https://huggingface.co/datasets/l3cube-pune/BharatGather) | CC BY 4.0 |
| [IFND repository](https://github.com/sonalgarg174/Dataset) | IFND statement dataset | Research only |

## Prepared data summary

| Dataset | Raw records | Prepared | Excluded | Train | Validation | Test | Binary fake mapping |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ISOT | 44,898 | 38,829 | 6,069 | 31,063 | 3,883 | 3,883 | Verified from file membership |
| Indian Fake News | 3,729 | 2,229 | 1,500 | 1,783 | 223 | 223 | Verified from FAKE/REAL labels |
| FRI | 9,999 | 9,828 | 171 | 7,861 | 984 | 983 | Not verified; raw 0/1 retained |
| IFND | 56,714 | 55,018 | 1,696 | 44,014 | 5,502 | 5,502 | Verified from repository README |
| BharatFakeNewsKosh | 26,232 | 25,106 | 1,126 | 20,084 | 2,511 | 2,511 | Not verified; booleans retained |
| BharatGather | 14,470 | 11,216 | 3,254 | 7,619 | 848 | 2,749 | Verified: raw 0=false, 1=true |
| LIAR | 12,791 | 12,749 | 42 | 10,200 | 1,283 | 1,266 | Six labels preserved; no binary collapse |
| BS Detector | 874 source rows; 21,313 posts | 874 source rows; 21,313 posts | 0 | n.a. | n.a. | n.a. | Auxiliary metadata, not target labels |

The prepared total for labeled text/claim datasets is 154,975 records. The BS Detector tables remain separate and unsplit.

## Inspection findings

### ISOT

- Two English article files contain 23,481 fake and 21,417 real raw records.
- Normalized text analysis removed 6,069 duplicate records. Five additional near-duplicate clusters were kept intact within splits.
- The truthful class is sourced from Reuters while fake articles come from other outlets. This strong source-label shortcut can inflate in-domain performance and must be tested with source-held-out evaluation before model claims are made.
- Kaggle reports an MIT license for the package, but underlying article copyright and the original authors' citation requirements still apply.

### Indian Fake News

- Raw labels are nearly balanced (1,877 FAKE; 1,852 REAL), but 1,492 normalized duplicates and eight empty texts were excluded.
- The prepared set is strongly imbalanced (1,851 FAKE; 378 REAL), showing that repeated real records created much of the apparent raw balance.
- The Kaggle package is marked CC0.

### FRI

- Three missing article bodies and 168 normalized duplicates were excluded.
- The repository provides no license file and does not define whether `0` or `1` means fake. The pipeline therefore leaves `label_binary_fake` blank.
- The README says the collected data were imbalanced, while the supplied CSV is approximately balanced. This discrepancy needs author clarification before experimental use.

### IFND

- The file contains 56,714 India-focused statements with labels `TRUE` and `Fake`; 11,321 dates are missing.
- Normalized text found 522 statement groups carrying conflicting labels. All 1,044 records in those groups were quarantined rather than relabeled. Another 652 duplicate records were excluded.
- The source is title/statement level rather than full-article text. It should not be treated as interchangeable with ISOT articles.
- The repository limits use to research and requires dataset citations.

### BharatFakeNewsKosh

- The primary `A` sheet has 26,232 records and 19 fields. `Sheet1` is malformed staging data and `Sheet3` is empty, so neither was included.
- The workbook contains 15,913 `True` and 10,319 `False` values, but the Kaggle description reports 13,721 legitimate and 12,511 fraudulent examples. Because those counts and semantics do not reconcile, no binary mapping was asserted.
- There were 244 normalized text groups with conflicting booleans (488 records) and 638 additional duplicates.
- License: CC BY-NC 4.0. Commercial use is not permitted by that license.

### BharatGather

- The downloaded release contains 11,576 train and 2,894 test records (14,470 total). The dataset card states 14,469, while the arXiv paper states 14,646; the files are treated as authoritative for this run.
- The release includes web-scraped, YouTube, LLM-extracted, and synthetic records. Synthetic provenance must remain available during later evaluation.
- Normalized deduplication removed 3,231 records. Twenty-three train-side records from near-duplicate clusters spanning the official test split were excluded. A deterministic 10% validation subset was derived from the remaining official train data; the evaluation-side record was retained when cross-split clusters occurred.
- License: CC BY 4.0.

### LIAR

- The supplied official train/validation/test files contain 12,791 short claims and six truthfulness labels.
- Seven normalized text groups carried conflicting labels (16 records); 24 additional duplicates and two train-side near-duplicate records crossing an evaluation split were excluded.
- The six labels remain unchanged. A binary fake/real mapping would be a new research choice and was not made.
- The repository states research-only use and notes that original sources retain copyright.

### BS Detector

- `data_final.csv` is a 874-domain source-category list, not article-level ground truth. `posts_data.csv` has 21,313 unlabeled link records.
- Both tables were normalized structurally and left unsplit. Treating the source list as truth labels would introduce circular source bias.
- The repository contains an LGPL-3.0 license.

## Cleaning and split policy

The pipeline applies only model-neutral cleanup: Unicode NFKC, HTML entity decoding, markup removal, invalid control-character removal, and whitespace normalization. Original and cleaned text are both retained. Case, punctuation, URLs, numbers, entities, and stop words are not removed.

Exact duplicate keys use case-folded, punctuation-normalized text with URLs represented consistently. Empty text is excluded. Every record in a normalized text group with conflicting labels is quarantined. Same-label duplicates keep one deterministic representative.

Potential near duplicates are clustered with a 64-bit SimHash over a bounded sample of normalized word trigrams; the maximum Hamming distance is 3. Clusters are assigned wholly to one generated split. This heuristic reduces obvious leakage but does not prove event-level independence.

For sources without official splits, approximately stratified, cluster-aware 80/10/10 splits were generated. LIAR's official split names were retained after leakage exclusions. BharatGather's official test split was retained, and validation was derived from its remaining train set. The evaluation split takes precedence when a detected cluster crosses an official boundary.

## Cross-source overlap

The datasets were kept separate. Exact normalized overlap after within-source cleaning found:

- 266 common texts between IFND and BharatFakeNewsKosh. IFND labels all of these as `Fake`, while BharatFakeNewsKosh contains both boolean values (122 `False`, 144 `True`).
- One common text between FRI label `1` and Indian Fake News label `FAKE`. A single overlap is insufficient to establish FRI label semantics.

These conflicts are evidence against blindly concatenating the corpora.

## Verification results

- Nine automated tests passed on Python 3.14.5.
- The real-output validator passed all 10 raw archive hashes, 32 prepared-output hashes, and all eight dataset output groups.
- Cleaned files reload successfully with the expected columns and record counts.
- No empty cleaned text, normalized exact duplicates, record-ID overlap, or detected near-duplicate cluster overlap remains across generated splits.
- The complete preparation was rerun with seed 20261017. The manifest SHA-256 was identical before and after: `e634e8fd0d1df418ded243c40cc82e2e629bcb47b70421712ac649537f9abf4f`.
- The raw archives were rehashed after both runs and remained unchanged.

## Raw archive checksums

| Raw file | Bytes | SHA-256 |
| --- | ---: | --- |
| `data/raw/_archives/github/bs-detector-dataset.zip` | 998,587 | `b12cc0a213a02ff0078a61f263d4b6520a13e63fe4aeecdb6735ad2de50fb22f` |
| `data/raw/_archives/github/fake-news-classification.zip` | 9,750,078 | `d40674678d4b6955431054a6032fdd7ceed8d919aaf43e8ddc8894fa512a78b6` |
| `data/raw/_archives/github/ifnd-dataset.zip` | 3,000,556 | `ef349be0841a9f92bf7d69ec616a0dc39be09d56e8e5903974025771f32e3657` |
| `data/raw/_archives/github/liar-dataset.zip` | 1,167,507 | `3bef92b8c62d22b8b6632ebd8ba72710f5ac60b07e557acc75e78e18fc2e4fab` |
| `data/raw/_archives/huggingface/bharatgather-README.md` | 5,795 | `2480d8e2b891911b3d32a64df95310abc4eefa1953c0b74eed7260a4daff654d` |
| `data/raw/_archives/huggingface/bharatgather-test.json` | 14,859,515 | `4c6194051773247bef1f1326c7747916a9841348bca72db4768dde161639ddf3` |
| `data/raw/_archives/huggingface/bharatgather-train.json` | 60,221,803 | `0d68c47c24d57178d486a57dfa2c3adf62c6ec8a9d9ac65e13a265778b91a5d4` |
| `data/raw/_archives/kaggle/bharatfakenewskosh.zip` | 10,735,967 | `330feb9b24b48f49c4113ca6de87695f1bd0821d29361a86e2b27ab35a77c69c` |
| `data/raw/_archives/kaggle/indian-fake-news.zip` | 3,697,394 | `d245810ba20471d04b18d5e9ae0a278ba23546ce4ef6f204527380951456094c` |
| `data/raw/_archives/kaggle/isot-fake-news-dataset.zip` | 42,975,963 | `3574ba338715ab019606e6f95c2f7b7ddf7c23d4617728c3cb1fdf4941b42435` |

## Remaining decisions and limitations

- Confirm FRI label semantics and licensing before using it for binary classification.
- Resolve BharatFakeNewsKosh's boolean-label/count discrepancy with authoritative documentation before mapping it.
- Choose the final research corpus and evaluation design. The prepared datasets are candidates, not an endorsed combined benchmark.
- Add event-, time-, and source-held-out evaluations where metadata supports them; text-similarity clustering alone cannot prevent all semantic leakage.
- Decide whether synthetic BharatGather examples belong in training, ablation-only analysis, or a separate robustness experiment.
- Respect each source's license and citation requirements. The most restrictive applicable terms govern any combined distribution.

No classifier, transformer, RAG, NLI, fusion model, LLM explanation service, production backend, or frontend was implemented in Phase 4.
