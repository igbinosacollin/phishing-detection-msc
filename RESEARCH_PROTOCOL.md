# Research-completion protocol

## Approved project scope

**Title:** *Detecting Phishing URLs and Emails Using Machine Learning: A
Comparative Study of Classical Classifiers*

This protocol retains the approved title and proposal objectives. It separates
the existing UCI result from the evidence required to support a practical URL
and email-triage system.

## What the existing results support

The stored Full-30, Deployable-13 and Lexical-9 results are useful benchmark
ablations over a historic UCI feature table. They do **not** demonstrate
end-to-end performance of `url_features.extract()` on current raw URLs. Until
the extractor and trained model are evaluated together on raw URLs, avoid claims
such as "the deployed app delivers F1 0.7805".

## Evidence required before writing results

1. Obtain supervisor approval for the exact raw-URL data sources, storage and
   safety controls. Do not load or visit suspect pages.
2. Follow the proposal's data direction: use a frozen PhishTank snapshot for
   verified phishing URLs and a frozen Tranco list for legitimate candidates.
   Record a permanent Tranco list identifier and do not describe popularity as
   proof that every URL is benign. Preserve the source, collection/verification
   time, label rationale, licence/terms and checksum.
3. Use `raw_url_data.prepare_raw_url_data()` to reject conflicting labels,
   detect duplicates and create a data audit.
4. Create a registered-domain-disjoint outer test partition. Restrict model,
   calibrator and threshold selection to the development partition.
5. Train the five proposal models through the same shared lexical extractor used
   by the application. Treat live WHOIS/DNS as a separately reported optional
   condition, including failures and end-to-end latency.
6. Report fixed-model ablations as well as best-in-condition results. Include
   precision, recall, F1, ROC-AUC, PR-AUC, calibration, error counts, bootstrap
   confidence intervals, and measured end-to-end latency.
7. Keep a later-time or independent-source holdout when sufficient data permits.
   Do not report that result until all decisions are locked.
8. Build and test the Streamlit URL mode with safe textual inputs. Build the
   email mode only after the URL model is frozen and document its parsing and
   failure behaviour.

## Data-source guardrails

- PhishTank data should be retained as a dated, hashed snapshot with the feed
  fields used to establish the label. Never make the feed source, target brand,
  reputation status or collection timestamp into a model feature.
- Tranco provides a reproducible ranked-domain list, not a ground-truth benign
  URL corpus. Record why and how a candidate URL was selected, and exclude it
  if any label conflict is discovered.
- The original UCI feature table remains a baseline comparison only. It cannot
  replace a raw URL dataset in the application experiment.

## Proposal-objective evidence map

| Objective | Evidence to retain |
|---|---|
| O1 literature review | Search strings, databases, inclusion/exclusion log, at least 25 verified papers, evidence matrix |
| O2 data | Frozen source snapshots, data card, cleaning/audit report, class and source balance |
| O3 feature engineering | Shared feature code, unit tests, feature schema and failure cases |
| O4 five models | Fixed configuration file, cross-validation outputs and environment versions |
| O5 evaluation | Locked-test report, metrics, uncertainty, SHAP and end-to-end timing methodology |
| O6 Streamlit | Source, screenshots, functional test record and known limitations |
| O7 email mode | Webhook source, safe sample messages, parsing tests and security controls |
| O8 dissertation/repository/demo | Git history, README, appendices, schedule, meeting log and ethics evidence |

## Dissertation authoring guardrails

Write only from completed, traceable evidence. The dissertation must be the
student's own work and argument. This file is a research protocol, not
submission-ready chapter prose.
