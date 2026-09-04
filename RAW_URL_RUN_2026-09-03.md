# Frozen raw-URL run — 3 September 2026

This is a factual run record, not dissertation prose. The private raw feeds,
predictions, model bundle, and full machine-readable report are deliberately
ignored by Git; they are stored locally under `data/restricted/` and
`artifacts/raw_url/`.

## Input and safeguards

- Requested input: 10,000 PhishTank verified-online URLs and 10,000 Tranco
  top-domain URLs, collected as text only at 13:12 UTC.
- After canonicalisation: 19,891 unique records (9,891 phishing; 10,000
  legitimate-proxy), 109 same-label duplicates removed, and 15,412 registered
  domains.
- PhishTank SHA-256: `14b20b127c5a4282cd6bddb97505871b1a89ff573b1c4901a41c4c90eb3eccf7`.
- Tranco SHA-256: `5d1d073ec8e4b3dad56be769f51f7ca7ce1e48f2c84344e059660f04b590950c`.
- The extractor used 26 deterministic lexical features. It did not fetch URLs,
  use DNS, or use WHOIS.

## Evaluation design

- Outer registered-domain-disjoint locked test: 3,977 URLs from 3,083 domains.
- Development partition: 15,914 URLs; the 11,926-row fit portion was used for
  grouped five-fold cross-validation, while a separate 3,988-row domain-disjoint
  calibration partition chose the final classification threshold.
- Five proposal classifiers were compared: Logistic Regression, Decision Tree,
  Random Forest, SVM (RBF), and XGBoost. The highest grouped-CV F1 selected
  XGBoost; the final threshold was 0.62. Locked-test results did not select the
  model or threshold.

## Locked-test result for the selected model

| Metric | Result | Bootstrap 95% CI |
| --- | ---: | --- |
| Accuracy | 0.9736 | 0.9684–0.9779 |
| Precision | 0.9926 | 0.9886–0.9958 |
| Recall | 0.9540 | 0.9452–0.9622 |
| F1 | 0.9729 | 0.9679–0.9775 |
| ROC-AUC | 0.9949 | 0.9930–0.9966 |
| PR-AUC | 0.9959 | 0.9946–0.9972 |

The threshold produced 91 false negatives and 14 false positives. SHAP was
available for the selected XGBoost model; its leading mean absolute feature was
the number of subdomains, followed by URL length and path length.

## Required limitations for the dissertation

These scores are limited to this data snapshot and label construction.
PhishTank reports verified phishing URLs, but Tranco popularity supplies a
legitimate **proxy**, not proof that every candidate URL is safe. The generated
Tranco URLs use HTTPS at the root and may create a source-construction signal.
The results must not be described as general internet safety, email-classifier
performance, or a guarantee that a low-risk score is safe. A later-time or
independent-source holdout remains the next research validation step.
