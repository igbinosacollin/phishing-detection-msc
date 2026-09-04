# Dissertation evidence map

**Approved title:** *Detecting Phishing URLs and Emails Using Machine Learning:
A Comparative Study of Classical Classifiers*

This is a chapter-planning checklist. It is not dissertation prose and does not
replace the student's own analysis, reflection or Viva preparation.

| Section | Indicative allocation | Evidence that must exist before drafting |
|---|---:|---|
| Abstract | 200–250 words | Final verified question, method, results and conclusion only |
| 1. Introduction | 1,300–1,500 | Current cited problem context, approved aim, original SMART objectives, scope and SEPLi framing |
| 2. Literature review | 3,000–3,500 | Completed source matrix with at least 25 verified papers and critical thematic comparison |
| 3. Methodology and practical design | 2,800–3,200 | Data card, ethics controls, extractor schema, split manifest, five-model configuration and application design |
| 4. Results and evaluation | 2,800–3,200 | Locked-test outputs, uncertainty, figures/tables, SHAP, latency method, error analysis and literature comparison |
| 5. Project evaluation and reflection | 1,800–2,100 | Objective-by-objective evidence, limitations, alternatives, project-management records and student-authored reflection |
| 6. Conclusion and recommendations | 800–1,000 | Direct answer to the approved question, conclusions supported by completed evidence and bounded recommendations |
| References and appendices | excluded from word count where course rules allow | Harvard list, proposal, schedule, meeting log, ethics evidence, source manifest, feature schema, test results and selected technical outputs |

## Required claim discipline

- Describe UCI results as a historical pre-extracted-feature benchmark unless raw
  URLs pass through the same extractor and model end-to-end.
- Do not call Tranco URLs verified legitimate; state the exact candidate-label
  rule and report it as a limitation.
- Do not report a model as best until the selected model, calibrator and threshold
  are locked before the final test.
- Do not present email triage as email-classifier performance unless a separate
  labelled email study has been completed.
- Use only figures and values regenerated from retained run outputs.

## Objective-to-appendix cross-reference

| Objective | Appendix evidence |
|---|---|
| O1 | Search protocol, selection log and literature matrix |
| O2 | Source manifest, data audit and frozen split manifest |
| O3 | Feature schema, unit tests and failure-case examples |
| O4 | Configuration, environment versions and cross-validation outputs |
| O5 | Final evaluation tables, uncertainty method, SHAP and timing records |
| O6 | URL-mode source, test record and screenshot using safe sample input |
| O7 | Email-parser tests, safe fixture messages and security design |
| O8 | Repository README, project schedule, supervision log and ethics evidence |
