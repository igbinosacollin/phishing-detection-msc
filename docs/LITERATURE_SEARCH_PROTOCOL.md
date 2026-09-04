# Literature-search protocol

This is an evidence-management record for Objective O1. It is not dissertation
prose and should be completed only with sources the student has read and can
explain in the Viva.

## Review question

What evidence supports or challenges classical machine-learning approaches for
detecting phishing URLs and emails, particularly where models use lexical,
host-based, HTML/content, reputation, or visual features?

## Databases and search record

Search IEEE Xplore, ACM Digital Library, Scopus, Web of Science and Google
Scholar. Record the date, exact query, returned count and selection decision for
each search. Suggested terms include:

```text
(phishing OR malicious URL OR phishing website) AND
(machine learning OR classifier OR XGBoost OR random forest OR SVM) AND
(URL OR lexical OR feature engineering OR deployment OR generalisation)
```

## Eligibility rules

- Peer-reviewed journal, conference or systematic-review paper.
- Published 2018–2025 for the core review; earlier seminal work may be included
  where it defines a dataset, feature family or method used by the project.
- Relevant to phishing URLs, phishing websites or email-to-URL detection.
- Full text and bibliographic details can be verified.
- Exclude duplicate records, non-phishing malicious-URL studies unless clearly
  labelled as such, unverifiable citations and papers without a clear method.

## Required evidence matrix fields

Use `literature_matrix_template.csv` to record at least 25 eligible papers.
Do not count a paper until the DOI/landing page, method and reported limitation
have been checked.

## Critical synthesis prompts

For each thematic section, compare rather than list papers:

1. Which feature types are available before a user opens a link?
2. Does the study use raw URLs, pre-extracted features, live page content or
   third-party reputation services?
3. Are train and test data separated by time, registered domain or source?
4. Which metrics and error costs are reported?
5. What makes the reported result transferable—or not—to the proposed two-input
   URL and email-triage system?
