# Phishing URL Detection Under Deployment Constraints

Code for an MSc Cybersecurity dissertation (PROM02, University of Sunderland)
comparing five classical classifiers for phishing URL detection, and measuring what
it costs to refuse the information a deployed tool cannot ethically obtain.

## The finding

The suspect page is never fetched. Loading a suspected phishing page runs
attacker-controlled code, confirms the address is live, and risks retrieving malware.
Zhang et al. (2021) also show that many phishing sites serve benign content to
anything resembling a crawler, so a fetched page may be deliberately misleading.

Auditing the 30 UCI benchmark features against that constraint leaves **13 of 30**
obtainable: 9 from the URL text, 4 from WHOIS and DNS metadata about the domain.

| Feature set | Best model | Accuracy | F1 | 95% CI (F1) |
|---|---|---|---|---|
| Full 30 (benchmark) | Random Forest / XGBoost | 0.9769 | 0.9738 | 0.9662–0.9806 |
| **Deployable 13** | **XGBoost** | 0.8042 | **0.7805** | 0.7595–0.7994 |
| Lexical 9 (offline) | SVM (RBF) | 0.7440 | 0.7305 | 0.7083–0.7520 |

The gap is 0.1933 F1, significant at p = 4.7e-90 (McNemar). The three most important
features carry 64.8% of the model's decision weight and none is obtainable without
loading the page.

Random Forest and XGBoost are statistically indistinguishable on the full set
(McNemar p = 1.000, 8 disagreements each way), so the selection is made on latency
and interpretability rather than on F1.

## A negative result worth reading

A second model trained on raw URLs scored **0.9729 F1** on a domain-disjoint locked
test set, and is not used. Its legitimate class was built from Tranco domains as bare
`https://<domain>/` roots while its phishing class was real PhishTank URLs, so it
learned to separate short addresses from long ones. It classifies every real
legitimate URL carrying a path as phishing with 100% confidence. Reproduce it:

    python predict.py --artefact "https://www.bbc.co.uk/news/technology-68351312"

## Live demo

Deployed on Streamlit Community Cloud. Note that WHOIS speaks on TCP port 43, which
many hosting providers block outbound. Where that happens the four host-based
features report as unavailable and the model falls back to the nine lexical features,
which measures 0.7305 F1 rather than 0.7805. The interface says so on screen rather
than presenting a weaker verdict as if nothing had changed.

## Running it

    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

    python predict.py "https://example.com/login"     # CLI
    python predict.py --no-network "https://..."      # skip WHOIS/DNS
    streamlit run app.py --server.port 8534           # web app, 3 input modes
    python email_monitor.py --demo                    # email mode, no mailbox needed
    pytest tests -q                                   # 43 tests

Credentials for the email mode are read from environment variables; see `.env.example`.
Nothing is stored in this repository.

## Reproducing the results

    python train_deployable.py    # three-tier comparison
    python stats_analysis.py      # bootstrap CIs, McNemar, PR curves
    python make_figures.py        # every figure
    python tune_threshold.py      # operating point analysis

The UCI dataset is fetched by identifier at runtime, so no local copy is needed.

## Limitations

Trained on data collected in 2014. One feature encoding certificate status carries
32.6% of the model's decision weight, and free automated certificate authorities have
since made TLS near universal. WHOIS is frequently rate-limited or withheld (`.uk`
returns nothing), so the deployed system often degrades toward the 9-feature tier.
No adversarial robustness testing was performed; AlEroud and Karabatis (2020) show
GANs can evade classifiers of this kind.

## Licence

Code released for academic assessment. The UCI Phishing Websites Dataset is
distributed by its own authors under their terms.
