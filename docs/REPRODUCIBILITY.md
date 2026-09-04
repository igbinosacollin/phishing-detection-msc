# Reproducibility guide

## Scope

The repository currently contains a UCI benchmark baseline and raw-URL research
scaffolding. Do not treat the stored model files as a complete contemporary
application evaluation until a frozen raw-URL dataset has passed the documented
protocol in `RESEARCH_PROTOCOL.md`.

## Environment

Create an isolated environment and install the development requirements:

```bash
cd phishing-detection-msc
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

Record the exact package versions, operating-system details, Python version,
Git commit and random seed in every experiment run. A package lock file should
be generated only after the raw-URL experiment configuration is agreed and has
been run successfully.

## Network-free checks

Run the test suite without creating a local pytest cache:

```bash
python -m pytest -q -p no:cacheprovider tests
```

The extractor tests use safe test URLs and do not load any web page. The raw-data
tests use temporary synthetic rows only.

## Raw-URL preparation

After approved source snapshots are stored outside Git, validate and split them:

```bash
python raw_url_data.py \
  --input /safe/local/path/approved_raw_urls.csv \
  --output-dir /safe/local/path/research_outputs
```

Keep the resulting final test partition locked. Do not run model selection,
calibration or threshold tuning against it.
