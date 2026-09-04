.PHONY: test smoke train app webhook

test:
	python -m unittest discover -s tests -v

smoke:
	python make_smoke_dataset.py --output data/smoke_raw_urls.csv
	python train_raw_url_model.py --input data/smoke_raw_urls.csv --output-dir artifacts/smoke --quick --bootstrap-iterations 50

train:
	python acquire_raw_url_data.py --output-dir data/restricted --per-class 10000
	python train_raw_url_model.py --input data/restricted/raw_url_input.csv --output-dir artifacts/raw_url

app:
	streamlit run streamlit_app.py

webhook:
	python run_email_webhook.py
