
.PHONY: setup ingest train demo ui

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt || true

ingest:
\tpython manage.py ingest-csv --csv ${CSV} --out_dir data/signs/jsonl

train:
\tpython manage.py train --data data/signs/jsonl --out models/sign_dtw.pkl --T 32 --k 3

demo:
\tpython manage.py demo

ui:
\tpython manage.py ui
