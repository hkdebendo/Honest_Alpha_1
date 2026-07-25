PY = python

all: data features run report

data:
	$(PY) -m src.run data

features:
	$(PY) -m src.run features

run:
	$(PY) -m src.run run

ablation:
	$(PY) -m src.run ablation

controls:
	$(PY) -m src.run controls

report:
	$(PY) -m src.run report

test:
	$(PY) -m pytest tests -v

.PHONY: all data features run ablation controls report test
