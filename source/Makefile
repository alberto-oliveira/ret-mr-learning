help:
	@cat Makefile

PATHCFG?="./path_2.cfg"
DBPARAMS?="./dbparams.cfg
DATASET?="all"
DESCRIPTOR?="1"

extraction: run_extraction.py
	python run_extraction.py $(DATASET) $(DESCRIPTOR) $(EXPCFG)

experiment: run_experiments.py
	python run_experiments.py $(DATASET) $(DESCRIPTOR) $(EXPCFG)

evaluation: run_experiments.py
	python run_experiments.py $(DATASET) $(DESCRIPTOR) $(EXPCFG)

