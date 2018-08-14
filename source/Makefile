help:
	@cat Makefile

PATHCFG?="./path_2.cfg"
DBPARAMS?="./dbparams.cfg
DATASET?=all
DESCRIPTOR?=1
CFGDIR?="./expcfg/"
CFGPATH?=$(CFGDIR)$(CFGNAME)

extraction: run_extraction_old.py
	python run_extraction.py $(DATASET) $(DESCRIPTOR) $(CFGPATH)

experiment: run_experiments.py
ifndef SAMPLE
	python run_experiments.py $(DATASET) $(DESCRIPTOR) $(CFGPATH)
else
	python run_experiments.py $(DATASET) $(DESCRIPTOR) $(CFGPATH) -s $(SAMPLE)
endif

baseline: create_baselines.py
	python create_baselines.py $(DATASET) $(DESCRIPTOR) $(BSLNAME) $(OUTFOLDER)


.PHONY: view list

view:
		@cat $(CFGPATH)

list:
		@ls $(CFGDIR)