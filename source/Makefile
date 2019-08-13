help:
	@cat Makefile

PATHCFG?="./path_2.cfg"
DBPARAMS?="./dbparams.cfg
DATASET?=all
RKTPNUM?=1
CFGDIR?="./expcfg/"
CFGPATH?=$(CFGDIR)$(CFGNAME)
OUTROOT?="../output/2x5-fold/"
ROOTPATH?="./"

extraction: run_extraction.py
	python run_extraction.py $(DATASET) $(RKTPNUM) $(CFGPATH) -r $(ROOTPATH)

experiment: run_experiments.py
ifndef SAMPLE
	python run_experiments.py $(DATASET) $(RKTPNUM) $(CFGPATH) $(OVW) -r $(ROOTPATH)
else
	python run_experiments.py $(DATASET) $(RKTPNUM) $(CFGPATH) -s $(SAMPLE) -r $(ROOTPATH) $(OVW)
endif

pipeline: extraction experiment

baseline: create_baselines.py
	python create_baselines.py $(DATASET) $(RKTPNUM) $(BSLNAME) $(OUTFOLDER)

.PHONY: view list clean

view:
	@cat $(CFGPATH)

list:
	@ls $(CFGDIR)