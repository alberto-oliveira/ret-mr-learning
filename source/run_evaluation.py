#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse

from common.mappings import descriptor_map
from common.cfgloader import cfgloader
from common.evaluation import Evaluator

from ExperimentManager import ExperimentManager

def run_evaluation(dataset, descriptor, evalcfgfile):

    pathcfg = cfgloader("/home/alberto/SpotME/projects/performance-prediction/sources/"
                      "ret-mr-learning/source/path.cfg")

    key = "{0:s}_desc{1:d}".format(dataset, descriptor)
    evtor = Evaluator(evalcfgfile=evalcfgfile, key=key, pathcfg=pathcfg)

    evtor.evaluate()
    evtor.write_results()

    print("--- Done ---")
    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset to run experiment.",
                        type=str,
                        choices=list(descriptor_map.keys()))

    parser.add_argument("descnum", help="descriptor number. If the descriptor number does not exist for the dataset."
                        "exits with error.",
                        type=int)

    parser.add_argument("evalconfig", help="path to evaluation config file.")

    args = parser.parse_args()

    run_evaluation(args.dataset, args.descnum, args.evalconfig)
