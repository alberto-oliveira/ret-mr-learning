#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse

from common.mappings import descriptor_map
from common.cfgloader import cfgloader

from ExperimentManager import ExperimentManager

def run_experiment(dataset, descriptor, expcfgfile):

    e_manager = ExperimentManager(pathcfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/source/"
                                          "path_2.cfg",
                                  dbparamscfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/"
                                              "source/dbparams.cfg")
    e_manager.set_experiment_map([(dataset, descriptor)])

    expcfg = cfgloader(expcfgfile)

    if expcfg['DEFAULT']['type'] == 'wbl':
        e_manager.run_weibull_mr(expcfgfile)
        e_manager.run_irp_to_rpp_conversion(expcfg['DEFAULT']['expname'], 2, 7)
    if expcfg['DEFAULT']['type'] == 'lrn':
        e_manager.run_learning_mr(expcfgfile)
        e_manager.run_irp_to_rpp_conversion(expcfg['DEFAULT']['expname'], 2, 7)

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

    parser.add_argument("expconfig", help="path to experiment config file.")

    args = parser.parse_args()

    run_experiment(args.dataset, args.descnum, args.expconfig)