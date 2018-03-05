#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse

from common.mappings import descriptor_map
from common.cfgloader import cfgloader

from ExperimentManager import ExperimentManager

def run_experiment(dataset_choices, expcfgfile):

    e_manager = ExperimentManager(pathcfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/source/"
                                          "path_2.cfg",
                                  dbparamscfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/"
                                              "source/dbparams.cfg")

    expcfg = cfgloader(expcfgfile)

    for dataset in dataset_choices:
        for descnum in dataset_choices[dataset]:
            e_manager.add_to_experiment_map(dataset, descnum)

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

    if args.dataset == "all":
        dataset_choices = descriptor_map
    else:
        if args.descnum in descriptor_map[args.dataset]:
            dataset_choices = dict()
            dataset_choices[args.dataset] = [args.descnum]
        else:
            print("Unavailable descriptor number {0:d} for dataset {1:s}.".format(args.descnum, args.dataset))
            print("Choise are: ", descriptor_map[args.dataset], "   Exiting\n---")
            sys.exit(2)

    run_experiment(dataset_choices, args.expconfig)