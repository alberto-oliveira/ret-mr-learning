#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse

from rankutils.mappings import descriptor_map
from rankutils.cfgloader import cfgloader

from ExperimentManager import ExperimentManager


def run_experiment(dataset_choices, expcfgfile, sval):

    e_manager = ExperimentManager(pathcfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/source/"
                                          "path_2.cfg",
                                  dbparamscfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/"
                                              "source/dbparams.cfg")

    expcfg = cfgloader(expcfgfile)

    for dataset in dataset_choices:
        for descnum in dataset_choices[dataset]:
            e_manager.add_to_experiment_map(dataset, descnum)

    if expcfg['DEFAULT']['type'] == 'stat':
        e_manager.run_statistical_mr(expcfgfile, sampling=sval)
    if expcfg['DEFAULT']['type'] == 'learn':
        e_manager.run_learning_mr(expcfgfile)
    if expcfg['DEFAULT']['type'] == 'stat_pos':
        e_manager.run_stat_positional_mr(expcfgfile)

    print("--- Done ---")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset to run experiment.",
                        type=str,
                        choices=list(descriptor_map.keys()) + ["all"])

    parser.add_argument("descnum", help="descriptor number. If the descriptor number does not exist for the dataset."
                        "exits with error.",
                        type=int)

    parser.add_argument("expconfig", help="path to experiment config file.")
    parser.add_argument("-s", "--sampling", help="Optional sampling value for training sets in weibull mode.",
                        type=float, default=1.0)

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

    run_experiment(dataset_choices, args.expconfig, args.sampling)