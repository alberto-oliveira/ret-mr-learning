#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse

from rankutils.mappings import ranking_type_map
from rankutils.cfgloader import cfgloader

from ExperimentManager import ExperimentManager


def run_experiment(dataset_choices, expcfgfile, sval, ovw):

    e_manager = ExperimentManager(pathcfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/source/"
                                          "path_2.cfg",
                                  dbparamscfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/"
                                              "source/dbparams.cfg")

    expcfg = cfgloader(expcfgfile)
    e_manager.set_experiment_map(dataset_choices)

    #print(e_manager.expmap)

    if expcfg['DEFAULT']['type'] == 'stat':
        e_manager.run_statistical_mr_v2(expcfgfile, sampling=sval, overwrite=ovw)
    if expcfg['DEFAULT']['type'] == 'learn':
        e_manager.run_pos_learning_mr_v2(expcfgfile)
    if expcfg['DEFAULT']['type'] == 'single_learn':
        e_manager.run_single_learning_mr_v2(expcfgfile)
    if expcfg['DEFAULT']['type'] == 'block_learn':
        e_manager.run_block_learning_mr_v2(expcfgfile)
    if expcfg['DEFAULT']['type'] == 'sequence_label':
        e_manager.run_sequence_labeling_mr(expcfgfile)

    print("--- Done ---")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset to run experiment.",
                        type=str,
                        choices=list(ranking_type_map.keys()) + ["all"])

    parser.add_argument("rktpnum", help="Number linked to the ranking approach used for the dataset.",
                        type=int)

    parser.add_argument("expconfig", help="path to experiment config file.")
    parser.add_argument("-s", "--sampling", help="Optional sampling value for training sets in weibull mode.",
                        type=float, default=1.0)

    parser.add_argument("-o", "--overwrite", help="Overwrite existing files.",
                        action="store_true")

    args = parser.parse_args()

    if args.dataset == "all":
        dataset_choices = ranking_type_map
    else:
        if args.rktpnum in ranking_type_map[args.dataset]:
            dataset_choices = dict()
            dataset_choices[args.dataset] = [args.rktpnum]
        elif args.rktpnum == -1:
            dataset_choices = dict()
            dataset_choices[args.dataset] = list(ranking_type_map[args.dataset])
        else:
            print("Unavailable raking-type number {0:d} for dataset {1:s}.".format(args.descnum, args.dataset))
            print("Choices are: ", ranking_type_map[args.dataset], "   Exiting\n---")
            sys.exit(2)

    run_experiment(dataset_choices, args.expconfig, args.sampling, args.overwrite)
