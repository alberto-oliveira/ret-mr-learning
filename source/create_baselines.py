#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse

from rankutils.mappings import ranking_type_map, baseline_map
from rankutils.utilities import completedir

from ExperimentManager import ExperimentManager


def create_baselines(dataset_choices, baseline, outfolder):

    e_manager = ExperimentManager(pathcfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/source/"
                                          "path_2.cfg",
                                  dbparamscfg="/home/alberto/phD/projects/performance_prediction/ret-mr-learning/"
                                              "source/dbparams.cfg")

    e_manager.set_experiment_map(dataset_choices)

    e_manager.run_baselines(baseline, 10, outfolder)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset to run experiment.",
                        type=str,
                        choices=list(ranking_type_map.keys()) + ["all"])

    parser.add_argument("rktpnum", help="descriptor number. If the descriptor number does not exist for the dataset."
                        "exits with error.",
                        type=int)

    parser.add_argument('baseline', help='Baseline name', choices=list(baseline_map.keys()), type=str)
    parser.add_argument('outfolder', help='Output folder name', type=str)

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

    create_baselines(dataset_choices, args.baseline, completedir(args.outfolder))