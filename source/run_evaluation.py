#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import argparse

from rankutils.mappings import ranking_type_map
from rankutils.cfgloader import cfgloader
from rankutils.evaluation import Evaluator

from matplotlib.backends.backend_pdf import PdfPages

def run_evaluation(dataset_choices, evalcfgfile, outprefix):

    pathcfg = cfgloader("/home/alberto/phD/projects/performance_prediction/ret-mr-learning/source/path_2.cfg")
    dbp = cfgloader("/home/alberto/phD/projects/performance_prediction/ret-mr-learning/source/dbparams.cfg")

    pdfMCC = PdfPages(outprefix + "-IRP_MCC.pdf")
    pdfNACC = PdfPages(outprefix + "-IRP_NACC.pdf")
    pdfRPP = PdfPages(outprefix + "-RPP_NACC.pdf")
    #pdfPATK = PdfPages(outprefix + "_patk_correlation.pdf")

    for dataset in dataset_choices:
        for descnum in dataset_choices[dataset]:
            print(". Running Evaluation on", dataset, " -- descriptor", descnum)
            key = "{0:s}_desc{1:d}".format(dataset, descnum)
            evtor = Evaluator(evalcfgfile=evalcfgfile, key=key, pathcfg=pathcfg)

            evtor.evaluate()
            evtor.write_results()
            evtor.draw_irp_results(measure='MCC', outf=pdfMCC)
            evtor.draw_irp_results(measure='NACC', outf=pdfNACC)
            evtor.draw_rpp_results(outf=pdfRPP)

    pdfMCC.close()
    pdfNACC.close()
    pdfRPP.close()
    #pdfPATK.close()
    print("--- Done ---")
    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset to run experiment.",
                        type=str,
                        choices=list(ranking_type_map.keys()) + ['all'])

    parser.add_argument("rkgnum", help="descriptor number. If the descriptor number does not exist for the dataset."
                        "exits with error.",
                        type=int)

    parser.add_argument("evalconfig", help="path to evaluation config file.")

    parser.add_argument("outprefix", help="output figure name prefix.")

    args = parser.parse_args()

    if args.dataset == "all":
        dataset_choices = ranking_type_map
    else:
        if args.descnum in ranking_type_map[args.dataset]:
            dataset_choices = dict()
            dataset_choices[args.dataset] = [args.descnum]
        else:
            print("Unavailable descriptor number {0:d} for dataset {1:s}.".format(args.descnum, args.dataset))
            print("Choise are: ", ranking_type_map[args.dataset], "   Exiting\n---")
            sys.exit(2)

    run_evaluation(dataset_choices, args.evalconfig, args.outprefix)
