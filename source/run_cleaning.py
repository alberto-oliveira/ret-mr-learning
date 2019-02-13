# /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import glob
import shutil
import argparse

from rankutils.cfgloader import cfgloader


def run_cleaning(expcfgfile, dryrun, resonly):

    expcfg = cfgloader(expcfgfile)
    pathcfg = cfgloader("path_2.cfg")

    dellist = []

    outfvdirname = expcfg.get('DEFAULT', 'expname')
    clf = expcfg.get('IRP', 'classifier', fallback='')

    if clf != '':
        outdirname = "{0:s}.{1:s}".format(outfvdirname, clf)
    else:
        outdirname = outfvdirname

    prevdset = ''

    print("-- The listed files/directories will be cleaned:")
    for s in pathcfg:
        if s != "DEFAULT":

            dset = s.split('_')[0]
            if dset != prevdset:
                print('.', dset)
                prevdset = dset

            print("    |_", pathcfg.get(s, 'rktpdir'))

            if not resonly:
                try:
                    #print('{0:s}{1:s}'.format(pathcfg.get(s, 'feature'), outfvdirname))
                    fvdir = glob.glob('{0:s}*{1:s}*'.format(pathcfg.get(s, 'feature'), outfvdirname))[0]
                    aux = fvdir.rsplit('/', 3)
                    print("        |_ {0:s}".format('/'.join(aux[1:])))
                    dellist.append(fvdir)
                except IndexError:
                    pass

            try:
                #print('{0:s}*{1:s}*'.format(pathcfg.get(s, 'output'), outdirname))
                outdir = glob.glob('{0:s}*{1:s}*'.format(pathcfg.get(s, 'output'), outdirname))[0]
                aux = outdir.rsplit('/', 3)
                print("        |_ {0:s}".format('/'.join(aux[1:])))
                dellist.append(outdir)
            except IndexError:
                pass

    print()
    ans = None
    while ans != 'y' and ans != 'n':

        ans = input("-- Do you wish the delete the files (yY/nN): ").lower()

    if ans == 'y':
        for p in dellist:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
        print("-- deleted!")

    print('...exiting')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("expconfig", help="path to experiment config file.", type=str)
    parser.add_argument("-n", "--dryrun", help="list directories that will be cleaned.", action="store_true")
    parser.add_argument("-r", "--resonly", help="Do not clean features, only results", action="store_true")

    args = parser.parse_args()

    run_cleaning(args.expconfig, args.dryrun, args.resonly)
