# /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import glob
import shutil
import argparse

from rankutils.cfgloader import cfgloader

def TEST(pathlist):

    for path in pathlist:
	print(path)
	
	return

def run_cleaning(expcfgfile, newname):

    expcfg = cfgloader(expcfgfile)
    pathcfg = cfgloader("path_2.cfg")

    renlist = []

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

            try:
                #print('{0:s}*{1:s}*'.format(pathcfg.get(s, 'output'), outdirname))
                outdir = glob.glob('{0:s}*{1:s}*'.format(pathcfg.get(s, 'output'), outdirname))[0]
                aux = outdir.rsplit('/', 3)
                print("        |_ {0:s}".format('/'.join(aux[1:])))
                renlist.append(outdir)
            except IndexError:
                pass

    print()
    ans = None
    while ans != 'y' and ans != 'n':

        ans = input("-- Do you wish the rename the folders (yY/nN): ").lower()

    if ans == 'y':
        for p in renlist:
            parts = os.path.split(p)
            print(os.path.join(parts[0], newname))
            shutil.move(p, os.path.join(parts[0], newname))
        print("-- deleted!")

    print('...exiting')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("expconfig", help="path to experiment config file.", type=str)
    parser.add_argument("newname",  help="new name for output files.", type=str)

    args = parser.parse_args()

    run_cleaning(args.expconfig, args.newname)
