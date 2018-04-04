#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import configparser

def cfgloader(cfgfile):
    """

    Loads a config file into a ConfigParser object.

    :param cfgfile: path to the configuration file.
    :return: ConfigParser object.
    """

    if os.path.isfile(cfgfile):
        config = configparser.ConfigParser()
        config.read(cfgfile)
    else:
        raise ValueError("<{0:s}> is not a file".format(cfgfile))

    return config

""" Tests the loader """
if __name__ == "__main__":

    cfgfile = sys.argv[1]

    config = config_loader(cfgfile)

    for s in config.sections():
        for opt in config[s]:
            if opt not in config['DEFAULT']:
                print("{0:s} = {1:s}".format(opt, config[s][opt]))

        print("---")
