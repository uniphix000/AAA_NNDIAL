#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import argparse
from datautils import *











def main():
    # cmd
    cmd = argparse.ArgumentParser('NNDIAL')
    cmd.add_argument('--config', default='./config/NDM.cfg')
    cmd.add_argument('--mode', type = str, default='train')

    args = cmd.parse_args()

    # configuration
    config = Configuration(args)

    # load dataset
    datareader = DataReader(config.corpusfile, config.dbfile, config.semidictfile, config.ontologyfile,
        config.split, config.lengthen, config.percent,
        config.shuffle, config.trk_enc, config.verbose, args.mode, config.policy,
        config.latent)


























if __name__ == '__main__':
    main()