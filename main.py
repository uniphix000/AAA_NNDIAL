#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import argparse
from datautils import *
from module import *











def main():
    # cmd
    cmd = argparse.ArgumentParser('NNDIAL')
    cmd.add_argument('--config', default='./config/NDM.cfg')
    cmd.add_argument('--mode', type = str, default='train')
    cmd.add_argument('--embed_size', type = int, default=200)
    cmd.add_argument('--hidden_size', type = int, default=200)
    cmd.add_argument('--max_epoch', type = int, default=200)



    args = cmd.parse_args()

    # configuration
    config = Configuration(args)

    # load dataset
    datareader = DataReader(config.corpusfile, config.dbfile, config.semidictfile, config.ontologyfile,
        config.split, config.lengthen, config.percent,
        config.shuffle, config.trk_enc, config.verbose, args.mode, config.policy,
        config.latent)

    model = Network(args, datareader)
    model.train()


























if __name__ == '__main__':
    main()