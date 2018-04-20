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
    cmd.add_argument('--encoder_type', type = str, default='lstm')
    cmd.add_argument('--lr', type = float, default=1)





    args = cmd.parse_args()

    # configuration
    config = Configuration(args)

    # load dataset
    datareader = DataReader(config.corpusfile, config.dbfile, config.semidictfile, config.ontologyfile,
        config.split, config.lengthen, config.percent,
        config.shuffle, config.trk_enc, config.verbose, args.mode, config.policy,
        config.latent)

    model = Network(args, datareader)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    # for i in range(args.max_epoch):
    #     logging.info('--------------------Round {0}---------------------'.format(i))
    #     while True:
    #         data = model.datareader.read(mode='train')
    #         if data == None:
    #             break
    #
    #         '''
    #         source_0, source_len_1, masked_source_2, masked_source_len_3,\
    #         target_4, target_len_5, masked_target_6, masked_target_len_7,\
    #         snapshot_8, change_9, goal_10, inf_trk_label_11, req_trk_label_12,\
    #         db_degree_13, srcfeat_14, tarfeat_15, finished_16, utt_group_17 = data  # 除了goal和finished以外长度都为对话轮次长
    #         '''
    #
    #         # zero_grad
    #         model._zero_grad()
    #
    #         dialogue_loss = model.recurr(data)
    #         print dialogue_loss
    #         # backward
    #         dialogue_loss.backward()
    #
    #         optimizer.step()


























if __name__ == '__main__':
    main()