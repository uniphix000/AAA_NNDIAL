#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import argparse
from datautils import *
from module import *
from loader.GentScorer import *
import numpy as np











def main():
    # cmd
    cmd = argparse.ArgumentParser('NNDIAL')
    cmd.add_argument('--config', default='./config/NDM.cfg')
    cmd.add_argument('--mode', type = str, default='train')
    cmd.add_argument('--embed_size', type = int, default=500)
    cmd.add_argument('--hidden_size', type = int, default=500)
    cmd.add_argument('--max_epoch', type = int, default=5000)
    cmd.add_argument('--encoder_type', type = str, default='lstm')
    cmd.add_argument('--lr', type = float, default=0.01)





    args = cmd.parse_args()

    # configuration
    config = Configuration(args)

    # init Model
    nndial_model = NNDIAL(args, config)

    # train Model
    #nndial_model.train_model()

    # test
    nndial_model.test_model()



class NNDIAL(nn.Module):
    def __init__(self, args, config):
        super(NNDIAL, self).__init__()

        self.args = args

        # load dataset
        self.datareader = DataReader(config.corpusfile, config.dbfile, config.semidictfile, config.ontologyfile,
            config.split, config.lengthen, config.percent,
            config.shuffle, config.trk_enc, config.verbose, args.mode, config.policy,
            config.latent)

        # model
        self.model = Network(args, config, self.datareader)

        #
        self.inf_dimensions = self.datareader.infoseg
        self.req_dimensions = self.datareader.reqseg


    def train_model(self):
        for i in range(self.args.max_epoch):
            logging.info('--------------------Round {0}---------------------'.format(i))
            total_loss = 0
            count = 0
            while True:
                data = self.datareader.read()
                if data == None:
                    logging.info('Round Completed!')
                    break

                '''
                source_0, source_len_1, masked_source_2, masked_source_len_3,\
                target_4, target_len_5, masked_target_6, masked_target_len_7,\
                snapshot_8, change_9, goal_10, inf_trk_label_11, req_trk_label_12,\
                db_degree_13, srcfeat_14, tarfeat_15, finished_16, utt_group_17 = data  # 除了goal和finished以外长度都为对话轮次长
                '''
                ###############tansform#################
                # print self.datareader.vocab
                # for idx,len in enumerate(data[3]):
                #     idx_list = data[2][idx][:len]
                #     sentence = [self.datareader.vocab[id] for id in idx_list]
                #     print sentence
                # for idx,len in enumerate(data[7]):
                #     idx_list = data[6][idx][:len]
                #     sentence = [self.datareader.vocab[id] for id in idx_list]
                #     print sentence
                # print data[2]
                # print data[3]
                # print data[6]
                # print data[7]

                # dialogue_recur
                dialogue_loss, target, predict = self.model.recurr(data)

                if type(dialogue_loss) == int:
                    continue

                total_loss += dialogue_loss.data[0]
                count += 1
                logging.info('count: {0}'.format(count))
            logging.info('avg_loss: {0}'.format(total_loss/(count+0.000000000001)))

            # save model
            torch.save(self.model.state_dict(), './model/save/model')


    def test_model(self):

        #?
        self.verbose = 1

        # evaluator
        bscorer = BLEUScorer()
        parallel_corpus = []
        best_corpus = []

        # load testing data
        testset = self.datareader.iterate(mode='test')

        # statistics for calulating semi performance
        stats = self._statsTable()

        # gate stats
        gstats = np.zeros((4))
        num_sent = 0.0

        # for each dialog
        for cnt in range(len(testset)):
            # initial state
            if self.verbose>0:
                print '='*25 + ' Dialogue '+ str(cnt) +' '+ '='*28
            #print '##############################################################'
            # read one example
            source, source_len, masked_source, masked_source_len,\
            target, target_len, masked_target, masked_target_len,\
            snapshot, change_label, goal, inf_trk_label, req_trk_label,\
            db_degree, srcfeat, tarfeat, finished, utt_group = testset[cnt]

            # initial selection
            selected_venue  = -1
            venue_offered   = None

            # initial belief
            flatten_belief_tm1 = variable_tensor([0]*self.inf_dimensions[-1], 'Float').squeeze(0)
            for i in range(len(self.inf_dimensions)-1):
                flatten_belief_tm1[self.inf_dimensions[i+1]-1] = 1.0

            # for each turn
            reqs = []
            generated_utt_tm1 = ''
            for t in range(len(source)):
                if self.verbose>0:
                    print '-'*28 + ' Turn '+ str(t) +' '+ '-'*28
                # extract source and target sentence for that turn
                source_t        = source[t][:source_len[t]]
                masked_source_t = masked_source[t][:masked_source_len[t]]
                masked_target_t = masked_target[t][:masked_target_len[t]]
                # this turn features
                srcfeat_t   = srcfeat[t]

                # previous target
                masked_target_tm1, target_tm1, starpos_tm1, vtarpos_tm1, offer = \
                    self.datareader.extractSeq(generated_utt_tm1,type='target')

                tarfeat_tm1 = [starpos_tm1,vtarpos_tm1]

                # utterance preparation
                source_utt = ' '.join([self.datareader.vocab[w] for w in source_t])
                masked_source_utt= ' '.join([self.datareader.vocab[w]
                        for w in masked_source_t])
                masked_target_utt= ' '.join([self.datareader.vocab[w]
                        for w in masked_target_t])

                # prepare pytorch variable
                masked_source_t = variable_tensor(masked_source_t, 'Long')
                masked_target_tm1 = variable_tensor(masked_target_tm1+[1], 'Long')

                # read and understand user sentence
                self.model.load_state_dict(torch.load('./model/save/model'))
                masked_intent_t = self.model.read( masked_source_t )
                full_belief_t_np, flatten_belief_t_tensor, belief_t = self.model.track(
                        flatten_belief_tm1, masked_source_t, masked_target_tm1,
                        srcfeat_t, tarfeat_tm1 )

                # search DB
                db_degree_t, query = self._searchDB(flatten_belief_t_tensor.data.numpy())
                # score table
                scoreTable = self._genScoreTable(full_belief_t_np)
                # generation
                db_degree_t_var = variable_tensor(db_degree_t, 'Float')
                generated,sample_t,_ = self.model.talk(
                        masked_intent_t, belief_t, db_degree_t_var,
                        masked_source_t, variable_tensor(masked_target_t, 'Long'), scoreTable)

                # choose venue
                venues = [i for i, e in enumerate(db_degree_t[:-6]) if e != 0 ]
                # keep the current venue
                if selected_venue in venues: pass
                else: # choose the first match as default index
                    if len(venues)!=0:  selected_venue = random.choice(venues)
                    # no matched venues
                    else: selected_venue = None

                # lexicalise generated utterance
                generated_utts = []
                for gen in generated:
                    generated_utt = ' '.join([self.datareader.vocab[g] for g in gen[0]])
                    generated_utts.append(generated_utt)
                gennerated_utt = generated_utts[0]

                # calculate semantic match rate
                twords = [self.datareader.vocab[w] for w in masked_target_t]
                for gen in generated:
                    gwords = [self.datareader.vocab[g] for g in gen[0]]
                    for gw in gwords:
                        if gw.startswith('[VALUE_') or gw.startswith('[SLOT_'):
                            if gw in twords: # match target semi token
                                stats['approp'][0] += 1.0
                            stats['approp'][1] += 1.0
                    #gstats += np.mean( np.array(gen[2][1:]),axis=0 )
                    num_sent += 1

                # update history belief
                flatten_belief_tm1 = flatten_belief_t_tensor[:self.inf_dimensions[-1]]

                # for calculating success: check requestable slots match
                requestables = ['phone','address','postcode','food','area','pricerange']
                for requestable in requestables:
                    if '[VALUE_'+requestable.upper()+']' in gennerated_utt:
                        reqs.append(self.datareader.reqs.index(requestable+'=exist'))
                # check offered venue
                if '[VALUE_NAME]' in generated_utt and selected_venue!=None:
                    venue_offered = self.datareader.db2inf[selected_venue]

                ############################### debugging ############################
                if self.verbose>0:
                    print 'User Input :\t%s'% source_utt
                    print '           :\t%s'% masked_source_utt
                    print
                if 1:
                    #if self.verbose>1:
                    if 1:
                        print 'Belief Tracker :'
                        print '  | %16s%13s%20s|' % ('','Informable','')
                        print '  | %16s\t%5s\t%20s |' % ('Prediction','Prob.','Ground Truth')
                        print '  | %16s\t%5s\t%20s |' % ('------------','-----','------------')
                    for i in range(len(self.inf_dimensions)-1):
                        bn = self.inf_dimensions[i]
                        psem = self.datareader.infovs[np.argmax(np.array(full_belief_t_np[i]))+bn]
                        ysem = self.datareader.infovs[np.argmax(np.array(\
                                inf_trk_label[t][bn:self.inf_dimensions[i+1]+bn]))+bn]
                        prob = full_belief_t_np[i][np.argmax(np.array(full_belief_t_np[i]))]
                        #print '%20s\t%.3f\t%20s' % (psem,prob,ysem)
                        if self.verbose>1:
                            print '  | %16s\t%.3f\t%20s |' % (psem,prob,ysem)

                        # counting stats
                        slt,val = ysem.split('=')
                        if 'none' not in ysem:
                            if psem==ysem: # true positive
                                stats['informable'][slt][0] += 1.0
                            else: # false negative
                                stats['informable'][slt][1] += 1.0
                        else:
                            if psem==ysem: # true negative
                                stats['informable'][slt][2] += 1.0
                            else: # false positive
                                stats['informable'][slt][3] += 1.0

                #if self.trk=='rnn' and self.trkreq==True:
                if 1:
                    if self.verbose>1:
                        print '  | %16s%13s%20s|' % ('','Requestable','')
                        print '  | %16s\t%5s\t%20s |' % ('Prediction','Prob.','Ground Truth')
                        print '  | %16s\t%5s\t%20s |' % ('------------','-----','------------')
                    #infbn = 3 if self.trkinf else 0
                    infbn = 3
                    for i in range(len(self.req_dimensions)-1):
                        bn = self.req_dimensions[i]
                        ysem = self.datareader.reqs[np.argmax(np.array(\
                                req_trk_label[t][bn:self.req_dimensions[i+1]+bn]))+bn]
                        psem = self.datareader.reqs[ \
                            np.argmax(np.array(full_belief_t_np[infbn+i])) +\
                            self.req_dimensions[i] ]
                        prob = np.max(np.array(full_belief_t_np[infbn+i]))
                        if self.verbose>1:
                            print '  | %16s\t%.3f\t%20s |' % (psem,prob,ysem)

                        # counting stats
                        slt,val = ysem.split('=')
                        if slt+'=exist'==ysem:
                            if psem==ysem: # true positive
                                stats['requestable'][slt][0] += 1.0
                            else: # false negative
                                stats['requestable'][slt][1] += 1.0
                        else:
                            if psem==ysem: # true negative
                                stats['requestable'][slt][2] += 1.0
                            else: # false positive
                                stats['requestable'][slt][3] += 1.0

                    # offer change tracker
                    bn = self.req_dimensions[-1]
                    psem = 0 if full_belief_t_np[-1][0]>=0.5 else 1
                    ysem = np.argmax(change_label[t])
                    if ysem==0:
                        if psem==ysem:
                            stats['requestable']['change'][0] += 1.0
                        else:
                            stats['requestable']['change'][1] += 1.0
                    else:
                        if psem==ysem:
                            stats['requestable']['change'][2] += 1.0
                        else:
                            stats['requestable']['change'][3] += 1.0
                    prdtvenue = 'venue=change' if psem==0 else 'venue=not change'
                    truevenue = 'venue=change' if ysem==0 else 'venue=not change'
                    prob      = full_belief_t_np[-1][0] if psem==0 else 1-full_belief_t_np[-1][0]
                    if self.verbose>1:
                        print '  | %16s\t%.3f\t%20s |' % (prdtvenue,prob,truevenue)

                if self.verbose>0:
                    match_number = np.argmax(np.array(db_degree_t[-6:]))
                    match_number = str(match_number) if match_number<5 else '>5'
                    print
                    print 'DB Match     : %s' % match_number
                    print
                    print 'Generated    : %s' % generated_utts[0]
                    for g in generated_utts[1:]:
                        print '             : %s'% g
                    print
                    print 'Ground Truth : %s' % masked_target_utt
                    print
                #raw_input()
                ############################### debugging ############################
                generated_utt_tm1 = masked_target_utt

                parallel_corpus.append([generated_utts,[masked_target_utt]])
                best_corpus.append([[generated_utt],[masked_target_utt]])

            # at the end of the dialog, calculate goal completion rate
            if venue_offered!=None and finished:
                if set(venue_offered).issuperset(set(goal[0].nonzero()[0].tolist())):
                    stats['vmc'] += 1.0
                    if set(reqs).issuperset(set(goal[1].nonzero()[0].tolist())):
                        stats['success'] += 1.0

        # evaluation result
        print 80*'#'
        print 35*'#' + '  Metrics ' + 35*'#'
        print 80*'#'
        print 'Venue Match Rate     : %.1f%%' % (100*stats['vmc']/float(len(testset)))
        print 'Task Success Rate    : %.1f%%' % (100*stats['success']/float(len(testset)))
        #if self.dec!='none':
        if 1:
            print 'BLEU                 : %.4f' % (bscorer.score(best_corpus))
            print 'Semantic Match       : %.1f%%' % (100*stats['approp'][0]/stats['approp'][1])
        print 35*'#' + ' Trackers ' + 35*'#'
        print '---- Informable  '+ 63*'-'
        infslots = ['area','food','pricerange']
        joint = [0.0 for x in range(4)]
        for i in range(len(infslots)):
            s = infslots[i]
            joint = [joint[i]+stats['informable'][s][i] for i in range(len(joint))]
            tp, fn, tn, fp = stats['informable'][s]
            p = tp/(tp+fp)*100
            r = tp/(tp+fn)*100
            ac= (tp+tn)/(tp+tn+fp+fn)*100
            print '%12s :\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t|' %\
                (s, p, r, 2*p*r/(p+r), ac)
        tp, fn, tn, fp = joint
        p = tp/(tp+fp)*100
        r = tp/(tp+fn)*100
        ac= (tp+tn)/(tp+tn+fp+fn)*100
        print 80*'-'
        print '%12s :\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t|' %\
                ('joint', p, r, 2*p*r/(p+r), ac)
        print '---- Requestable '+ 63*'-'
        reqslots = ['area','food','pricerange','address','postcode','phone']#,'change']
        joint = [0.0 for x in range(4)]
        for i in range(len(reqslots)):
            s = reqslots[i]
            joint = [joint[i]+stats['requestable'][s][i] for i in range(len(joint))]
            tp, fn, tn, fp = stats['requestable'][s]
            p = tp/(tp+fp)*100
            r = tp/(tp+fn)*100
            ac= (tp+tn)/(tp+tn+fp+fn)*100
            print '%12s :\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t|' %\
                (s, p, r, 2*p*r/(p+r), ac)
        tp, fn, tn, fp = joint
        p = tp/(tp+fp)*100
        r = tp/(tp+fn)*100
        ac= (tp+tn)/(tp+tn+fp+fn)*100
        print 80*'-'
        print '%12s :\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t|' %\
                ('joint', p, r, 2*p*r/(p+r), ac)
        print 80*'-'
        print '%12s :\t| %7s\t| %7s\t| %7s\t| %7s\t|' %\
                ('Metrics', 'Prec.', 'Recall', 'F-1', 'Acc.')
        print 80*'#'


    def _statsTable(self):
        return {'informable':{
                    'pricerange': [10e-9, 10e-4, 10e-4, 10e-4],
                    'food'      : [10e-9, 10e-4, 10e-4, 10e-4],
                    'area'      : [10e-9, 10e-4, 10e-4, 10e-4]
            },  'requestable':{
                    'pricerange': [10e-9, 10e-4, 10e-4, 10e-4],
                    'area'      : [10e-9, 10e-4, 10e-4, 10e-4],
                    'food'      : [10e-9, 10e-4, 10e-4, 10e-4],
                    'postcode'  : [10e-9, 10e-4, 10e-4, 10e-4],
                    'address'   : [10e-9, 10e-4, 10e-4, 10e-4],
                    'phone'     : [10e-9, 10e-4, 10e-4, 10e-4],
                    'name'      : [10e-9, 10e-4, 10e-4, 10e-4],
                    'change'    : [10e-9, 10e-4, 10e-4, 10e-4]
            },
            'vmc': 10e-7, 'success': 10e-7, 'approp': [10e-7,10e-7]
        }


    def _genScoreTable(self, sem_j):
        scoreTable = {}
        # requestable tracker scoreTable
        #if self.trk=='rnn' and self.trkreq==True:
        if 1:
            #infbn = 3 if self.trkinf else 0
            infbn = 3
            for i in range(len(self.req_dimensions)-1):
                bn = self.req_dimensions[i]
                # prediction for this req tracker
                psem = self.datareader.reqs[ \
                    np.argmax(np.array(sem_j[infbn+i])) +\
                    self.req_dimensions[i] ]
                #print psem
                # slot & value
                s,v = psem.split('=')
                if s=='name': # skip name slot
                    continue
                # assign score, if exist, +reward
                score = -0.05 if v=='none' else 0.2
                # slot value indexing
                vidx = self.datareader.vocab.index('[VALUE_'+s.upper()+']')
                sidx = self.datareader.vocab.index('[SLOT_'+s.upper()+']')
                scoreTable[sidx] = score
                scoreTable[vidx] = score # reward [VALUE_****] if generate
        # informable tracker scoreTable
        #if self.trk=='rnn' and self.trkinf==True:
        if 1:
            for i in range(len(self.inf_dimensions)-1):
                bn = self.inf_dimensions[i]
                # prediction for this inf tracker
                psem = self.datareader.infovs[np.argmax(np.array(sem_j[i]))+bn]
                #print psem
                # slot & value
                s,v = psem.split('=')
                # if none, discourage gen. if exist, encourage gen
                score = -0.5 if (v=='none' or v=='dontcare') else 0.05
                # slot value indexing
                vidx = self.datareader.vocab.index('[VALUE_'+s.upper()+']')
                sidx = self.datareader.vocab.index('[SLOT_'+s.upper()+']')
                if not scoreTable.has_key(sidx) or scoreTable[sidx]<=0.0:
                    scoreTable[sidx] = 0.0 # less encourage for [SLOT_****]
                if not scoreTable.has_key(vidx) or scoreTable[vidx]<=0.0:
                    scoreTable[vidx] = score # encourage [SLOT_****]

        return scoreTable


    def _searchDB(self,b):

        query = []  # 仅informable的非dontcare非none的值
        q = []  # 仅informable的可为dontcare可为none的值
        db_logic = []
        # formulate query for search
        #if self.trkinf==True:
        if 1:
            for i in range(len(self.inf_dimensions)-1):
                b_i = b[self.inf_dimensions[i]:self.inf_dimensions[i+1]] # 相同的key下不同的value，对它们进行binary truth?
                idx = np.argmax(np.array(b_i)) + self.inf_dimensions[i]  # 找出可能性最大的序号
                # ignore dont care case
                s2v = self.datareader.infovs[idx]  # {'area=centre','area=east',...} 选出了其中的某一个
                if '=dontcare' not in s2v and '=none' not in s2v:  # 如果挑出的非空
                    query.append(idx)
                q.append(idx)
            # search through db by query  # 这就是论文中的x_t
            for entry in self.datareader.db2inf:
                if set(entry).issuperset(set(query)):  # entry是否包含query
                    db_logic.append(1)
                else:
                    db_logic.append(0)
            # form db count features  # degree的分级制
            dbcount = sum(db_logic)
            if dbcount<=3:
                dummy = [0 for x in range(6)]
                dummy[dbcount] = 1
                db_logic.extend(dummy)
            elif dbcount<=5:
                db_logic.extend([0,0,0,0,1,0])
            else:
                db_logic.extend([0,0,0,0,0,1])
        else:
            db_logic = [0,0,0,0,0,1]
        return db_logic, q



















if __name__ == '__main__':
    main()