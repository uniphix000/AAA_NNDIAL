#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
from ConfigParser import SafeConfigParser
from ast import literal_eval
from utils.tools import findSubList
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from utils.nlp import normalize
import sys
import re
import operator
import simplejson as json
import math
import random
from pprint import pprint
import itertools
import numpy as np
from copy import deepcopy


digitpat = re.compile('\d+')

class DataReader(object):
    '''
    载入语料数据并切分，载入词汇表，构建数据库，
    '''
    inputvocab = []
    outputvocab= []
    ngrams = {}
    idx2ngs = []

    def __init__(self,
            corpusfile, dbfile, semifile, s2vfile,
            split, lengthen, percent, shuffle,
            trkenc, verbose, mode, att=False, latent_size=1):

        self.att = True if att=='attention' else False
        self.dl  = latent_size
        self.data  = {'train':[],'valid':[],'test':[]} # container for data
        self.mode = 'train'     # mode  for accessing data
        self.index = 0          # index for accessing data

        # data manipulators
        self.split  = DataSplit(split)  # split helper 初始化
        self.trkenc = trkenc
        self.lengthen = lengthen
        self.shuffle= shuffle

        # NLTK stopword module
        self.stopwords = set(stopwords.words('english'))
        for w in ['!',',','.','?','-s','-ly','</s>','s']:
            self.stopwords.add(w)

        # loading files
        self.db       = self.loadjson(dbfile)
        self.s2v      = self.loadjson(s2vfile)  # slot to value {'requestable':,'informable':（预设）,'other':...}
        self.semidict = self.loadjson(semifile)  # 一对多词
        self.dialog   = self.loadjson(corpusfile)
        # producing slot value templates and db represetation
        self.prepareSlotValues()
        self.structureDB()  # 将kb变成idx的list

        # load dialog
        self.loadVocab()
        if mode!='sds':
            self.loadDialog()  # 主要是完成了value-->key的替换
            self.loadSemantics()  # 类似于一个parser，匹配每句话中出现的info/req_slots，并匹配分degree

        # goal
        self.parseGoal()

        # split dataset
        if mode!='sds':
            self._setupData(percent)

        if verbose : self._printStats()

    def loadDialog(self):
        """
        index dialog
        :return:
        """

        # index words and make it suitable for NN input
        self.sourceutts = []  # source端的utterance
        self.targetutts = []  # target端的utterance
        self.masked_sourceutts = []
        self.masked_targetutts = []
        self.sourcecutoffs = []
        self.targetcutoffs = []
        self.masked_sourcecutoffs = []
        self.masked_targetcutoffs = []

        # delexicalised positions
        self.delsrcpos = []
        self.deltarpos = []

        # finished dialogs
        self.finished = []

        # venue specific - offered/changing
        self.offers = []
        self.changes= []

        # snapshot vectors
        self.snapshot_vecs = []

        # for each dialogue
        dcount = 0.0
        tcount = 0.0

        # for VAE initialisation
        self.sentGroupIndex = []
        groupidx = 0

        for d in self.dialog:

            # consider finished flag
            if d.has_key('finished'):
                self.finished.append(d['finished'])
            else:
                self.finished.append(True)

            # print loading msgs
            dcount += 1.0
            print '\tloading dialog from file ... finishing %.2f%%\r' %\
                (100.0*float(dcount)/float(len(self.dialog))),
            sys.stdout.flush()

            # container for each turn
            sourceutt   = []
            targetutt   = []
            m_sourceutt = []
            m_targetutt = []
            utt_group   = []

            srcpos = []
            tarpos = []

            maxtar = -1
            maxsrc = -1
            maxmtar= -1
            maxmsrc= -1
            maxfeat= -1

            offers  = []
            changes = []
            prevoffer = []
            offered = False

            snapshot_vecs = []

            # for each turn in a dialogue
            for t in range(len(d['dial'])):
                tcount += 1
                turn = d['dial'][t]
                # extract system side sentence feature
                sent = turn['sys']['sent']
                mtar, tar, spos, vpos, venues \
                    = self.extractSeq(sent,type='target')  #

                # store sentence group
                utt_group.append(self.sentGroup[groupidx])
                groupidx += 1

                # changing offer label
                if len(venues)!=0 and venues[0] not in prevoffer: # not matching
                    if prevoffer==[]: # new offer
                        change = [0,1]
                    else: # changing offer
                        change = [1,0]
                    prevoffer = venues
                else:
                    change = [0,1]
                changes.append(change)

                # offer label
                if offered or len(venues)!=0: # offer has happened
                    offer = [1,0]
                    offered = True
                else:
                    offer = [0,1]
                offers.append(offer)

                # delexicalised
                if len(mtar)>maxtar:
                    maxtar = len(mtar)
                m_targetutt.append(mtar)

                # extract snapshot vectors
                snapshot_vec = [[0.0 for x in range(len(self.snapshots))]]
                # add offer and change to snapshot vector
                if offer==[1,0] : snapshot_vec[0][
                        self.snapshots.index('OFFERED') ] = 1.0
                if change==[1,0]: snapshot_vec[0][
                        self.snapshots.index('CHANGED') ] = 1.0

                # attentive snapshot
                for w in mtar[::-1]:
                    ssvec = deepcopy(snapshot_vec[0])
                    if self.vocab[w] in self.snapshots:
                        ssvec[ self.snapshots.index(
                            self.vocab[w]) ] = 1.0
                    snapshot_vec.insert(0,ssvec)
                # decide changing snapshot or not
                if self.att==True:
                    snapshot_vecs.append(snapshot_vec[:-1])
                else:
                    snapshot_vecs.append([deepcopy(snapshot_vec[0])
                        for x in snapshot_vec[:-1]])

                # handling positional features
                for f in spos:
                    if len(f)>maxfeat:
                        maxfeat = len(f)
                for f in vpos:
                    if len(f)>maxfeat:
                        maxfeat = len(f)
                tarpos.append([spos,vpos])

                # non delexicalised
                if len(tar)>maxmtar:
                    maxmtar = len(tar)
                targetutt.append(tar)

                # usr responses
                sent = turn['usr']['transcript']
                msrc, src, spos, vpos, _ = self.extractSeq(sent,type='source')

                # delexicalised
                if len(msrc)>maxsrc:
                    maxsrc = len(msrc)
                m_sourceutt.append(msrc)

                # handling positional features
                for f in spos:
                    if len(f)>maxfeat:
                        maxfeat = len(f)
                for f in vpos:
                    if len(f)>maxfeat:
                        maxfeat = len(f)
                srcpos.append([spos,vpos])

                # non delexicalised
                if len(src)>maxmsrc:
                    maxmsrc = len(src)
                sourceutt.append(src)

            # sentence group
            self.sentGroupIndex.append(utt_group)

            # offers
            self.changes.append(changes)
            self.offers.append(offers)

            # padding for snapshots
            for i in range(len(m_targetutt)):
                snapshot_vecs[i].extend(
                        [snapshot_vecs[i][0]]*\
                        (maxtar-len(m_targetutt[i])) )

            # padding unk tok
            m_sourcecutoff = []
            m_targetcutoff = []
            for i in range(len(m_targetutt)):
                m_targetcutoff.append(len(m_targetutt[i]))
                m_targetutt[i].extend(
                        [self.vocab.index('<unk>')]*\
                        (maxtar-len(m_targetutt[i])) )
            for i in range(len(m_sourceutt)):
                m_sourcecutoff.append(len(m_sourceutt[i]))
                m_sourceutt[i].extend(
                        [self.vocab.index('<unk>')]*\
                        (maxsrc-len(m_sourceutt[i])) )

            # non delexicalised version
            sourcecutoff = []
            targetcutoff = []
            for i in range(len(targetutt)):
                targetcutoff.append(len(targetutt[i]))
                targetutt[i].extend(
                        [self.vocab.index('<unk>')]*\
                        (maxmtar-len(targetutt[i])) )

            for i in range(len(sourceutt)):
                sourcecutoff.append(len(sourceutt[i]))
                sourceutt[i].extend(
                        [self.vocab.index('<unk>')]*\
                        (maxmsrc-len(sourceutt[i])) )

            # padding positional features
            for i in range(len(tarpos)):
                for j in range(len(tarpos[i])):
                    for k in range(len(tarpos[i][j])):
                        tarpos[i][j][k].extend(\
                            [-1]*(maxfeat-len(tarpos[i][j][k])))
            for i in range(len(srcpos)):
                for j in range(len(srcpos[i])):
                    for k in range(len(srcpos[i][j])):
                        srcpos[i][j][k].extend(\
                            [-1]*(maxfeat-len(srcpos[i][j][k])))

            # entire dialogue matrix
            self.sourceutts.append(sourceutt)
            self.targetutts.append(targetutt)
            self.sourcecutoffs.append(sourcecutoff)
            self.targetcutoffs.append(targetcutoff)

            self.masked_sourceutts.append(m_sourceutt)
            self.masked_targetutts.append(m_targetutt)
            self.masked_sourcecutoffs.append(m_sourcecutoff)
            self.masked_targetcutoffs.append(m_targetcutoff)

            self.snapshot_vecs.append(snapshot_vecs)
            # positional information
            self.delsrcpos.append(srcpos)
            self.deltarpos.append(tarpos)

    def loadSemantics(self):

        # sematic labels
        self.info_semis = []
        self.req_semis  = []
        self.db_logics = []

        sumvec      = np.array([0 for x in range(self.infoseg[-1])])
        # for each dialogue
        dcount = 0.0
        for dx in range(len(self.dialog)):
            d = self.dialog[dx]
            # print loading msgs
            dcount += 1.0
            print '\tloading semi labels from file ... finishing %.2f%%\r' %\
                (100.0*float(dcount)/float(len(self.dialog))),
            sys.stdout.flush()

            # container for each turn
            info_semi   = []
            req_semi    = []
            semi_idxs   = []
            db_logic    = []

            # for each turn in a dialogue
            for t in range(len(d['dial'])):
                turn = d['dial'][t]

                # read informable semi
                semi = sorted(['pricerange=none','food=none','area=none']) \
                        if len(info_semi)==0 else deepcopy(info_semi[-1])
                for da in turn['usr']['slu']:
                    for s2v in da['slots']:
                        # skip invalid slots
                        if len(s2v)!=2 or s2v[0]=='slot':
                            continue
                        s,v = s2v
                        # need to replace the slot with system request
                        if v=='dontcare' and s=='this':
                            sdas = d['dial'][t-1]['sys']['DA']
                            for sda in sdas:
                                if sda['act']=='request':
                                    s = sda['slots'][0][-1]
                                    break
                        toreplace = None
                        for sem in semi:
                            if s in sem:
                                toreplace = sem
                                break
                        if s=='this':
                            continue
                        else:
                            if toreplace:
                                semi.remove(toreplace)
                            semi.append(s+'='+v)

                # if goal changes not venue changes
                if self.changes[dx][t]==[1,0]:
                    if info_semi[-1]!=sorted(semi):
                        self.changes[dx][t] = [0,1]

                info_semi.append(sorted(semi))

                # indexing semi and DB
                vec = [0 for x in range(self.infoseg[-1])]
                constraints = []
                for sem in semi:
                    if 'name=' in sem:
                        continue
                    vec[self.infovs.index(sem)] = 1
                    if self.infovs.index(sem) not in self.dontcare:
                        constraints.append(self.infovs.index(sem))
                semi_idxs.append(vec)
                sumvec += np.array(vec)
                infosemi = semi

                # check db match
                match = [len(filter(lambda x: x in constraints, sub)) \
                        for sub in self.db2inf]
                venue_logic = [int(x>=len(constraints)) for x in match]
                vcount = 0
                for midx in range(len(venue_logic)):
                    if venue_logic[midx]==1:
                        vcount += len(self.idx2db[midx])
                if vcount<=3:
                    dummy = [0 for x in range(6)]
                    dummy[vcount] = 1
                    venue_logic.extend(dummy)
                elif vcount<=5:
                    venue_logic.extend([0,0,0,0,1,0])
                else:
                    venue_logic.extend([0,0,0,0,0,1])
                db_logic.append(venue_logic)

                # read requestable semi
                semi =  sorted(['food','pricerange','area'])+\
                        sorted(['phone','address','postcode'])
                for da in turn['usr']['slu']:
                    for s2v in da['slots']:
                        if s2v[0]=='slot':
                            for i in range(len(semi)):
                                if s2v[1]==semi[i]:
                                    semi[i] += '=exist'
                for i in range(len(semi)):
                    if '=exist' not in semi[i]:
                        semi[i] += '=none'
                vec = [0 for x in range(self.reqseg[-1])]
                for sem in semi:
                    vec[self.reqs.index(sem)] = 1
                req_semi.append(vec)

            self.info_semis.append(semi_idxs)  # 一个稀疏向量[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            self.req_semis.append( req_semi )
            self.db_logics.append(db_logic)  # 为每句话打的匹配分

    def extractSeq(self,sent,type='source',normalise=False,index=True):
        '''

        :param sent:
        :param type:
        :param normalise:
        :param index:
        :return:
        '''
        # setup vocab
        if type=='source':  vocab = self.vocab
        elif type=='target':vocab = self.vocab

        # standardise sentences
        if normalise:
            sent = normalize(sent)

        # preporcessing
        words = sent.split()
        if type=='source':
            if len(words)==0: words = ['<unk>']
        elif type=='target':
            words = ['</s>'] + words + ['</s>']

        # indexing, non-delexicalised
        if index:
            idx  = map(lambda w: vocab.index(w) if w in vocab else 0, words)
        else:
            idx = words

        # delexicalise all
        sent = self.delexicalise(' '.join(words),mode='all')  # value-->key的替换
        sent = re.sub(digitpat,'[VALUE_COUNT]',sent)  # 正则表达式，将数字替换成[]表示
        words= sent.split()

        # formulate delex positions
        allvs = self.infovs+self.reqs
        sltpos = [[] for x in allvs]
        valpos = [[] for x in allvs]
        names = []
        for i in range(len(words)):
            if '::' not in words[i]:
                continue
            # handling offer changing
            if words[i].startswith('[VALUE_NAME]'):
                name = words[i].replace('[VALUE_NAME]::','')
                names.append(name)
            # remove pos identifier
            tok, ID = words[i].split("::")
            words[i] = tok
            # record position
            mytok,sov = tok[1:-1].lower().split('_')
            ID = ID.replace('-',' ')
            mylist = sltpos if mytok=='slot' else valpos
            for j in range(len(allvs)):
                s,v = allvs[j].split('=')
                comp = s if mytok=='slot' else v
                if comp==ID:
                    if mytok=='slot':
                        sltpos[j].append(i)
                    else:
                        valpos[j].append(i)

        # indexing, delexicalised
        if index:
            midx = map(lambda w: vocab.index(w) if w in vocab else 0, words)
        else:
            midx = words

        return midx, idx, sltpos, valpos, names

    def delexicalise(self,utt,mode='all'):
        '''
        value-->key
        </s> there are several restaurant -s in the south part of town that serve expensive food . do you have a cuisine preference ? </s>
        </s> there are several restaurant -s in the [VALUE_AREA]::south [SLOT_AREA]::area that serve [VALUE_PRICERANGE]::expensive [SLOT_FOOD]::food . do you have a [SLOT_FOOD]::food preference ? </s>
        :param utt:
        :param mode:
        :return:
        '''
        inftoks =   ['[VALUE_'+s.upper()+']' for s in self.s2v['informable'].keys()] + \
                    ['[SLOT_' +s.upper()+']' for s in self.s2v['informable'].keys()] + \
                    ['[VALUE_DONTCARE]','[VALUE_NAME]'] +\
                    ['[SLOT_' +s.upper()+']' for s in self.s2v['requestable'].keys()]
        reqtoks =   ['[VALUE_'+s.upper()+']' for s in self.s2v['requestable'].keys()]
        for i in range(len(self.values)):
            # informable mode, preserving location information
            if mode=='informable' and self.slots[i] in inftoks:
                tok = self.slots[i]+'::'+(self.supervalues[i]).replace(' ','-')
                utt = (' '+utt+' ').replace(' '+self.values[i]+' ',' '+tok+' ')
                utt = utt[1:-1]
            # requestable mode
            elif mode=='requestable' and self.slots[i] in reqtoks:
                utt = (' '+utt+' ').replace(' '+self.values[i]+' ',' '+self.slots[i]+' ')
                utt = utt[1:-1]
            elif mode=='all':
                tok = self.slots[i]+'::'+(self.supervalues[i]).replace(' ','-') \
                        if self.slots[i] in inftoks else self.slots[i]
                utt = (' '+utt+' ').replace(' '+self.values[i]+' ',' '+tok+' ')
                utt = utt[1:-1]
        utt = re.sub(digitpat,'[VALUE_COUNT]',utt)
        return utt

    def delexicaliseOne(self,utt,toks,repl):
        for tok in toks:
            utt = (' '+utt+' ').replace(' '+tok+' ',' '+repl+' ')
            utt = utt[1:-1]
        return utt

    def prepareSlotValues(self):
        """
        向s2v中加入kb里已存在键的requestable值
        :return:
        """
        print '\tprepare slot value templates ...'
        # put db requestable values into s2v
        for e in self.db:
            for s,v in e.iteritems():
                if self.s2v['requestable'].has_key(s):
                    self.s2v['requestable'][s].append(v.lower())
                if self.s2v['other'].has_key(s):
                    self.s2v['other'][s].append(v.lower())
        # sort values  # 将值排序
        for s,vs in self.s2v['informable'].iteritems():
            self.s2v['informable'][s] = sorted(list(set(vs)))
        for s,vs in self.s2v['requestable'].iteritems():
            self.s2v['requestable'][s] = sorted(list(set(vs)))
        for s,vs in self.s2v['other'].iteritems():
            self.s2v['other'][s] = sorted(list(set(vs)))

        # make a 1-on-1 mapping for delexicalisation  # 从semidict中提取
        self.supervalues = []  # 最简单的最抽象的value food
        self.values = []  # type of food , cuisine...
        self.slots  = []  # SLOT_FOOD

        for s,vs in self.s2v['informable'].iteritems():
            # adding slot delexicalisation
            self.supervalues.extend([s for x in self.semidict[s]])
            self.values.extend([normalize(x) for x in self.semidict[s]])
            self.slots.extend(['[SLOT_'+s.upper()+']' for x in self.semidict[s]])
            # adding value delexicalisation
            for v in vs:
                self.supervalues.extend([v for x in self.semidict[v]])
                self.values.extend([normalize(x) for x in self.semidict[v]])
                self.slots.extend(['[VALUE_'+s.upper()+']' for x in self.semidict[v]])
        for s,vs in self.s2v['requestable'].items()+self.s2v['other'].items():
            # adding value delexicalisation
            self.values.extend([normalize(v) for v in vs])
            self.supervalues.extend([v for v in vs])
            self.slots.extend(['[VALUE_'+s.upper()+']' for v in vs])
            # adding slot delexicalisation
            self.supervalues.extend([s for x in self.semidict[s]])
            self.values.extend([normalize(x) for x in self.semidict[s]])
            self.slots.extend(['[SLOT_'+s.upper()+']' for x in self.semidict[s]])
        # incorporate dontcare values
        self.values.extend([normalize(v) for v in self.semidict['any']])
        self.supervalues.extend(['dontcare' for v in self.semidict['any']])
        self.slots.extend(['[VALUE_DONTCARE]' for v in self.semidict['any']])

        # sorting according to length
        self.values, self.supervalues, self.slots = zip(*sorted(\
                zip(self.values,self.supervalues,self.slots),\
                key=lambda x: len(x[0]),reverse=True))  # 将三个list按照长度同步的进行排序

        # for generating semantic labels
        self.infovs = []  # {'area=centre','area=east',...}
        self.infoseg = [0]  # s2v informable中有3个键,指示长度:[0,7,100,105]
        self.reqs = []  # 3个键与exit和none的组合
        self.reqseg = [0]  #
        self.dontcare = []  # 如果用户不指名信息，则从dontcare中随机
        for s in sorted(self.s2v['informable'].keys()):
            self.infovs.extend([s+'='+v for v in self.s2v['informable'][s]])
            self.infovs.append(s+'=dontcare')
            self.infovs.append(s+'=none')
            self.infoseg.append(len(self.infovs))
            # dont care values
            self.dontcare.append(len(self.infovs)-1)
            self.dontcare.append(len(self.infovs)-2)  # fixme
        for s in sorted(self.s2v['informable'].keys()):
            self.reqs.extend([s+'=exist',s+'=none'])
            self.reqseg.append(len(self.reqs))
        for s in sorted(self.s2v['requestable'].keys()):
            self.reqs.extend([s+'=exist',s+'=none'])
            self.reqseg.append(len(self.reqs))

        # for ngram indexing
        self.ngs2v = []  # [('area', ['centre', 'east', 'north', 'south', 'west', 'any', 'none']), ...]
        for s in sorted(self.s2v['informable'].keys()):
            self.ngs2v.append( (s, self.s2v['informable'][s] + ['any','none']) )
        for s in sorted(self.s2v['informable'].keys()):
            self.ngs2v.append( (s,['exist','none']) )
        for s in sorted(self.s2v['requestable'].keys()):
            self.ngs2v.append( (s,['exist','none']) )

    def loadjson(self,filename):
        with open(filename) as data_file:
            for i in range(5):
                data_file.readline()
            data = json.load(data_file)
        return data

    def _printStats(self):
        print '\n==============='
        print 'Data statistics'
        print '==============='
        print 'Train    : %d' % len(self.data['train'] )
        print 'Valid    : %d' % len(self.data['valid'] )
        print 'Test     : %d' % len(self.data['test']  )
        print '==============='
        print 'Voc      : %d' % len(self.vocab)
        if self.trkenc=='ng':
            print 'biGram:  : %d' % len(self.bigrams)
            print 'triGram: : %d' % len(self.trigrams)
        if self.trkenc=='ng':
            print 'All Ngram: %d' % len(self.ngrams)
        print '==============='
        print 'Venue    : %d' % len(self.db2inf)
        print '==============='

    def _setupData(self,percent):

        # zip corpus
        if self.trkenc=='ng':
            trksrc = self.ngram_source
            trktar = self.ngram_target
        else:
            trksrc = self.delsrcpos
            trktar = self.deltarpos

        corpus = [  self.sourceutts,        self.sourcecutoffs,
                    self.masked_sourceutts, self.masked_sourcecutoffs,
                    self.targetutts,        self.targetcutoffs,
                    self.masked_targetutts, self.masked_targetcutoffs,
                    self.snapshot_vecs,
                    self.changes,   self.goals,
                    self.info_semis,        self.req_semis,
                    np.array(self.db_logics),
                    trksrc,                 trktar,
                    self.finished,          self.sentGroupIndex]
        corpus = zip(*corpus)

        # split out train+valid
        train_valid = self.split.train_valid(corpus)

        # cut dataset according to percentage
        percent = float(percent)/float(100)
        train_valid = train_valid[:int(len(train_valid)*percent)]

        # split into train/valid/test
        self.data['train'] = self.split.train(train_valid)
        self.data['valid'] = self.split.valid(train_valid)
        self.data['test']  = self.split.test(corpus)

    def read(self,mode='train'):
        ## default implementation for read() function
        if self.mode!=mode:
            self.mode = mode
            index = 0

        # end of data , reset index & return None
        if self.index>=len(self.data[mode]):
            data = None
            self.index = 0

            if mode!='test': # train or valid, do shuffling
                if self.shuffle=='static': # just shuffle current set
                    random.shuffle(self.data[mode])
                elif self.shuffle=='dynamic':
                    # shuffle train + valid together
                    train_valid = self.data['train']+self.data['valid']
                    random.shuffle(train_valid)
                    self.data['train'] = self.split.train(train_valid)
                    self.data['valid'] = self.split.valid(train_valid)
            return data

        # 1 dialog at a time
        data = deepcopy(list(self.data[mode][self.index]))
        lengthen_idx = 1
        while   lengthen_idx<self.lengthen and \
                self.index+lengthen_idx<len(self.data[mode]):
            #lengthen the data by combining two data points
            nextdata = deepcopy(list(self.data[mode][self.index+lengthen_idx]))
            data = self.lengthenData(data,nextdata,mode)
            lengthen_idx += 1
        self.index += lengthen_idx
        return data

    def lengthenData(self,data,addon,mode):
        #for t in range(len(data[10])):
        #    print np.nonzero(np.array(data[10][t]))
        for i in range(len(data)): # for every data matrix
            if isinstance(data[i],list):
                idx = [0,2,4,6]
                if i in idx: # sequences, need padding
                    maxleng = max(len(data[i][0]),len(addon[i][0]))
                    for t in range(len(data[i])): # for each turn
                        data[i][t].extend([0]*(maxleng-len(data[i][t])))
                    for t in range(len(addon[i])): # for each turn
                        addon[i][t].extend([0]*(maxleng-len(addon[i][t])))
                idx = [8]
                if i in idx: # snapshot vectors
                    maxleng = max(len(data[i][0]),len(addon[i][0]))
                    for t in range(len(data[i])): # turn
                        data[i][t].extend([[-1 for cnt in \
                            range(len(data[i][t][0]))]]*(maxleng-len(data[i][t])))
                    for t in range(len(addon[i])):# turn
                        addon[i][t].extend([[-1 for cnt in \
                            range(len(addon[i][t][0]))]]*(maxleng-len(addon[i][t])))
                idx = [14,15]
                if i in idx: # ngram/position features
                    maxleng = max(len(data[i][0][0][0]),len(addon[i][0][0][0]))
                    for t in range(len(data[i])): # turn
                        for x in range(len(data[i][t])): # slot or value
                            for sv in range(len(data[i][t][x])): # each value
                                data[i][t][x][sv].extend([-1]*\
                                    (maxleng-len(data[i][t][x][sv])))
                    for t in range(len(addon[i])):# turn
                        for x in range(len(addon[i][t])):# slot or value
                            for sv in range(len(addon[i][t][x])):# each value
                                addon[i][t][x][sv].extend([-1]*\
                                    (maxleng-len(addon[i][t][x][sv])))
                data[i] = addon[i] + data[i]
        # propagte tracker labels
        for t in range(len(data[11])):
            for s in range(len(self.infoseg[:-1])):
                if t!=0 and data[11][t][self.infoseg[s]:self.infoseg[s+1]][-1]==1:
                    data[11][t][self.infoseg[s]:self.infoseg[s+1]] = \
                        data[11][t-1][self.infoseg[s]:self.infoseg[s+1]]
            #print np.nonzero(np.array(data[10][t]))
        #print np.array(data[0]).shape
        #raw_input()
        """
        for i in range(len(data)):
            try: data[i] = np.array(data[i],dtype='float32')
            except: pass
        """
        return data

    def iterate(self,mode='test',proc=True):
        # default implementation for iterate() function
        return self.data[mode]

    def structureDB(self):

        # all informable values
        print '\tformatting DB ...'

        # represent each db entry with informable values
        self.db2inf = []  # [[0, 46, 100], [1, 44, 100], [0, 42, 101], [3, 24, 101],... 将db中的各个条目变成idx表示
        self.db2idx  = []  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14,  将bd中的条目可重复的标上序号，注意：是informable的可重复
        self.idx2db = []  # [[[0, 46, 100], [0, 46, 100], [0, 46, 100]], [[1, 44, 100]], [[0, 42, 101], [0, 42, 101], [0, 42, 101], [0, 42, 101], [0, 42, 101], [0, 42, 101]], [[3, 24, 101] 序号对应的db中的idx表示
        self.idx2ent= {}  # {0: [{'name': 'pizza hut city centre', 'area': 'centre',...  序号对应原始的条目
        for i in range(len(self.db)):
            e = self.db[i]
            e2inf = []
            for s,v in e.iteritems():
                if s in self.s2v['informable']:
                    e2inf.append( self.infovs.index(s+'='+v) )  # 返回idx
            e2inf = sorted(e2inf)  # ['area=..','name=..',...]=>[0,46,100]

            # if not repeat, create new entry
            if e2inf not in self.db2inf:
                self.db2inf.append(e2inf)
                self.db2idx.append(len(self.db2inf)-1)
                self.idx2db.append([e2inf])
                self.idx2ent[self.db2inf.index(e2inf)] = [e]
            else: # if repeat, indexing back
                self.db2idx.append(self.db2inf.index(e2inf))
                self.idx2db[self.db2inf.index(e2inf)].append(e2inf)
                self.idx2ent[self.db2inf.index(e2inf)].append(e)

        # create hash for finding db index by name
        self.n2db = {}  # 根据name来索引db中的条目
        for i in range(len(self.db)):
            self.n2db[self.db[i]['name'].lower()] = self.db2idx[i]

    def loadVocab(self):

        # iterate through dialog and make vocab
        self.inputvocab = ['[VALUE_DONTCARE]','[VALUE_COUNT]']
        self.outputvocab= ['[VALUE_DONTCARE]','[VALUE_COUNT]']
        self.vocab = []

        # init inputvocab with informable values
        for s,vs in self.s2v['informable'].iteritems():
            for v in vs:
                if v=='none': continue
                self.inputvocab.extend(v.split())
            self.inputvocab.extend( ['[SLOT_'+s.upper()+']','[VALUE_'+s.upper()+']'])
            self.outputvocab.extend(['[SLOT_'+s.upper()+']','[VALUE_'+s.upper()+']'])

        # init inputvocab with requstable values
        for s,vs in self.s2v['requestable'].iteritems():
            for v in vs:
                if v=='none': continue
                self.inputvocab.extend(v.split())
            self.inputvocab.extend( ['[SLOT_'+s.upper()+']','[VALUE_'+s.upper()+']'])
            self.outputvocab.extend(['[SLOT_'+s.upper()+']','[VALUE_'+s.upper()+']'])

        # add every word in semidict into vocab
        for s in self.semidict.keys():
            for v in self.semidict[s]:
                self.inputvocab.extend(v.split())

        # for grouping sentences
        sentKeys = {}
        self.sentGroup= []

        # lemmatizer
        lmtzr = WordNetLemmatizer()

        # form lexican
        ivocab = []
        ovocab = []
        for i in range(len(self.dialog)):

            print '\tsetting up vocab, finishing ... %.2f%%\r' %\
                (100.0*float(i)/float(len(self.dialog))),
            sys.stdout.flush()

            # parsing dialog
            for j in range(len(self.dialog[i]['dial'])):
                # text normalisation
                self.dialog[i]['dial'][j]['sys']['sent'] = normalize(
                        self.dialog[i]['dial'][j]['sys']['sent'])
                self.dialog[i]['dial'][j]['usr']['transcript'] = normalize(
                        self.dialog[i]['dial'][j]['usr']['transcript'])
                # this turn
                turn = self.dialog[i]['dial'][j]

                # system side
                words,_,_,_,_ = self.extractSeq(turn['sys']['sent'],\
                    type='target',index=False)

                ovocab.extend(words)

                # sentence group key
                key = tuple(set(sorted(
                    [lmtzr.lemmatize(w) for w in words if w not in self.stopwords])))
                if key in sentKeys:
                    sentKeys[key][1] += 1
                    self.sentGroup.append( sentKeys[key][0] )
                else:
                    sentKeys[key] = [len(sentKeys),1]
                    self.sentGroup.append( sentKeys[key][0] )

                # user side
                words = self.delexicalise(turn['usr']['transcript']).split()
                mwords,words,_,_,_ = self.extractSeq(turn['usr']['transcript'],\
                    type='source',index=False)
                ivocab.extend(mwords)
                ivocab.extend(words)
                """
                for hyp in t['usr']['asr']:
                    words = self.delexicalise(normalize(hyp['asr-hyp'])).split()
                    ivocab.extend(words)
                """

        # re-assigning sentence group w.r.t their frequency
        mapping = {}
        idx = 0
        cnt = 0
        for key,val in sorted(sentKeys.iteritems(),key=lambda x:x[1][1],reverse=True):
            mapping[val[0]] = idx
            #print idx, val[1], key
            if idx<self.dl-1: cnt+=val[1]
            idx += 1
        #raw_input()
        print '\tsemi-supervised action examples: %2.2f%%' % \
                (float(cnt)/float(len(self.sentGroup))*100)
        for i in range(len(self.sentGroup)):
            self.sentGroup[i] = min(mapping[self.sentGroup[i]],self.dl-1)

        # set threshold for input vocab
        counts = dict()
        for w in ivocab:
            counts[w] = counts.get(w, 0) + 1
        self.inputvocab = ['<unk>','</s>','<slot>','<value>'] + \
                sorted(list(set(self.inputvocab+\
                [w for w,c in sorted(counts.iteritems(),key=operator.itemgetter(1)) if c>1])))

        # set threshold for output vocab
        counts = dict()
        for w in ovocab:
            counts[w] = counts.get(w, 0) + 1
        self.outputvocab = ['<unk>','</s>'] + \
                sorted(list(set(self.outputvocab+['thank','you','goodbye']+\
                [w for w,c in sorted(counts.iteritems(),key=operator.itemgetter(1))])))

        # the whole vocab
        self.vocab = ['<unk>','</s>','<slot>','<value>'] + \
                list(set(self.inputvocab[4:]).union(self.outputvocab[2:]))

        # create snapshot dimension
        self.snapshots = ['OFFERED','CHANGED']
        for w in self.outputvocab:
            if w.startswith('[VALUE'):
                self.snapshots.append(w)
        self.snapshots = sorted(self.snapshots)

    def parseGoal(self):
        # parse goal into dict format
        self.goals = []
        # for computing corpus success
        requestables = ['phone','address','postcode','food','area','pricerange']
        vmc, success = 0., 0.
        # for each dialog
        for i in range(len(self.dialog)):
            d = self.dialog[i]
            goal = [np.zeros(self.infoseg[-1]),
                    np.zeros(self.reqseg[-1])]
            for s2v in d['goal']['constraints']:
                s,v = s2v
                s2v = s+'='+v
                if v!='dontcare' and v!='none':
                    #goal['inf'].append( self.infovs.index(s2v) )
                    goal[0][self.infovs.index(s2v)] = 1
            for s in d['goal']['request-slots']:
                if s=='pricerange' or s=='area' or s=='food':
                    continue
                #goal['req'].append(self.reqs.index(s+'=exist'))
                goal[1][self.reqs.index(s+'=exist')] = 1
            self.goals.append(goal)
            # compute corpus success
            m_targetutt = self.masked_targetutts[i]
            m_targetutt_len = self.masked_targetcutoffs[i]
            # for computing success
            offered = False
            requests= []
            # iterate each turn
            for t in range(len(m_targetutt)):
                sent_t = [self.vocab[w] for w in
                        m_targetutt[t][:m_targetutt_len[t]]][1:-1]
                if '[VALUE_NAME]' in sent_t: offered=True
                for requestable in requestables:
                    if '[VALUE_'+requestable.upper()+']' in sent_t:
                        requests.append(self.reqs.index(requestable+'=exist'))
            # compute success
            if offered:
                vmc += 1.
                if set(requests).issuperset(set(goal[1].nonzero()[0].tolist())):
                    success += 1.
        print '\tCorpus VMC       : %2.2f%%' % (vmc/float(len(self.dialog))*100)
        print '\tCorpus Success   : %2.2f%%' % (success/float(len(self.dialog))*100)


class DataSplit(object):  # 用于划分数据
    # data split helper , for split dataset into train/valid/test
    def __init__(self,split):
        self.split = split
        self.sum = sum(split)
    def train_valid(self,data):
        # split the dataset into train+valid
        e = int(len(data)*float(sum(self.split[:2]))/float(self.sum))
        return data[:e]
    def train(self,train_valid):
        # split training from train+valid
        e = int(len(train_valid)*\
                float(self.split[0])/float((sum(self.split[:2]))))
        return train_valid[:e]
    def valid(self,train_valid):
        # split validation from train+valid
        s = len(self.train(train_valid))
        return train_valid[s:]
    def test(self,data):
        # split the dataset into testing
        s = len(self.train_valid(data))
        return data[s:]


class Configuration(object):
    def __init__(self, args):
        '''

        :return:
        '''
        parser = SafeConfigParser()
        parser.read(args.config)

        self.corpusfile = parser.get('file', 'corpus')
        self.dbfile = parser.get('file', 'db')
        self.semidictfile = parser.get('file', 'semidict')
        self.ontologyfile = parser.get('file', 'ontology')
        self.split = literal_eval(parser.get('data','split'))
        self.lengthen = parser.getint('data', 'lengthen')
        self.percent = parser.get('data', 'percent')
        self.shuffle = parser.get('data', 'shuffle')
        self.trk_enc = parser.get('trk','trkenc')
        self.verbose = parser.getint('gen','verbose')
        self.mode = args.mode
        self.policy = parser.get('ply','policy')
        self.latent = parser.getint('ply','latent') if self.policy=='latent' else 0
        self.topk           = parser.getint('gen','topk')
        self.beamwidth      = parser.getint('gen','beamwidth')
        self.repeat_penalty = parser.get('gen','repeat_penalty')
        self.token_reward   = parser.getboolean('gen','token_reward')
        self.alpha          = parser.getfloat('gen','alpha')

