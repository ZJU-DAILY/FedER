import argparse
import logging
import os
import random
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
import faiss
from sklearn import preprocessing
from data import MyDataset
from tensorboardX import SummaryWriter
from os.path import abspath, dirname, join, exists
from apex import amp
from knowledge import *
import matplotlib.pyplot as plt
from utils import *
import Levenshtein

class NCESoftmaxLoss(nn.Module):

    def __init__(self, device):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([batch_size]).to(self.device).long()
        loss = self.criterion(x, label)
        return loss


class BertEncoder(nn.Module):
    def __init__(self, args, device='cuda', alpha_aug=0.8):
        super().__init__()
        self.lm = args.lm
        self.bert = AutoModel.from_pretrained('./huggingface/bert-base-uncased')
        self.device = device
        self.args = args
        self.alpha_aug = alpha_aug

        # linear layer
        # hidden_size = self.bert.config.hidden_size  #768
        # self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        # self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.criterion = NCESoftmaxLoss(self.device)
        # self.aggregation_layer = torch.nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        # self.wv = nn.Parameter(torch.zeros(size=(1, 256)))
        # nn.init.xavier_uniform_(self.wv)

    def contrastive_loss(self, pos_1, pos_2, neg_value, neg_aug_value=None):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        if neg_aug_value is not None:
            # l_aug_neg = torch.bmm(pos_1.view(bsz, 1, -1), neg_aug_value.view(bsz, -1, 1))
            neg_aug_value = neg_aug_value.view(bsz, -1, 768)  #(batch_size, 8, 768)

            l_aug_neg = torch.bmm(pos_1.view(bsz, 1, -1), torch.transpose(neg_aug_value, 1, 2))  #(batch_size, 768, 8)
            l_aug_neg = l_aug_neg.view(bsz, 1)
            # l_aug_neg = l_aug_neg.view(bsz, 8)
            logits = torch.cat((l_pos, l_aug_neg, l_neg), dim=1)
        else:
            logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        return self.criterion(logits / self.args.t)


    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def forward(self, x1, x2=None ):
        x1 = x1.to(self.device)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            hidden_states = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            hidden_states = self.bert(x1)[0]  #batch_size, 256, 768
        # hidden_states = self.linear1(hidden_states)
        # hidden_states = self.linear1(hidden_states)
        # max pooling
        # out = torch.max(hidden_states, dim=1)[0]
        # mean pooling
        m = nn.AdaptiveAvgPool2d((1, 768))
        out = m(hidden_states)
        out = torch.squeeze(out)
        # print(out.size())
        # print(out.size())
        # out = self.linear(enc)
        return out



def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class Trainer(object):
    def __init__(self, args, training=True, seed=37):
        # # Set the random seed manually for reproducibility.
        self.seed = seed
        fix_seed(seed)
        self.args = args
        self.device = torch.device(self.args.device)
        # generate train data for A
        myset1 = MyDataset(path=args.path1,max_len=256,lm=args.lm,da=args.da,neg_da=args.neg_da,dual_da=args.dual_da,key_position=args.key_position,dk=args.dk)
        padder = myset1.pad
        self.set1_size = myset1.__len__()
        self.set1_id2t = myset1.id2t
        self.loader1 = Data.DataLoader(dataset=myset1,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       collate_fn = padder)


        del myset1
        # generate test data for A
        myset1 = MyDataset(path=args.path1,
                           max_len=256,
                           lm=args.lm,
                           da=None,
                           neg_da=None)
        padder = myset1.pad
        self.eval_loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
            collate_fn=padder
        )


        #generate train data for B
        myset2 =  MyDataset(path=args.path2,
                           max_len=256,
                            lm=args.lm,
                            da=args.da,
                            neg_da=args.neg_da,
                            dual_da=args.dual_da,
                            key_position = args.key_position,
                            dk=args.dk
                           )
        padder = myset2.pad
        self.set2_size = myset2.__len__()
        self.set2_id2t = myset2.id2t
        self.loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=True,
            collate_fn=padder
        )

        del myset2
        # generate test data for B
        myset2 = MyDataset(path=args.path2,
                           max_len=256,
                           lm=args.lm,
                           da=None,
                           neg_da=None
                           )

        padder = myset2.pad
        self.eval_loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
            collate_fn=padder
        )

        del myset2
        self.model = None
        self.iteration = 0

        def match_loader(path):
            match = []
            p = open(path, 'r')
            i=0
            for line in p:
                id_1, id_2 = line.strip().split(' ')
                match.append((int(id_1),int(id_2)))
                i+=1
            return match
        self.match = match_loader(self.args.match_path)
        self.neg_queue1 = None
        self.neg_queue1_id = []
        self.neg_queue2 = None
        self.neg_queue2_id = []

        if training:
            self.writer = SummaryWriter(log_dir=args.logdir)
            # self.writer = SummaryWriter(
            #     log_dir=join(PROJ_DIR, 'log', self.args.model, self.args.model_language, self.args.time),
            #     comment=self.args.time)
            device = self.device
            self.model = BertEncoder(self.args).to(self.device)  # encoder q
            self._model = BertEncoder(self.args).to(self.device)  # encoder k
            self._model.update(self.model)  # moto update
            # self.blocker = HashBlocker(self.args, self.args.hc_length).to(self.device)
            self.iteration = 0
            self.lr = self.args.lr
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

            if self.args.fp16:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O2')



    def train(self, start=0):
        begin = time.time()
        fix_seed(self.seed)
        PROJ_DIR = abspath(dirname(__file__))
        task = self.args.task
        task = task.replace('/', '_')
        if not os.path.exists(join(PROJ_DIR, 'log_aug')):
            os.mkdir(join(PROJ_DIR, 'log_aug'))
        if not os.path.exists(join(PROJ_DIR, 'log_aug', task)):
            os.mkdir(join(PROJ_DIR, 'log_aug', task))



        filename = os.path.join(PROJ_DIR, 'log_aug', task+'/lm={}_da={}_negda={}_dualda={}_dk={}_bsize={}_qsize={}.txt'.format(
            self.args.lm,
            self.args.da,
            self.args.neg_da,
            self.args.dual_da,
            self.args.dk,
            str(self.args.batch_size),
            str(self.args.queue_length),
            ))

        all_data_batches = []
        aug_data_batches = []
        pos_aug_data_batches = []
        neg_aug_data_batches = []

        for batch_id, batch in tqdm(enumerate(self.loader1)):  # data from table1
            if len(batch) == 2:
                tuple_tokens, tuple_id = batch
                all_data_batches.append([1, tuple_tokens, tuple_id])
            elif len(batch) == 3:
                tuple_tokens, tuple_aug_tokens, tuple_id = batch # T(batch_size, 256) T(batch_size, neg_num, 256) T(batch_size)
                all_data_batches.append([1, tuple_tokens, tuple_id])
                aug_data_batches.append([1, tuple_aug_tokens, tuple_id])
            else:
                tuple_tokens, tuple_pos_aug_tokens, tuple_neg_aug_tokens, tuple_id = batch
                all_data_batches.append([1, tuple_tokens, tuple_id])
                pos_aug_data_batches.append([1, tuple_pos_aug_tokens, tuple_id])
                neg_aug_data_batches.append([1, tuple_neg_aug_tokens, tuple_id])




        for batch_id, batch in tqdm(enumerate(self.loader2)):  # data from table2
            if len(batch) == 2:
                tuple_tokens, tuple_id = batch
                all_data_batches.append([2, tuple_tokens, tuple_id])
            elif len(batch) == 3:
                tuple_tokens, tuple_aug_tokens, tuple_id = batch
                aug_data_batches.append([2, tuple_aug_tokens, tuple_id])
                all_data_batches.append([2, tuple_tokens, tuple_id])
            else:
                tuple_tokens, tuple_pos_aug_tokens, tuple_neg_aug_tokens, tuple_id = batch
                all_data_batches.append([2, tuple_tokens, tuple_id])
                pos_aug_data_batches.append([2, tuple_pos_aug_tokens, tuple_id])
                neg_aug_data_batches.append([2, tuple_neg_aug_tokens, tuple_id])

        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(all_data_batches)
        if self.args.da is not None or self.args.neg_da is not None:
            random.seed(randnum)
            random.shuffle(aug_data_batches)
        if self.args.dual_da is not None:
            random.seed(randnum)
            random.shuffle(pos_aug_data_batches)
            random.seed(randnum)
            random.shuffle(neg_aug_data_batches)

        for epoch in range(start, self.args.epoch):
            adjust_learning_rate(self.optimizer, epoch, self.lr)
            for batch_id, (party_id, tuple_embed, tuple_id) in enumerate(all_data_batches):
                pos_batch = None
                if party_id == 1:
                    with torch.no_grad():
                        if self.neg_queue1 == None:
                            self.neg_queue1 = tuple_embed.unsqueeze(0)
                            self.neg_queue1_id .append(batch_id)
                        else:
                            self.neg_queue1 = torch.cat((self.neg_queue1, tuple_embed.unsqueeze(0)), dim=0)
                            self.neg_queue1_id.append(batch_id)

                    # id_data = tuple_id.squeeze()
                    if self.neg_queue1.shape[0] == self.args.queue_length + 1:
                        pos_batch = self.neg_queue1[0]
                        self.neg_queue1 = self.neg_queue1[1:]
                        neg_queue = self.neg_queue1
                        if self.args.da is not None or self.args.neg_da is not None:
                            id = self.neg_queue1_id[0]
                            self.neg_queue1_id = self.neg_queue1_id[1:]
                            _, batch_aug, _ = aug_data_batches[id]  # batch_aug (batch_size, neg_num, 256)
                        if self.args.dual_da is not None:
                            id = self.neg_queue1_id[0]
                            self.neg_queue1_id = self.neg_queue1_id[1:]
                            _, pos_batch_aug, _ = pos_aug_data_batches[id]  # batch_aug (batch_size, neg_num, 256)
                            _, neg_batch_aug, _ = neg_aug_data_batches[id]
                    else:
                        continue

                else:
                    with torch.no_grad():
                        if self.neg_queue2 == None:
                            self.neg_queue2 = tuple_embed.unsqueeze(0) # 1* batch_size * len
                            self.neg_queue2_id.append(batch_id)
                        else:
                            self.neg_queue2 = torch.cat((self.neg_queue2, tuple_embed.unsqueeze(0)), dim=0)
                            self.neg_queue2_id.append(batch_id)

                    if self.neg_queue2.shape[0] == self.args.queue_length + 1:
                        pos_batch = self.neg_queue2[0]
                        self.neg_queue2 = self.neg_queue2[1:]
                        neg_queue = self.neg_queue2
                        if self.args.da is not None or self.args.neg_da is not None:
                            id = self.neg_queue2_id[0]
                            self.neg_queue2_id = self.neg_queue2_id[1:]
                            _, batch_aug, _ = aug_data_batches[id]
                        if self.args.dual_da is not None:
                            id = self.neg_queue2_id[0]
                            self.neg_queue2_id = self.neg_queue2_id[1:]
                            _, pos_batch_aug, _ = pos_aug_data_batches[id]  # batch_aug (batch_size, neg_num, 256)
                            _, neg_batch_aug, _ = neg_aug_data_batches[id]
                    else:
                        continue

                self.optimizer.zero_grad()
                pos_1 = self.model(pos_batch.squeeze(0))  # model q

                with torch.no_grad():
                    self._model.eval()
                    if self.args.da is not None: # da MixDA
                        pos_2 = self._model(batch_aug.squeeze(0))
                    #   pos_2 = self._model(pos_batch.squeeze(0), batch_aug.squeeze(0))  # model k
                    # else:
                    #     pos_2 = self._model(pos_batch.squeeze(0))  # with pos self
                        # pos_2 = F.normalize(pos_2, p=2, dim=1)

                    elif self.args.neg_da is not None:
                        batch_aug = batch_aug.view(-1, 256)
                        neg_aug = self._model(batch_aug.squeeze(0))


                    elif self.args.dual_da is not None:
                        pos_2 = self._model(pos_batch_aug.squeeze(0))
                        neg_batch_aug = neg_batch_aug.view(-1, 256)
                        neg_aug = self._model(neg_batch_aug.squeeze(0))
                    else:
                        pos_2 = self._model(pos_batch.squeeze(0))


                    neg_shape = neg_queue.shape
                    neg_queue = neg_queue.reshape(neg_shape[0] * neg_shape[1], neg_shape[2]) #batch_size, que.size, token_maxlen
                    neg_value = self._model(neg_queue)
                    # neg_value = F.normalize(neg_value, p=2, dim=1)
                    del neg_queue


                # contrastive
                if self.args.neg_da is not None or self.args.dual_da is not None:
                    contrastive_loss = self.model.contrastive_loss(pos_1, pos_2, neg_value, neg_aug)
                else:
                    contrastive_loss = self.model.contrastive_loss(pos_1, pos_2, neg_value)

                loss = contrastive_loss
                del pos_1
                del pos_2
                del neg_value

                self.iteration += 1

                if self.args.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=True)

                self.optimizer.step()
                if batch_id == len(all_data_batches) - 1:
                    # if batch_id % 200 == 0 or batch_id == len(all_data_batches) - 1:
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id,
                                                                loss.detach().cpu().data / self.args.batch_size))
                    with open(filename, 'a+') as pf:
                        pf.write('epoch: {} batch: {} loss: {}\n'.format(epoch, batch_id,
                                                                loss.detach().cpu().data / self.args.batch_size))



                    hit1_valid_12, hit10_valid_12, hit1_valid_21, hit10_valid_21 = self.evaluate(str(epoch) + ": batch " + str(batch_id), filename)


                #     if hit1_valid > best_hit1_valid:
                #         best_hit1_valid = hit1_valid
                #         best_hit1_valid_hit10 = hit10_valid
                #         best_hit1_valid_epoch = epoch
                #         record_epoch = epoch
                #         record_batch_id = batch_id
                #         record_hit1 = hit1_test
                #         record_hit10 = hit10_test
                #     if hit10_valid > best_hit10_valid:
                #         best_hit10_valid = hit10_valid
                #         best_hit10_valid_hit1 = hit1_valid
                #         best_hit10_valid_epoch = epoch
                #         if hit1_valid == best_hit1_valid:
                #             record_epoch = epoch
                #             record_batch_id = batch_id
                #             record_hit1 = hit1_test
                #             record_hit10 = hit10_test
                #
                #     if hit1_test > best_hit1_test:
                #         best_hit1_test = hit1_test
                #         best_hit1_test_hit10 = hit10_test
                #         best_hit1_test_epoch = epoch
                #     if hit10_test > best_hit10_test:
                #         best_hit10_test = hit10_test
                #         best_hit10_test_hit1 = hit1_test
                #         best_hit10_test_epoch = epoch
                #     log.write('Test Hit@1(10)    = {}({}) at epoch {} batch {}'.format(hit1_test, hit10_test, epoch,
                #                                                                           batch_id))
                #     log.write('Best Valid Hit@1  = {}({}) at epoch {}'.format(best_hit1_valid, best_hit1_valid_hit10,
                #                                                                  best_hit1_valid_epoch))
                #     log.write(
                #         'Best Valid Hit@10 = {}({}) at epoch {}'.format(best_hit10_valid, best_hit10_valid_hit1,
                #                                                         best_hit10_valid_epoch))
                #     log.write('Test @ Best Valid = {}({}) at epoch {} batch {}'.format(record_hit1, record_hit10,
                #                                                                           record_epoch,
                #                                                                           record_batch_id))
                #
                #     log.write('Best Test  Hit@1  = {}({}) at epoch {}'.format(best_hit1_test, best_hit1_test_hit10,
                #                                                                  best_hit1_test_epoch))
                #     log.write('Best Test  Hit@10 = {}({}) at epoch {}'.format(best_hit10_test, best_hit10_test_hit1,
                #                                                                  best_hit10_test_epoch))
                #     log.write("====================================")
                #
                #     print('Test Hit@1(10)    = {}({}) at epoch {} batch {}'.format(hit1_test, hit10_test, epoch,
                #                                                                    batch_id))
                #     print('Best Valid Hit@1  = {}({}) at epoch {}'.format(best_hit1_valid, best_hit1_valid_hit10,
                #                                                           best_hit1_valid_epoch))
                #     print('Best Valid Hit@10 = {}({}) at epoch {}'.format(best_hit10_valid, best_hit10_valid_hit1,
                #                                                           best_hit10_valid_epoch))
                #     print('Test @ Best Valid = {}({}) at epoch {} batch {}'.format(record_hit1, record_hit10,
                #                                                                    record_epoch, record_batch_id))
                #
                #     print('Best Test  Hit@1  = {}({}) at epoch {}'.format(best_hit1_test, best_hit1_test_hit10,
                #                                                           best_hit1_test_epoch))
                #     print('Best Test  Hit@10 = {}({}) at epoch {}'.format(best_hit10_test, best_hit10_test_hit1,
                #                                                           best_hit10_test_epoch))
                #     print("====================================")
                # update
                self._model.update(self.model)
        end = time.time()
        print("============================================================\n")



    def evaluate(self, step, filename):
        with open (filename, 'a+') as pf:
            pf.write("Evaluate at epoch {}...".format(step))
        print("Evaluate at epoch {}...".format(step))

        ids_1, ids_2, vector_1, vector_2, hash_code_1, hash_code_2 = list(), list(), list(), list(), list(), list()
        inverse_ids_1, inverse_ids_2 = dict(), dict()
        neg_vector_1, neg_vector_2 = list(), list()
        with torch.no_grad():
            self.model.eval()
            for sample_id_1, (tuple_token_1, tuple_id_1) in tqdm(enumerate(self.eval_loader1)):
                tuple_vector_1 = self.model(tuple_token_1)
                # tuple_hash_1 = self.blocker(tuple_vector_1, istrain=False).squeeze().detach().cpu().numpy().tolist()
                tuple_vector_1 = tuple_vector_1.squeeze().detach().cpu().numpy()
                vector_1.append(tuple_vector_1)
                tuple_id_1 = tuple_id_1.squeeze().tolist()
                if isinstance(tuple_id_1, int):
                    tuple_id_1 = [tuple_id_1]
                ids_1.extend(tuple_id_1)

            for sample_id_2, (tuple_token_2, tuple_id_2) in tqdm(enumerate(self.eval_loader2)):
                tuple_vector_2 = self.model(tuple_token_2)
                tuple_vector_2 = tuple_vector_2.squeeze().detach().cpu().numpy()
                vector_2.append(tuple_vector_2)
                tuple_id_2 = tuple_id_2.squeeze().tolist()
                if isinstance(tuple_id_2, int):
                    tuple_id_2 = [tuple_id_2]
                ids_2.extend(tuple_id_2)

                # hash_code_2.extend(tuple_hash_2)

        for idx, _id in enumerate(ids_2):
            inverse_ids_2[_id] = idx  # _id的实体的index

        for idx, _id in enumerate(ids_1):
            inverse_ids_1[_id] = idx  # entity id to index

        def cal_hit(v1, v2, writer, match, set1_id2t, set2_id2t, step, neg_v1=None, neg_v2=None): #link dic
            PseudoMatch = list()
            Similarity = list()
            TrueMatch = match

            sim_score = torch.tensor(v1.dot(v2.T))
            sim_score_A = torch.tensor(v1.dot(v1.T))
            sim_score_B = torch.tensor(v2.dot(v2.T))
            if neg_v1 is not None:
                aug_sim_score_A = np.sum(v1*neg_v1, axis=1)
                print(aug_sim_score_A)
            if neg_v2 is not None:
                aug_sim_score_B = np.sum(v2*neg_v2, axis=1)

            distA, topkA = torch.topk(sim_score, k=2, dim=1)
            distB, topkB = torch.topk(sim_score, k=2, dim=0)
            self_distA, self_topkA = torch.topk(sim_score_A, k=10, dim=1)
            self_distB, self_topkB = torch.topk(sim_score_B, k=10, dim=1)


            topkB = topkB.t()
            distB = distB.t()
            lenA = topkA.shape[0]
            threshold = 0.03

            for e1_index in range(lenA):
                e2_index = topkA[e1_index][0].item()
                if e1_index == topkB[e2_index][0].item():
                    PseudoMatch.append((ids_1[e1_index], ids_2[e2_index]))
                    # filter with similarity constraint
                    # e1_id = ids_1[e1_index]
                    # e2_id = ids_2[e2_index]
                    # if key1[e1_id] == "" or key2[e2_id] == "" or Levenshtein.distance(key1[e1_id], key2[e2_id]) < 4:
                    #     PseudoMatch.append((ids_1[e1_index], ids_2[e2_index]))
                    # if sim_score[e1_index][e2_index] >= self_distA[e1_index][5].item() and sim_score[e1_index][e2_index] >= self_distB[e2_index][5]:
                    #     PseudoMatch.append((ids_1[e1_index], ids_2[e2_index]))
                    # filter with neg_sample's similarity constraint
                    # if neg_v1 is not None:
                    #     if sim_score[e1_index][e2_index] >= aug_sim_score_A[e1_index] and sim_score[e1_index][e2_index] >= aug_sim_score_B[e2_index]:
                    #         PseudoMatch.append((ids_1[e1_index], ids_2[e2_index]))

                    # filter with threshold constraint
                    # e2_ = topkA[e1][1]
                    # e1_ = topkB[e2][1]
                    # score1 = (sim_score[e1][e2] - sim_score[e1][e2_]).item()
                    # score2 = (sim_score[e1][e2] - sim_score[e1_][e2]).item()
                    # if score1 >= threshold and score2 >= threshold:
                    #     PseudoMatch.append((ids_1[e1], ids_2[e2]))


            # TrueMatch = match
            # PseudoMatch = list()
            # Similarity = list()
            # v1_ = v1
            # v2_ = v2
            # index = faiss.IndexFlatIP(v2_.shape[1])
            # index.add(np.ascontiguousarray(v2_))
            # D, I = index.search(np.ascontiguousarray(v1_), 10)
            #
            # top1_index_12 = I[:, 0]
            # top1_similarity_12 = D[:, 0]
            # top1_id_12 = np.array(ids_2)[top1_index_12]
            # top1_12 = dict()
            # for i in range(len(ids_1)):
            #     top1_12[ids_1[i]] = top1_id_12[i]
            #
            # v2_ = v2
            # v1_ = v1
            # index = faiss.IndexFlatIP(v1_.shape[1])
            # index.add(np.ascontiguousarray(v1_))
            # D, I = index.search(np.ascontiguousarray(v2_), 10)
            # top1_index_21 = I[:, 0]
            # top1_id_21 = np.array(ids_1)[top1_index_21]
            # top1_21 = dict()
            # for i in range(len(ids_2)):
            #     top1_21[ids_2[i]] = top1_id_21[i]
            #
            # for i, id1 in enumerate(top1_12):
            #     if top1_21[top1_12[id1]] == id1:
            #         PseudoMatch.append((id1, top1_12[id1]))
            #         Similarity.append(top1_similarity_12[i])

            tp = 0
            tp_sim = []
            fp_sim = []
            fn = 0
            fn_sim = []
            # for i, pair in enumerate(PseudoMatch):
            #     if pair in TrueMatch:
            #         tp += 1
            #         tp_sim.append(Similarity[i])
            #     else:
            #         fp_sim.append((pair, Similarity[i]))
            match_dic = {}  # dict A->B
            invers_match_dic = {}  # dict B->A
            PseudoMatch_dic = {}   # dict A->B
            invers_PseudoMatch_dic = {}  #dict B->A
            len_pm = len(PseudoMatch)
            for pair in TrueMatch: #list
                if pair[0] not in match_dic:
                    match_dic[pair[0]] = []  # one left may be matched to multi-right entity
                match_dic[pair[0]].append(pair[1])

            for pair in TrueMatch:
                if pair[1] not in invers_match_dic:
                    invers_match_dic[pair[1]] = []  # one left may be matched to multi-right entity
                invers_match_dic[pair[1]].append(pair[0])

            for pair in PseudoMatch:
                PseudoMatch_dic[pair[0]] = [pair[1]]

            for pair in PseudoMatch:
                invers_PseudoMatch_dic[pair[1]] = [pair[0]]

            wrong_match = 0
            extra_match = 0

            tp_sim = []
            wrong_match_sim = []
            extra_match_sim = []

            PseudoMatch_dic_copy = PseudoMatch_dic.copy()
            for e1 in PseudoMatch_dic:
                e2 = PseudoMatch_dic[e1][0]
                if e1 in match_dic:
                    if e2 in match_dic[e1]:
                        tp += 1
                        tp_sim.append(sim_score[inverse_ids_1[e1]][inverse_ids_2[e2]])
                        for e2_ in match_dic[e1]:
                            if e2_ != e2 and e2_ not in invers_PseudoMatch_dic:
                                tp += 1
                                len_pm += 1
                                invers_PseudoMatch_dic[e2_] = [e1]
                                PseudoMatch_dic_copy[e1].append(e2_)
                        for e1_ in invers_match_dic[e2]:
                            if e1_ != e1 and e1_ not in PseudoMatch_dic_copy:
                                tp += 1
                                len_pm += 1
                                PseudoMatch_dic_copy[e1_] = e2
                                invers_PseudoMatch_dic[e2].append(e1_)
                    else:
                        wrong_match += 1
                        # wrong_match_sim.append(sim_score[inverse_ids_1[e1]][inverse_ids_2[e2]])
                        # with open(filename, 'a+') as pf:
                        #     pf.write("{} matches {} but a wrong match. S={}\n".format(str(e1), str(e2),
                        #         str(sim_score[inverse_ids_1[e1]][inverse_ids_2[e2]])))
                        #     pf.write("e1 {}\n".format(set1_id2t[e1]))
                        #     pf.write("e2 {}\n".format(set2_id2t[e2]))
                        #     pf.write("e* {}\n".format(set2_id2t[match_dic[e1][0]]))
                        # print("{} matches {} but a wrong match.".format(str(e1), str(e2)))
                        # print(sim_score[inverse_ids_1[e1]][inverse_ids_2[e2]])
                        # print("e1 {}".format(set1_id2t[e1]))
                        # print("e2 {}".format(set2_id2t[e2]))
                        # print("e* {}".format(set2_id2t[match_dic[e1][0]]))
                else:
                    extra_match += 1
                    # extra_match_sim.append(sim_score[inverse_ids_1[e1]][inverse_ids_2[e2]])
                    # with open(filename, 'a+') as pf:
                    #     pf.write("{} matches {} but {} in table A do not have matcher. S={}\n".format(str(e1),str(e2),str(e1),
                    #                                                   str(sim_score[inverse_ids_1[e1]][inverse_ids_2[e2]])))
                    #     pf.write("e1 {}\n".format(set1_id2t[e1]))
                    #     pf.write("e2 {}\n".format(set2_id2t[e2]))
                    # print("{} matches {} but {} in table A do not have matcher.".format(str(e1),str(e2),str(e1)))
                    # print("e1 {}".format(set1_id2t[e1]))
                    # print("e2 {}".format(set2_id2t[e2]))

            lost = 0
            for e1 in match_dic:
                if e1 not in PseudoMatch_dic_copy:
                    for e2 in match_dic[e1]:
                        lost += 1
                        # log.write("matcher lost, S={}".format(str(sim_score[inverse_ids_1[e1]][inverse_ids_2[e2]])))
                        # log.write("e1 {}".format(set1_id2t[e1]))
                        # log.write("e2 {}".format(set2_id2t[e2]))
                        # log.write("e1's top1 e2_: S={}".format(str(distA[inverse_ids_1[e1]][0].item())))
                        # log.write("e2_ {}".format(set2_id2t[ids_2[topkA[inverse_ids_1[e1]][0].item()]]))
                        # log.write("e2's top1 e1_: S={}".format(str(distB[inverse_ids_2[e2]][0].item())))
                        # log.write("e1_ {}".format(set1_id2t[ids_1[topkB[inverse_ids_2[e2]][0].item()]]))


            # for i, pair in enumerate(PseudoMatch):
            #     if pair in TrueMatch:
            #         tp += 1
            #         tp_sim.append(sim_score[inverse_ids_1[pair[0]]][inverse_ids_2[pair[1]]])
            #     else:
            #         fp_sim.append(sim_score[inverse_ids_1[pair[0]]][inverse_ids_2[pair[1]]])
            #
            # false_match = 0
            #
            # for i, pair in enumerate(TrueMatch):
            #     if pair not in PseudoMatch:
            #         print("false negative", pair[0], pair[1])
            #         print("similarity", sim_score[inverse_ids_1[pair[0]]][inverse_ids_2[pair[1]]])
            #         for p_pair in PseudoMatch:
            #             if p_pair[0] == pair[0]:
            #                false_match += 1
            #                print("pseudomatch: ", p_pair[0], p_pair[1] )
            #                print("similarity", sim_score[inverse_ids_1[p_pair[0]]][inverse_ids_2[p_pair[1]]])
            #             if p_pair[1] == pair[1]:
            #                false_match+=1
            #                print("pseudomatch: ", p_pair[0], p_pair[1])
            #                print("similarity", sim_score[inverse_ids_1[p_pair[0]]][inverse_ids_2[p_pair[1]]])


            # tp = np.array([int((pair in TrueMatch)) for pair in PseudoMatch]).sum()
            print("TrueMatch.Sie: ", len(TrueMatch))
            print("PseudoMatch.Sie: ", len(PseudoMatch))
            print("Ex_PseudoMatch.Sie: ", len_pm)
            print("True_Pos: ", tp)
            print("TP_Rate: {}".format(round(tp/len_pm, 3)))
            print("Lost: ", lost)
            print("Recall: {}".format(round(tp /len(TrueMatch), 3)))
            print("F1: {}".format(round((2*tp/len(TrueMatch)*tp/len_pm)/(tp/len_pm+tp/len(TrueMatch)),3)))
            print("Wrong_Match: {}".format(wrong_match))
            print("Extra_Match: {}".format(extra_match))
            with open (filename, 'a+') as pf:
                pf.write("TrueMatch.Sie: {}\n".format(len(TrueMatch)))
                pf.write("PseudoMatch.Sie: {}\n".format(len(PseudoMatch)))
                pf.write("Ex_PseudoMatch.Sie: {}\n".format(len_pm))
                pf.write("True_Pos: {}\n".format(tp))
                pf.write("TP_Rate: {}\n".format(round(tp/len_pm, 3)))
                pf.write("Lost: {}\n".format(lost))
                pf.write("Recall: {}\n".format(round(tp /len(TrueMatch), 3)))
                pf.write("F1: {}\n".format(round((2*tp/len(TrueMatch)*tp/len_pm)/(tp/len_pm+tp/len(TrueMatch)),3)))
                pf.write("Wrong_Match: {}\n".format(wrong_match))
                pf.write("Extra_Match: {}\n".format(extra_match))

            # tp_sim = np.array(tp_sim)
            # wrong_match_sim = np.array(wrong_match_sim)
            # extra_match_sim = np.array(extra_match_sim)
            # plt.figure(figsize=(10, 10))
            # plt.scatter(tp_sim, tp_sim, s=200, c='r')
            # plt.scatter(wrong_match_sim, wrong_match_sim, s= 100, c='g')
            # plt.scatter(extra_match_sim, extra_match_sim, s=30, c='b')
            # PROJ_DIR = abspath(dirname(__file__))
            # task = self.args.task
            # task = task.replace('/', '_')
            # dirs = os.path.join(PROJ_DIR, 'log', task + '/fig/')
            # if not os.path.exists(dirs):
            #     os.makedirs(dirs)
            # path = os.path.join(PROJ_DIR, 'log', task + '/fig/similarity_epoch_{}'.format(str(step)))
            # plt.savefig(path)


            # print("=============================================================")
            # print("FP_similarity: ")
            # for it in fp_sim:
            #     print(it[0], it[1])
            #
            # print("=============================================================")
            # print("TP_similarity: ")
            # for it in tp_sim:
            #     print(it)

            return tp, tp, tp, tp
            # return round(hit1_12, 3),round(hit10_12, 3), round(hit1_21, 3),round(hit10_21, 3)


        print('========Validation========')
        # print("-------------------------  ",len(self.match))
        v1 = np.vstack(vector_1).astype(np.float32)
        v2 = np.vstack(vector_2).astype(np.float32)
        # neg_v1 = np.vstack(neg_vector_1).astype(np.float32)
        # neg_v2 = np.vstack(neg_vector_2).astype(np.float32)
        # v1 = np.array(vector_1, dtype='float32')
        # v2 = np.array(vector_2, dtype='float32')
        vector_1 = preprocessing.normalize(v1)
        vector_2 = preprocessing.normalize(v2)
        # neg_vector_1 = preprocessing.normalize(neg_v1)
        # neg_vector_2 = preprocessing.normalize(neg_v2)
        # hash_1 = np.array(hash_code_1, dtype= 'int')
        # hash_2 = np.array(hash_code_2, dtype= 'int')


        # buckets_1 = build_buckets(hash_1, ids_1)
        # buckets_2 = build_buckets(hash_2, ids_2)
        match_dic = {}
        # DB = 0
        # B = 0
        for pair in self.match:
            if pair[0] not in match_dic:
                match_dic[pair[0]] = []  # one left may be matched to multi-right entity
            match_dic[pair[0]].append(pair[1])

        # for key in buckets_1:
        #     if key in buckets_2:
        #         B += len(buckets_1[key]) * len(buckets_2[key])
        #         for id_left in buckets_1[key]:
        #             if id_left in match_dic:
        #                 for i, id_right in enumerate(match_dic[id_left]):
        #                     if id_right in buckets_2[key]:
        #                         DB += 1
        #                         match_dic[id_left][i] = -1

        # E = self.set1_size * self.set2_size
        # DE = len(self.match)
        # RR = 1.0 - B/E
        # PC = DB/DE
        # print('--------Blocking--------')
        # print("RR: ", RR)
        # print("PC: ", PC)

        hit1_12, hit10_12, hit1_21, hit10_21 = cal_hit(vector_1, vector_2, self.writer, self.match, self.set1_id2t,
                                                       self.set2_id2t, step)

        # cal_hit2(vector_1, vector_2, self.match)
        return hit1_12, hit10_12, hit1_21, hit10_21

