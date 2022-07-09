import torch
import torch.utils.data as Data
from transformers import AutoTokenizer
from augment import Augmenter
from knowledge import *


class MyDataset(Data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 lm='bert-base-uncased',
                 da=None,
                 neg_da=None,
                 dual_da=None,
                 key_position=None,
                 dk=None
                 ):
        self.lm = lm
        self.tokenizer = AutoTokenizer.from_pretrained('./huggingface/bert-base-uncased')
        # self.tokenizer = AutoTokenizer.from_pretrained(self.lm)
        self.t = []
        self.t_id = []
        self.max_len = max_len
        self.domain = {}
        self.id2t = {}


        lines = open(path, 'r')
        for line in lines:
            t, t_id = line.strip().split('\t')
            self.t.append(t)
            self.t_id.append(int(t_id))
            self.id2t[int(t_id)] = t
        self.da = da
        self.neg_da = neg_da
        self.dual_da = dual_da
        self.dk = dk
        self.KD = {}
        self.key_position = None
        lines = open(path, 'r')
        if key_position is not None:
            self.key_position = [int(p) for p in key_position.split(" ")]
            for line in lines:
                col_pos = []  # reserve COL position in t
                t, t_id = line.strip().split('\t')
                t = t.split(" ")
                for i, v in enumerate(t):
                    if v == 'COL':
                        col_pos.append(i)
                for i, pos in enumerate(col_pos):
                    if int(i) not in self.domain:
                        if i < len(col_pos) - 1:
                            self.domain[i] = [' '.join(t[pos + 3:col_pos[i + 1]])]
                        else:
                            self.domain[i] = [' '.join(t[pos + 3:])]
                    else:
                        if i < len(col_pos) - 1:
                            self.domain[i].append(' '.join(t[pos + 3:col_pos[i + 1]]))
                        else:
                            self.domain[i].append(' '.join(t[pos + 3:]))
        if da is not None:
            self.augmenter = Augmenter()
            print("With Data Augment")
        elif neg_da is not None:
            self.augmenter = Augmenter()
            if dk is not None:  # domain knowledge
                print("With Domain Knowledge")
                if dk == 'product':
                    injector = ProductDKInjector()
                else:
                    injector = GeneralDKInjector()
                self.KD = injector.build_knowledge_domain(path)
            print("With Neg Data Augment")
        elif dual_da is not None:
            self.augmenter = Augmenter()
            print("With Dual Data Augment")
            if dk is not None:  # domain knowledge
                print("With Domain Knowledge")
                if dk == 'product':
                    injector = ProductDKInjector()
                else:
                    injector = GeneralDKInjector()
                self.KD = injector.build_knowledge_domain(path)
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.t)

    def __getitem__(self, index):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        x = self.tokenizer.encode(self.t[index],
                                  max_length=self.max_len,
                                  truncation=True)
        x_neg_aug_list = []
        if self.dual_da is not None:
            x_pos_aug = self.augmenter.pos_augment_sent_2(self.t[index], self.key_position)
            x_pos_aug = self.tokenizer.encode(x_pos_aug,
                                              max_length=self.max_len,
                                              truncation=True)
            for i in range(1):
                x_neg_aug = self.augmenter.neg_augment_sent_2(self.t[index], self.key_position, self.domain, self.dk,
                                                          self.KD)
                x_neg_aug = self.tokenizer.encode(x_neg_aug,
                                              max_length=self.max_len,
                                              truncation=True)
                x_neg_aug_list.append(x_neg_aug)
            return x, x_pos_aug, x_neg_aug_list, self.t_id[index]


        if self.da is not None:
            x_pos_aug = self.augmenter.pos_augment_sent_2(self.t[index], self.key_position)
            x_pos_aug = self.tokenizer.encode(x_pos_aug,
                                              max_length=self.max_len,
                                              truncation=True)
            return x, x_pos_aug, self.t_id[index]

        x_neg_aug_list = []
        if self.neg_da is not None:
            # for i in range(8):
            #     t_neg_aug = self.augmenter.neg_augment_sent(self.t[index],  self.key_position, self.domain, self.neg_da)
            #     x_neg_aug = self.tokenizer.encode(t_neg_aug,
            #                               max_length=self.max_len,
            #                               truncation=True)
            #     x_neg_aug_list.append(x_neg_aug)
            x_neg_aug = self.augmenter.neg_augment_sent_2(self.t[index], self.key_position, self.domain, self.dk,
                                                          self.KD)
            x_neg_aug = self.tokenizer.encode(x_neg_aug,
                                              max_length=self.max_len,
                                              truncation=True)

            # return x, x_neg_aug_list, self.t_id[index]
            return x, x_neg_aug, self.t_id[index]
        else:
            return x, self.t_id[index]


        # augment if da is set


    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 4:
            x1, x2, x3, y = zip(*batch)

            maxlen = 256
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            x3 = [[xii + [0] * (maxlen - len(xii)) for xii in xi] for xi in x3]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(x3), \
                   torch.LongTensor(y)
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = 256
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            # x2 = [[xii + [0] * (maxlen - len(xii)) for xii in xi] for xi in x2]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = 256
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)