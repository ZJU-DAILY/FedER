import argparse
import random
import time

from utils import *
import numpy as np
import torch
import torch.multiprocessing as mp
from data import *
import model.client1 as clien1
import model.client2 as clien2
def parse_options(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--lm", type=str, default='bert-base-uncased')
    parser.add_argument("--fp16", dest="fp16", action="store_false")
    parser.add_argument("--alpha_aug", type=float, default=0.8)

    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default='LaBSE')
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--neg_da", type=str, default=None)
    parser.add_argument("--dual_da", type=str, default=None)
    parser.add_argument("--dk", type=str, default='product')
    parser.add_argument('--local_ep', type=int, default=1)
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--rounds', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--queue_length', type=int, default=16)
    parser.add_argument('--dp_mechanism', type=str, default='Laplace') #'no_dp'  'Laplace'  'Gaussian'
    parser.add_argument('--dp_clip', type=int, default = 10)
    parser.add_argument('--dp_epsilon', type = float, default = 0.15) #run 0.001
    # parser.add_argument('-add_noise', type=int, default=0)
    parser.add_argument('--key_position', type=str, default=None)  # multi att '0 1 2...'
    parser.add_argument('-port', type=int, default=8800)
    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--task', type=str, default="/Structure/DBLP-ACM")
    parser.add_argument('--path1', type=str, default="./dataset/Structure/DBLP-ACM/train1.txt")
    parser.add_argument('--path2', type=str, default="./dataset/Structure/DBLP-ACM/train2.txt")
    parser.add_argument('--match_path', type=str, default="./dataset/Structure/DBLP-ACM/match.txt")














    return parser.parse_args()


if __name__ == '__main__':
    begin =time.time()
    parser = argparse.ArgumentParser()
    args = parse_options(parser)
    fix_seed(37)
    mp.set_start_method('spawn', force=True)
    train_set1 = MyDataset(path=args.path1,
                       max_len=256,
                       lm=args.lm,
                       da=args.da,
                       neg_da=args.neg_da,
                       dual_da=args.dual_da,
                       key_position=args.key_position,
                       dk=args.dk
                       )
    padder = train_set1.pad
    set1_size = train_set1.__len__()
    set1_id2t = train_set1.id2t
    loader1 = Data.DataLoader(
        dataset=train_set1,  # torch TensorDataset format
        batch_size=args.batch_size,  # all test data
        shuffle=True,
        drop_last=True,
        collate_fn=padder
    )

    del train_set1
    # generate test data for A
    valid_set1 = MyDataset(path=args.path1,
                       max_len=256,
                       lm=args.lm,
                       da=None,
                       neg_da=None
                       )
    padder = valid_set1.pad
    eval_loader1 = Data.DataLoader(
        dataset=valid_set1,  # torch TensorDataset format
        batch_size=args.batch_size,  # all test data
        shuffle=True,
        drop_last=False,
        collate_fn=padder
    )

    train_set2 = MyDataset(path=args.path2,
                           max_len=256,
                           lm=args.lm,
                           da=args.da,
                           neg_da=args.neg_da,
                           dual_da=args.dual_da,
                           key_position=args.key_position,
                           dk=args.dk
                           )
    padder = train_set2.pad
    set2_size = train_set2.__len__()
    set2_id2t = train_set2.id2t
    loader2 = Data.DataLoader(
        dataset=train_set2,  # torch TensorDataset format
        batch_size=args.batch_size,  # all test data
        shuffle=True,
        drop_last=True,
        collate_fn=padder
    )

    del train_set2
    # generate test data for B
    valid_set2 = MyDataset(path=args.path2,
                           max_len=256,
                           lm=args.lm,
                           da=None,
                           neg_da=None
                           )
    padder = valid_set2.pad
    eval_loader2 = Data.DataLoader(
        dataset=valid_set2,  # torch TensorDataset format
        batch_size=args.batch_size,  # all test data
        shuffle=True,
        drop_last=False,
        collate_fn=padder
    )


    p1 = mp.Process(target=clien1.train, args=(args, loader1, eval_loader1, set1_id2t, set2_id2t, set1_size,))
    p2 = mp.Process(target=clien2.train, args=(args, loader2, eval_loader2,set2_size,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
