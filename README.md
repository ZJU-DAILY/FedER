# **FedER: Effective and Privacy-Preserving Entity Resolution via Federated Learning**

FedER, an effective and privacy-preserving deep entity resolution framework powered by federated learning, which achieves promising ER accuracy and privacy protection of different data owners. FedER consists of two phases, i.e., federated match-aware representation learning (FMRL) and privacy-preserving similarity measurement (PPSM). In the first phase, FMRL embeds tuples of two different data owners into a uni-space. FMRL enables the match-aware representation learning across different owners without manually labeled matches/non-matches. In the second phase, PPSM supports the match/non-match decision-making in a privacy-preserving manner. 

## Requirements

* Python 3.7
* PyTorch 1.10.1
* CUDA 11.5
* NVIDIA A100 GPU
* HuggingFace Transformers 4.9.2
* NVIDIA Apex (fp16 training) 

Please refer to the source code to install all required packages in Python

## Datasets

We conduct experiments on five datasets, including DBLP-ACM, Walmart-Amazon, Amazon-Google, DBLP-ACM(Dirty), and Walmart-Amazon(Dirty). We provide all the datasets. 

##Run Experimental Case

To conduct the FedER for effective and privacy-preserving deep entity resolution on DBLP-ACM:

```
python fed_main.py --task "/Structure/DBLP-ACM"
```

The meaning of the flags:

--task: the datasets conducted on. e.g."/Structure/DBLP-ACM"

--path1: the path of relational table ownered by the data owner A. e.g. "./dataset/Structure/DBLP-ACM/train1.txt"

--path2: the path of relational table ownered by the data owner B. e.g. "./dataset/Structure/DBLP-ACM/train2.txt"

--match_path: the path of ground truth. e.g. "./dataset/Structure/DBLP-ACM/match.txt"

--rounds: total federated training round. e.g. 35

--local_epoch: the number of local epochs before each parameter exchange. e.g. 1

--dp_epsilon: the privacy budget of each owner to perform a single-round federated training. e.g. 0.15


--queue_length: the size of the queue to contain the prevoius batches. e.g. 16


## Acknowledgementt

We use the code of [Ditto](https://github.com/megagonlabs/ditto).

The original datasets are from [DeepMather](https://github.com/anhaidgroup/deepmatcher)
