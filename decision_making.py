import socket
import sys

import torch
import torch.multiprocessing as mp
from phe import *
import time
import pickle
import struct
import numpy as np
from tqdm import tqdm


def cal_match1(v1, args=8700):
    host = "localhost"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, 8700))
    s.listen(5)
    print("listening...")
    conn, addr = s.accept()  # acc connect
    print('[+] Connected with', addr)
    public_key, private_key = paillier.generate_paillier_keypair() # key generation

    v1 = v1.tolist()
    begin = time.time()
    encrypted_list = [[public_key.encrypt(x) for x in l] for l in v1]  # encrypt
    end = time.time()
    enc = pickle.dumps(encrypted_list)
    header = struct.pack('i', len(enc))
    conn.send(header)
    conn.sendall(enc) # send encrypted matrix [[v1]]

    header_struct = conn.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(0xffffffff, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    enc_res = pickle.loads(recv) # recv [[V1]]V2

    enc_res = enc_res.tolist()
    res = [[private_key.decrypt(x) for x in l] for l in enc_res]  # decryt
    res = np.array(res)



def cal_match2(v2):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "localhost"
    try:
        s.connect((host,8700))
    except Exception:
        print('[!] Server not found or not open')
        sys.exit()
    header_struct = s.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = s.recv(min(0xffffffff, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    enc_v1 = pickle.loads(recv)  # recv [[v1]]
    enc_v1 = np.array(enc_v1)
    enc_res = enc_v1.dot(v2.T)  # [[v1]]v2
    enc_res = pickle.dumps(enc_res)
    header = struct.pack('i', len(enc_res))
    s.send(header)
    s.sendall(enc_res) # send encryted result to A


def cal_match1_noise(v1):
    host = "localhost"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, 8790))
    s.listen(5)
    print("listening...")
    conn, addr = s.accept()  # acc connect
    print('[+] Connected with', addr)
    col = v1.shape[1]
    row = v1.shape[0]
    noise = np.random.random((1, col))
    noise = np.repeat(noise, row, axis=0)
    noise_v1 = v1 + noise
    send_noise_v1 = pickle.dumps(noise_v1)
    header = struct.pack('i', len(send_noise_v1))
    conn.send(header)
    conn.sendall(send_noise_v1)

    header_struct = conn.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(0xffffffff, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    recv_noise_v2 = pickle.loads(recv)

    sim_score = torch.tensor(v1.dot(recv_noise_v2.T))
    distA, topkA = torch.topk(sim_score, k=2, dim=1)


    send_topkA = pickle.dumps(topkA)
    header = struct.pack('i', len(send_topkA))
    conn.send(header)
    conn.sendall(send_topkA)

    header_struct = conn.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(0xffffffff, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    recv_noise_v2 = pickle.loads(recv)

    conn.close()

    print("cal_match1_noise",topkA)





def cal_match2_noise(v2):
    col = v2.shape[1]
    row = v2.shape[0]
    noise = np.random.random((1, col))
    noise = np.repeat(noise, row, axis=0)
    noise_v2 = v2 + noise
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "localhost"
    try:
        s.connect((host, 8790))
    except Exception:
        print('[!] Server not found or not open')
        sys.exit()
    header_struct = s.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = s.recv(min(0xffffffff, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    recv_noise_v1 = pickle.loads(recv)


    send_noise_v2 = pickle.dumps(noise_v2)
    header = struct.pack('i', len(send_noise_v2))
    s.send(header)
    s.sendall(send_noise_v2)

    sim_score = torch.tensor(v2.dot(recv_noise_v1.T))
    distB, topkB = torch.topk(sim_score, k=2, dim=1)


    header_struct = s.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = s.recv(min(0xffffffff, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    recv_noise_v2 = pickle.loads(recv)


    send_topkB = pickle.dumps(topkB)
    header = struct.pack('i', len(send_topkB))
    s.send(header)
    s.sendall(send_topkB)
    s.close()


    print("cal_match2_noise",topkB)





if __name__ == '__main__':
    begin = time.time()
    # v1 = np.random.uniform(-1,1,(2000,768))
    # v2 = np.random.uniform(-1,1,(2000,768))
    # v1 = np.random.uniform(-1, 1, (4000, 768))
    # v2 = np.random.uniform(-1, 1, (4000, 768))
    # v1 = np.random.uniform(-1, 1, (6000, 768))
    # v2 = np.random.uniform(-1, 1, (6000, 768))
    # v1 = np.random.uniform(-1, 1, (8000, 768))
    # v2 = np.random.uniform(-1, 1, (8000, 768))
    # v1 = np.random.uniform(-1, 1, (10000, 768))
    # v2 = np.random.uniform(-1, 1, (10000, 768))
    v1 = np.random.uniform(-1, 1, (12000, 768))
    v2 = np.random.uniform(-1, 1, (12000, 768))
    p1 = mp.Process(target=cal_match1_noise, args=(v1, ))
    p2 = mp.Process(target=cal_match2_noise, args=(v2,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end = time.time()
    print("running time",end-begin)




def add_noise(M):
    col = M.shape[1]
    row = M.shape[0]
    noise = np.random.random((1, col))
    noise = np.repeat(noise, row, axis=0)
    noise_M = M + noise
    return noise_M