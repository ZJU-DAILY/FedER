import copy
import pickle
import struct
from model.base_model import*

from utils import *
import socket

def train(args, loader1, eval_loader1, set1_id2t, set2_id2t, set_size=0, start=0):
    begin = time.time()
    fix_seed(args.seed)
    PROJ_DIR = abspath(dirname(__file__))
    task = args.task
    task = task.replace('/', '_')

    def match_loader(path):
        match = []
        p = open(path, 'r')
        i = 0
        for line in p:
            id_1, id_2 = line.strip().split(' ')
            match.append((int(id_1), int(id_2)))
            i += 1
        return match

    TrueMatch = match_loader(args.match_path)

    if not os.path.exists(join(PROJ_DIR, 'flog1')):
        os.mkdir(join(PROJ_DIR, 'flog1'))

    if not os.path.exists(join(PROJ_DIR, 'flog1', task)):
        os.mkdir(join(PROJ_DIR, 'flog1', task))


    data_queue1 = None
    pos_aug_queue1 = None
    neg_aug_queue1 = None
    aug_queue1 = None
    batch_aug = None
    pos_batch_aug = None
    neg_batch_aug = None

    device = torch.device(args.device)
    model = BertEncoder(args).to(device)  # encoder q
    _model = BertEncoder(args).to(device)  # encoder k
    _model.update(model)  # moto update
    iteration = 0
    lr = args.lr
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    host = "localhost"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, args.port))
    s.listen(5)
    print("listening...")
    conn, addr = s.accept()  # acc connect
    print('[+] Connected with', addr)
    best_f1 = 0
    for round_id in range(start, args.rounds):
        filename = os.path.join(PROJ_DIR, 'flog1',
                                task + '/da={}_negda={}_dualda={}_dk={}_bsize={}_qsize={}_rounds={}_localep={}_'
                                       'lr={}_key={}_dp_mechanism={}_clip={}_epsilon={}.txt'.format(
                                    args.da,
                                    args.neg_da,
                                    args.dual_da,
                                    args.dk,
                                    str(args.batch_size),
                                    str(args.queue_length),
                                    str(args.rounds),
                                    str(args.local_ep),
                                    str(args.lr),
                                    str(args.key_position),
                                    str(args.dp_mechanism),
                                    str(args.dp_clip),
                                    str(args.dp_epsilon)
                                ))

        epoch_loss = []
        begin = time.time()
        # Local Update
        for iter in range(args.local_ep):
            adjust_learning_rate(optimizer, int(round_id)*int(iter+1), lr)
            batch_loss = []
            for batch_id, batch in tqdm(enumerate(loader1)):  # data from table1
                if len(batch) == 2:
                    tuple_tokens, _ = batch

                elif len(batch) == 3:
                    tuple_tokens, tuple_aug_tokens, _ = batch  # T(batch_size, 256) T(batch_size, neg_num, 256) T(batch_size)
                    if aug_queue1 == None:
                        aug_queue1 = tuple_aug_tokens.unsqueeze(0)
                    else:
                        aug_queue1 = torch.cat((aug_queue1, tuple_aug_tokens.unsqueeze(0)), dim=0)
                else:
                    tuple_tokens, tuple_pos_aug_tokens, tuple_neg_aug_tokens, _ = batch
                    if pos_aug_queue1 == None:
                        pos_aug_queue1 = tuple_pos_aug_tokens.unsqueeze(0)
                    else:
                        pos_aug_queue1 = torch.cat((pos_aug_queue1, tuple_pos_aug_tokens.unsqueeze(0)), dim=0)
                    if neg_aug_queue1 == None:
                        neg_aug_queue1 = tuple_neg_aug_tokens.unsqueeze(0)
                    else:
                        neg_aug_queue1 = torch.cat((neg_aug_queue1, tuple_neg_aug_tokens.unsqueeze(0)), dim=0)

                if data_queue1 == None:

                    data_queue1 = tuple_tokens.unsqueeze(0)
                else:

                    data_queue1 = torch.cat((data_queue1, tuple_tokens.unsqueeze(0)), dim=0)


                if data_queue1.shape[0] == args.queue_length + 1:
                    pos_batch = data_queue1[0]
                    data_queue1 = data_queue1[1:]
                    neg_queue = data_queue1

                    if args.da is not None or args.neg_da is not None:
                        batch_aug = aug_queue1[0]
                        aug_queue1 = aug_queue1[1:]
                    elif args.dual_da is not None:
                        pos_batch_aug = pos_aug_queue1[0]
                        pos_aug_queue1 = pos_aug_queue1[1:]
                        neg_batch_aug = neg_aug_queue1[0]
                        neg_aug_queue1 = neg_aug_queue1[1:]
                    else:
                        pass
                else:
                    continue

                optimizer.zero_grad()
                pos_1 = model(pos_batch.squeeze(0))  # online model
                with torch.no_grad():
                    _model.eval()
                    if args.da is not None:  # pos aug
                        pos_2 = _model(batch_aug.squeeze(0))

                    elif args.neg_da is not None:
                        batch_aug = batch_aug.view(-1, 256)
                        neg_aug = _model(batch_aug.squeeze(0))
                        pos_2 = _model(pos_batch.squeeze(0))

                    elif args.dual_da is not None:
                        pos_2 = _model(pos_batch_aug.squeeze(0))
                        neg_batch_aug = neg_batch_aug.view(-1, 256)
                        neg_aug = _model(neg_batch_aug.squeeze(0))
                    else:
                        pos_2 = _model(pos_batch.squeeze(0))

                    neg_shape = neg_queue.shape
                    neg_queue = neg_queue.reshape(neg_shape[0] * neg_shape[1],
                                                  neg_shape[2])  # batch_size, que.size, token_maxlen
                    neg_value = _model(neg_queue)
                    del neg_queue

                # contrastive
                if args.neg_da is not None or args.dual_da is not None:
                    contrastive_loss = model.contrastive_loss(pos_1, pos_2, neg_value, neg_aug)
                else:
                    contrastive_loss = model.contrastive_loss(pos_1, pos_2, neg_value)

                loss = contrastive_loss
                del pos_1
                del pos_2
                del neg_value

                iteration += 1

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=True)

                if args.dp_mechanism != 'no_dp':
                    clip_gradients(model, args.dp_mechanism, args.dp_clip)
                optimizer.step()
                _model.update(model)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if args.dp_mechanism != 'no_dp':
            add_noise(model, args.dp_mechanism, args.lr, args.dp_clip, args.dp_epsilon, args.batch_size, dp_delta=None)
        model_weight = model.state_dict()
        local_loss = sum(epoch_loss) / len(epoch_loss)
        w = pickle.dumps(model_weight)
        conn.sendall(w)

        need_recv_size = len(w)
        recv = b""
        while need_recv_size > 0:
            x = conn.recv(min(0xffffffff, need_recv_size))
            recv += x
            need_recv_size -= len(x)

        weight_B = pickle.loads(recv)

        w_avg = copy.deepcopy(model_weight)
        for key in model_weight.keys():
            w_avg[key] += weight_B[key]
            w_avg[key] = torch.div(w_avg[key], 2)
        # update parameters
        model.load_state_dict(w_avg)
        _model.update(model)
        end = time.time()

        print('round: {} loss: {}'.format(round_id, local_loss))
        with open(filename, 'a+') as f:
            f.write('round: {} loss: {}\n'.format(round_id, local_loss))
            f.write('round {} time: {}\n'.format(round_id, end - begin))

        # PPSM
        ids_1, ids_2, vector_1, vector_2 = list(), list(), list(), list()
        with torch.no_grad():
            model.eval()
            for sample_id_1, (tuple_token_1, tuple_id_1) in enumerate(eval_loader1):
                tuple_vector_1 = model(tuple_token_1)
                tuple_vector_1 = tuple_vector_1.squeeze().detach().cpu().numpy()
                vector_1.append(tuple_vector_1)
                tuple_id_1 = tuple_id_1.squeeze().tolist()
                if isinstance(tuple_id_1, int):
                    tuple_id_1 = [tuple_id_1]
                ids_1.extend(tuple_id_1)


        v1 = np.vstack(vector_1).astype(np.float32)
        v1 = preprocessing.normalize(v1)
        header_struct = conn.recv(4)  # 4 length
        unpack_res = struct.unpack('i', header_struct)
        need_recv_size = unpack_res[0]
        recv = b""
        while need_recv_size > 0:
            xx = conn.recv(min(0xfffffffffff, need_recv_size))
            recv += xx
            need_recv_size -= len(xx)
        ids_2 = pickle.loads(recv)


        row = v1.shape[0]
        noise = np.random.random((1, 768))
        noise = np.repeat(noise, row, axis=0)
        noise_v1 = v1 + noise

        noise_v1 = pickle.dumps(noise_v1)
        header = struct.pack('i', len(noise_v1))
        conn.send(header)
        conn.sendall(noise_v1)


        header_struct = conn.recv(4)  # 4 length
        unpack_res = struct.unpack('i', header_struct)
        need_recv_size = unpack_res[0]
        recv = b""
        while need_recv_size > 0:
            x = conn.recv(min(0xfffffffffff, need_recv_size))
            recv += x
            need_recv_size -= len(x)
        v2_noise = pickle.loads(recv)


        header_struct = conn.recv(4)  # 4 length
        unpack_res = struct.unpack('i', header_struct)
        need_recv_size = unpack_res[0]
        recv = b""
        while need_recv_size > 0:
            xx = conn.recv(min(0xfffffffffff, need_recv_size))
            recv += xx
            need_recv_size -= len(xx)
        topkB = pickle.loads(recv)


        sim_score = torch.tensor(v1.dot(v2_noise.T))
        distA, topkA = torch.topk(sim_score, k=2, dim=1)


        ## evaluate
        inverse_ids_1, inverse_ids_2 = dict(), dict()
        for idx, _id in enumerate(ids_1):
            inverse_ids_1[_id] = idx  # entity id to index
        for idx, _id in enumerate(ids_2):
            inverse_ids_2[_id] = idx  # entity id to index

        # sim_score = torch.tensor(v1.dot(v2.T))
        # distA, topkA = torch.topk(sim_score, k=2, dim=1)
        # distB, topkB = torch.topk(sim_score, k=2, dim=0)
        # topkB = topkB.t()
        lenA = topkA.shape[0]
        PseudoMatch = []
        for e1_index in range(lenA):
            e2_index = topkA[e1_index][0].item()
            if e1_index == topkB[e2_index][0].item():
                PseudoMatch.append((ids_1[e1_index], ids_2[e2_index]))

        match_dic = {}  # dict A->B
        invers_match_dic = {}  # dict B->A
        PseudoMatch_dic = {}  # dict A->B
        invers_PseudoMatch_dic = {}  # dict B->A
        len_pm = len(PseudoMatch)
        for pair in TrueMatch:  # list
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

        tp = 0
        tp_sim = []

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

        if tp == 0:
            f1 = 0
        else:
            f1 = round((2 * tp / len(TrueMatch) * tp / len_pm) / (tp / len_pm + tp / len(TrueMatch)), 3)
        if f1 > best_f1:
            best_f1 = f1

    # torch.save(model.state_dict(), ("./checkpoints/out1{}".format(args.task) + "trained.pth")) #保存模型参数到checkpoint
    conn.close()
    s.close()
    end = time.time()
    print("============================================================\n")
    print("F1: ", best_f1)




