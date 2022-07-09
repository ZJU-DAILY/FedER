import copy
import pickle
import struct
import sys
from model.base_model import*
from utils import *
import socket


def train(args, loader2, eval_loader2, set_size=0, start=0):
    begin = time.time()
    fix_seed(args.seed)
    PROJ_DIR = abspath(dirname(__file__))
    task = args.task
    task = task.replace('/', '_')
    if not os.path.exists(join(PROJ_DIR, 'flog2')):
        os.mkdir(join(PROJ_DIR, 'flog2'))
    if not os.path.exists(join(PROJ_DIR, 'flog2', task)):
        os.mkdir(join(PROJ_DIR, 'flog2', task))

    data_queue2 = None
    pos_aug_queue2 = None
    neg_aug_queue2 = None
    aug_queue2 = None
    batch_aug = None
    pos_batch_aug = None
    device = torch.device(args.device)
    model = BertEncoder(args).to(device)  # encoder q
    _model = BertEncoder(args).to(device)  # encoder k
    _model.update(model)  # moto update
    iteration = 0
    lr = args.lr
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "localhost"
    try:
        s.connect((host, args.port))  #
    except Exception:
        print('[!] Server not found or not open')
        sys.exit()
    for round_id in range(start, args.rounds):
        # adjust_learning_rate(optimizer, round_id, lr)
        filename = os.path.join(PROJ_DIR, 'flog2',
                                task + '/lm={}_da={}_negda={}_dualda={}_dk={}_bsize={}_qsize={}_rounds={}_localep={}_'
                                       'lr={}_key={}_dp_mechanism={}_clip={}_epsilon={}.txt'.format(
                                    args.lm,
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

        for iter in range(args.local_ep):
            adjust_learning_rate(optimizer, int(round_id) * int(iter+1), lr)
            batch_loss = []
            for batch_id, batch in tqdm(enumerate(loader2)):  # data from table1
                if len(batch) == 2:
                    tuple_tokens, _ = batch
                elif len(batch) == 3:
                    tuple_tokens, tuple_aug_tokens, _ = batch  # T(batch_size, 256) T(batch_size, neg_num, 256) T(batch_size)
                    if aug_queue2 == None:
                        aug_queue2 = tuple_aug_tokens.unsqueeze(0)
                    else:
                        aug_queue2 = torch.cat((aug_queue2, tuple_aug_tokens.unsqueeze(0)), dim=0)
                else:
                    tuple_tokens, tuple_pos_aug_tokens, tuple_neg_aug_tokens, tuple_id = batch
                    if pos_aug_queue2 == None:
                        pos_aug_queue2 = tuple_pos_aug_tokens.unsqueeze(0)
                    else:
                        pos_aug_queue2 = torch.cat((pos_aug_queue2, tuple_pos_aug_tokens.unsqueeze(0)), dim=0)
                    if neg_aug_queue2 == None:
                        neg_aug_queue2 = tuple_neg_aug_tokens.unsqueeze(0)
                    else:
                        neg_aug_queue2 = torch.cat((neg_aug_queue2, tuple_neg_aug_tokens.unsqueeze(0)), dim=0)

                if data_queue2 == None:
                    data_queue2 = tuple_tokens.unsqueeze(0)
                else:
                    data_queue2 = torch.cat((data_queue2, tuple_tokens.unsqueeze(0)), dim=0)


                if data_queue2.shape[0] == args.queue_length + 1:
                    pos_batch = data_queue2[0]
                    data_queue2 = data_queue2[1:]
                    neg_queue = data_queue2
                    if args.da is not None or args.neg_da is not None:
                        batch_aug = aug_queue2[0]
                        aug_queue2 = aug_queue2[1:]
                    elif args.dual_da is not None:
                        pos_batch_aug = pos_aug_queue2[0]
                        pos_aug_queue2 = pos_aug_queue2[1:]
                        neg_batch_aug = neg_aug_queue2[0]
                        neg_aug_queue2 = neg_aug_queue2[1:]
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
        need_recv_size = len(w)
        recv = b""
        while need_recv_size > 0:
            x = s.recv(min(0xfffffffffff, need_recv_size))
            recv += x
            need_recv_size -= len(x)
        end = time.time()
        begin = time.time()
        s.sendall(w)
        end = time.time()
        weight_A = pickle.loads(recv)
        # aggregate
        w_avg = copy.deepcopy(model_weight)
        for key in model_weight.keys():
            w_avg[key] += weight_A[key]
            w_avg[key] = torch.div(w_avg[key], 2)
        # update parameters
        model.load_state_dict(w_avg)
        _model.update(model)

        #PPSM
        ids_2, vector_2 = list(), list()
        with torch.no_grad():
            model.eval()
            for sample_id_2, (tuple_token_2, tuple_id_2) in enumerate(eval_loader2):
                tuple_vector_2 = model(tuple_token_2)
                tuple_vector_2 = tuple_vector_2.squeeze().detach().cpu().numpy()
                vector_2.append(tuple_vector_2)
                tuple_id_2 = tuple_id_2.squeeze().tolist()
                if isinstance(tuple_id_2, int):
                    tuple_id_2 = [tuple_id_2]
                ids_2.extend(tuple_id_2)
        v2 = np.vstack(vector_2).astype(np.float32)
        v2 = preprocessing.normalize(v2)
        ids_2 = pickle.dumps(ids_2)
        header = struct.pack('i', len(ids_2))
        s.send(header)
        s.sendall(ids_2)
        header_struct = s.recv(4)  # 4 length
        unpack_res = struct.unpack('i', header_struct)
        need_recv_size = unpack_res[0]
        recv = b""
        while need_recv_size > 0:
            x = s.recv(min(0xfffffffffff, need_recv_size))
            recv += x
            need_recv_size -= len(x)
        v1_noise = pickle.loads(recv)
        row = v2.shape[0]
        noise = np.random.random((1, 768))
        noise = np.repeat(noise, row, axis=0)
        noise_v2 = v2 + noise

        noise_v2 = pickle.dumps(noise_v2)
        header = struct.pack('i', len(noise_v2))
        s.send(header)
        s.sendall(noise_v2)


        sim_score = torch.tensor(v2.dot(v1_noise.T))
        distB, topkB = torch.topk(sim_score, k=2, dim=1)

        send_topkB = pickle.dumps(topkB)
        header = struct.pack('i', len(send_topkB))
        s.send(header)
        s.sendall(send_topkB)




    s.close()
    end = time.time()
    print("============================================================\n")
    print("running time: ", end - begin)





