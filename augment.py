import random
from os.path import abspath, dirname, join, exists
from knowledge import *
class Augmenter(object):
    """Data augmentation operator.

    Support both span and attribute level augmentation operators.
    """
    def __init__(self):
        pass

    def augment(self, tokens, labels, left_pos, right_pos, op='del'):
        if 'del' in op:  #丢弃选中的连续几个token
            # insert padding to keep the length consistent
            # span_len = random.randint(1, 3)
            span_len = random.randint(1, 4)
            pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
            if pos1 < 0:
                return tokens, labels
            if pos2 < left_pos or pos1 > right_pos:
                new_tokens = tokens[:pos1] + tokens[pos2+1:]
                new_labels = labels[:pos1] + labels[pos2+1:]

            else:
                return tokens, labels
        elif 'swap' in op:  #shuffle选中的连续几个token的位置

            span_len = random.randint(2, 4)
            pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
            if pos1 < 0:
                return tokens, labels
            sub_arr = tokens[pos1:pos2+1]
            random.shuffle(sub_arr)
            new_tokens = tokens[:pos1] + sub_arr + tokens[pos2+1:]
            new_labels = labels[:pos1] + ['O'] * (pos2 - pos1 + 1) + labels[pos2+1:]

        elif 'replicate' in op:
            span_len = random.randint(2, 4)
            pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
            if pos1 < 0:
                return tokens, labels
            re_arr = tokens[pos1:pos2 + 1]
            re_num = random.randint(1, 3)
            pos = self.sample_position(tokens, labels)
            new_tokens = tokens[:pos] + re_arr*re_num + tokens[pos:]
            new_labels = labels[:pos] + ['O']*(pos2 - pos1 + 1)*re_num + labels[pos:]

        elif 'drop_token' in op:  #随机丢掉一些token
            new_tokens, new_labels = [], []
            for token, label in zip(tokens, labels):
                if label != 'O' or random.randint(0, 4) != 0:
                    new_tokens.append(token)
                    new_labels.append(label)
        elif 'ins' in op:   #加符号
            pos = self.sample_position(tokens, labels)

            symbol = random.choice('-*.,#&')
            new_tokens = tokens[:pos] + [symbol] + tokens[pos:]
            new_labels = labels[:pos] + ['O'] + labels[pos:]


        # elif 'append_col' in op:
        #     col_starts = [i for i in range(len(tokens)) if tokens[i] == 'COL']
        #     col_ends = [0] * len(col_starts)
        #     col_lens = [0] * len(col_starts)
        #     for i, pos in enumerate(col_starts):
        #         if i == len(col_starts) - 1:
        #             col_lens[i] = len(tokens) - pos
        #             col_ends[i] = len(tokens) - 1
        #         else:
        #             col_lens[i] = col_starts[i + 1] - pos
        #             col_ends[i] = col_starts[i + 1] - 1
        #
        #         if tokens[col_ends[i]] == '[SEP]':
        #             col_ends[i] -= 1
        #             col_lens[i] -= 1
        #             break
        #     candidates = [i for i, le in enumerate(col_lens) if le > 0]
        #     if len(candidates) >= 2:
        #         idx1, idx2 = random.sample(candidates,k=2)
        #         start1, end1 = col_starts[idx1], col_ends[idx1]
        #         sub_tokens = tokens[start1:end1+1]
        #         sub_labels = labels[start1:end1+1]
        #         val_pos = 0
        #         for i, token in enumerate(sub_tokens):
        #             if token == 'VAL':
        #                 val_pos = i + 1
        #                 break
        #         sub_tokens = sub_tokens[val_pos:]
        #         sub_labels = sub_labels[val_pos:]
        #
        #         end2 = col_ends[idx2]
        #         new_tokens = []
        #         new_labels = []
        #         for i in range(len(tokens)):
        #             if start1 <= i <= end1:
        #                 continue
        #             new_tokens.append(tokens[i])
        #             new_labels.append(labels[i])
        #             if i == end2:
        #                 new_tokens += sub_tokens
        #                 new_labels += sub_labels
        #         return new_tokens, new_labels
        #     else:
        #         new_tokens, new_labels = tokens, labels
        # elif 'drop_col' in op:
        #     col_starts = [i for i in range(len(tokens)) if tokens[i] == 'COL']
        #     col_ends = [0] * len(col_starts)
        #     col_lens = [0] * len(col_starts)
        #     for i, pos in enumerate(col_starts):
        #         if i == len(col_starts) - 1:
        #             col_lens[i] = len(tokens) - pos
        #             col_ends[i] = len(tokens) - 1
        #         else:
        #             col_lens[i] = col_starts[i + 1] - pos
        #             col_ends[i] = col_starts[i + 1] - 1
        #
        #         if tokens[col_ends[i]] == '[SEP]':
        #             col_ends[i] -= 1
        #             col_lens[i] -= 1
        #     candidates = [i for i, le in enumerate(col_lens) if le <= 8]
        #     if len(candidates) > 0:
        #         idx = random.choice(candidates)
        #         start, end = col_starts[idx], col_ends[idx]
        #         new_tokens = tokens[:start] + tokens[end+1:]
        #         new_labels = labels[:start] + labels[end+1:]
        #     else:
        #         new_tokens, new_labels = tokens, labels
        else:
            new_tokens, new_labels = tokens, labels

        return new_tokens, new_labels

    def dual_augment_sent(self, text, key_position, domain, dk, KD, op='all'):
        neg_text = text
        pos_text = text
        if key_position is not None:  #att level
            tokens = text.split(' ')
            col_pos = []
            lr_pos = []
            for i, token in enumerate(tokens):
                if token == 'COL':
                    col_pos.append(i)
            for key in key_position:
                if key < len(col_pos) - 1:
                    lr_pos.append((col_pos[key]+3, col_pos[key+1]))
                else:
                    lr_pos.append((col_pos[key]+3, len(tokens)))
            k = random.randint(0, len(key_position)-1) if len(key_position) > 1 else 0
            left_pos = lr_pos[k][0]
            right_pos = lr_pos[k][1]
            neg_text = self.neg_augment(key_position[k], left_pos, right_pos, neg_text, domain) # subsittute
            if random.random() > 1:
                pos_text = self.pos_augment_2(left_pos, right_pos, pos_text) # shift
            else:
                pos_text = self.pos_augment(left_pos, right_pos, pos_text, op) # random aug with key stayed
        else:
            pos_text = self.pos_augment(-1, -1, pos_text, op)
        if dk is not None:  # domain knowledge
            if dk == 'product':
                injector = ProductDKInjector()
            else:
                injector = GeneralDKInjector()
            tag = injector.get_tag(neg_text)
            # choose one tag to neg_aug
            if len(tag) > 0:
                i = random.randint(0, len(tag)-1)
                type = list(tag.keys())[i]
                pos1, pos2 = tag[type]
                origin_token = neg_text[pos1: pos2]
                new_tokens = random.sample(KD[type], 2)
                new_token = new_tokens[1] if new_tokens[0] == origin_token else new_tokens[0]
                neg_text = neg_text.replace(origin_token, new_token)
        return pos_text, neg_text

    def neg_augment_sent_2(self, text, key_position, domain, dk, KD):
        neg_text = text
        if key_position is not None:  # att level
            tokens = text.split(' ')
            col_pos = []
            lr_pos = []
            for i, token in enumerate(tokens):
                if token == 'COL':
                    col_pos.append(i)
            for key in key_position:
                if key < len(col_pos) - 1:
                    lr_pos.append((col_pos[key] + 3, col_pos[key + 1]))
                else:
                    lr_pos.append((col_pos[key] + 3, len(tokens)))
            k = random.randint(0, len(key_position) - 1) if len(key_position) > 1 else 0
            left_pos = lr_pos[k][0]
            right_pos = lr_pos[k][1]
            neg_text = self.neg_augment(key_position[k], left_pos, right_pos, neg_text, domain)  # subsittute


        # if dk is not None and random.random() > 0.5:  # domain knowledge
        if dk is not None:
            # if dk == 'product':
            #     injector = ProductDKInjector()
            #     injector = ProductDKInjector()
            # else:
            #     injector = GeneralDKInjector()
            if dk == 'product':
                tag = get_tag(neg_text, dk)
                # choose one tag to neg_aug
                if len(tag) > 0:
                    i = random.randint(0, len(tag) - 1)
                    type = list(tag.keys())[i]
                    pos1, pos2 = tag[type]
                    origin_token = neg_text[pos1: pos2]
                    new_tokens = random.sample(KD[type], 2)
                    new_token = new_tokens[1] if new_tokens[0] == origin_token else new_tokens[0]
                    neg_text = neg_text.replace(origin_token, new_token)
        return neg_text

    def pos_augment_sent_2(self, text, key_position, op='all'):
        # return text
        if key_position is not None:  # att level
            tokens = text.split(' ')
            col_pos = []
            lr_pos = []
            for i, token in enumerate(tokens):
                if token == 'COL':
                    col_pos.append(i)
            for key in key_position:
                if key < len(col_pos) - 1:
                    lr_pos.append((col_pos[key] + 3, col_pos[key + 1]))
                else:
                    lr_pos.append((col_pos[key] + 3, len(tokens)))
            k = random.randint(0, len(key_position) - 1) if len(key_position) > 1 else 0
            left_pos = lr_pos[k][0]
            right_pos = lr_pos[k][1]
            if random.random() > 0.5:
                pos_text = self.pos_augment_2(left_pos, right_pos, text) # shift
            else:
                pos_text = self.pos_augment(left_pos, right_pos, text, op) # random aug with key stayed
        else:
            pos_text = self.pos_augment(-1, -1, text, op)
        return pos_text

    def get_labels(self, tokens):
        labels = []
        for token in tokens:
            if token in ['COL', 'VAL']:
                labels.append('X')
            else:
                labels.append('O')
        return labels


    def neg_augment(self, att_pos, left_pos, right_pos, text, domain):
        tokens = text.split(' ')
        att_origin = ' '.join(tokens[left_pos: right_pos])
        new_atts = random.sample(domain[att_pos], 2)
        new_att = new_atts[1] if new_atts[0] == att_origin else new_atts[0]
        tokens = tokens[:left_pos] + new_att.split() + tokens[right_pos:]
        neg_text = ' '.join(tokens)
        return neg_text

    def pos_augment(self, left_pos, right_pos, text, op='all'):
        tokens = text.split(' ')
        # avoid the special tokens
        labels = self.get_labels(tokens)

        if op == 'all':
            ops = ['del', 'swap', 'replicate', 'ins']
            # randnum = random.randint(0, 100)
            # random.seed(randnum)
            r = random.randint(0, 3)
            op = ops[r]
            tokens, labels = self.augment(tokens, labels, left_pos, right_pos, op=op)
            # for op in ops:  #全部放进去
            #     tokens, labels = self.augment(tokens, labels, left_pos, right_pos, op=op)
        else:
            tokens, labels = self.augment(tokens, labels, left_pos, right_pos, op=op)
        pos_text = ' '.join(tokens)

        return pos_text


    def pos_augment_2(self, left_pos, right_pos, text):
        tokens = text.split(' ')
        labels = self.get_labels(tokens)
        # avoid the special tokens
        key_att = tokens[left_pos: right_pos]
        if random.random() > 0.5:
            tokens = tokens[:left_pos] + tokens[right_pos:]
            labels = labels[:left_pos] + labels[right_pos:]
        pos = self.sample_position(tokens, labels)
        new_tokens = tokens[:pos] + key_att + tokens[pos:]
        new_labels = labels[:pos] + ['O']*len(key_att) + labels[pos:]
        pos_text = ' '.join(new_tokens)
        return pos_text


    def augment_sent(self, text, op='all'):

        # 50% of chance of no-aug
        # if random.randint(0, 1) == 0:
        #     return text

        # tokenize the sentence
        current = ''
        tokens = text.split(' ')

        # avoid the special tokens
        labels = []
        for token in tokens:
            if token in ['COL', 'VAL']:
                labels.append('HD')
            else:
                labels.append('O')

        if op == 'all':
            # RandAugment: https://arxiv.org/pdf/1909.13719.pdf
            N = 3
            ops = ['del', 'swap', 'drop_col', 'append_col']
            for op in random.choices(ops, k=N):
                tokens, labels = self.augment(tokens, labels, op=op)
        else:
            tokens, labels = self.augment(tokens, labels, op=op)



        results = ' '.join(tokens)

        return results

    def sample_span(self, tokens, labels, span_len=3):
        candidates = []
        for idx, token in enumerate(tokens):
            if idx + span_len - 1 < len(labels) and ''.join(labels[idx:idx+span_len]) == 'O'*span_len:
                candidates.append((idx, idx+span_len-1))
        if len(candidates) <= 0:
            return -1, -1
        return random.choice(candidates)

    def sample_position(self, tokens, labels, tfidf=False):
        candidates = []
        for idx, token in enumerate(tokens):
            if labels[idx] == 'O':
                candidates.append(idx)
        if len(candidates) <= 0:
            return -1
        return random.choice(candidates)

    def neg_augment_sent(self, text, key_position, domain, op='all'):
        tokens = text.split(' ')
        for att_pos in key_position:
            pos_count = 0
            left_pos = 0
            right_pos = 0
            for i, token in enumerate(tokens):
                if token == 'COL':
                    pos_count += 1
                    if pos_count == att_pos:
                        left_pos = i + 3
                    if pos_count == att_pos + 1:
                        right_pos = i
                        break
            att_origin = ' '.join(tokens[left_pos: right_pos])

            new_atts = random.sample(domain[att_pos], 2)
            if new_atts[0] == att_origin:
                new_att = new_atts[1]
            else:
                new_att = new_atts[0]

            tokens = tokens[:left_pos] + new_att.split() + tokens[right_pos:]

        results = ' '.join(tokens)

        return results



# if __name__ == '__main__':
#     ag = Augmenter()
#     text = 'COL title VAL pocket with 25 x3 pack 2gb COL category VAL storage COL brand VAL innovera COL modelno VAL 39701 COL price VAL 5.28'
#
#     domain ={}
#     domain[0] = ["01", "02", '03']
#     domain[1] = ['11', '12', '13']
#     domain[2] = ['21', '22', '23']
#     domain[3] = ['31', '32', '33']
#     KD = {}
#     KD['SIZE'] = ['9X9', '99x99']
#     KD['CAPA'] = ['5gb', '500gb']
#     pos, neg = ag.dual_augment_sent(text,None,domain,'product', KD)
#     print(text)
#     print(pos)
#     print(neg)
