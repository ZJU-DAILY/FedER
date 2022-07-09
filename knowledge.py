import os
import re

import spacy


class DKInjector:
    def __init__(self):
        self.KD = {}
        self.initialize()

    def initialize(self):
        pass

    def transform(self, entry):
        return entry

    def get_tag(self, entry):
        tag = {}
        return tag

    # def transform_file(self, input_fn, overwrite=False):
    #     out_paug = input_fn + '.paug'
    #     out_naug = input_fn + '.naug'
    #     if not os.path.exists(out_paug) or \
    #         os.stat(out_paug).st_size == 0 or overwrite:
    #         with open(out_paug, 'w') as fout:
    #             for line in open(input_fn):
    #                 text = line.split('\t')
    #                 if len(text) == 2:
    #                     entry = self.transform(text[0])
    #                     fout.write(entry + '\t'+ text[1])
    #     return out_paug, out_naug


    def build_knowledge_domain(self, input_fn):
        for line in open(input_fn):
            text = line.split('\t')
            if len(text) == 2:
               self.transform(text[0])
        return self.KD




class ProductDKInjector(DKInjector):
    def initialize(self):
        """Initialize spacy"""
        self.nlp = spacy.load('en_core_web_lg')
        self.KD['SIZE'] = []
        self.KD['CAPA'] = []
        self.KD['VER'] = []
    # def get_tag(self, entry):
    #     tag = {}
    #     try:
    #         span = re.search('\d+\.?\d*\s?x\s?\d+\.?\d*', entry).span()
    #         tag['SIZE'] = span
    #     except:
    #         pass
    #
    #     try:
    #         span = re.search('\d+\.?\d*\s?(g[b]*|G[B]*)', entry).span()
    #         tag['CAPA'] = span
    #     except:
    #         pass
    #     try:
    #         span = re.search('[a-zA-Z]+\s?\d+\.?\d*', entry).span()
    #         tag['VER'] = span
    #     except:
    #         pass
    #     return tag

    def transform(self, entry):
        # regular expression
        try:
            match = re.search('\d+\.?\d*\s?x\s?\d+\.?\d*', entry).group()
            self.KD['SIZE'].append(match)
        except:
            pass
        try:
            match = re.search('\d+\.?\d*\s?(g[b]*|G[B]*)', entry).group()
            self.KD['CAPA'].append(match)
        except:
            pass
        # try:
        #     match = re.search('[a-zA-Z]+\s?\d+\.?\d*', entry).group()
        #     self.KD['VER'].append(match)
        # except:
        #     pass

        # res = ''
        # doc = self.nlp(entry, disable=['tagger', 'parser'])
        # ents = doc.ents
        # start_indices = {}
        # end_indices = {}
        #
        # for ent in ents:
        #     if str(ent) == 'VAL' or str(ent) == 'COL':
        #         continue
        #     start, end, label = ent.start, ent.end, ent.label_
        #     if label in ['NORP', 'GPE', 'LOC', 'PERSON', 'PRODUCT']:  #认成这些的全部当做product
        #         if 'VAL' in str(ent):
        #             start_indices[start+1] = 'PRODUCT'
        #         else:
        #             start_indices[start] = 'PRODUCT'
        #         end_indices[end] = 'PRODUCT'
        #
        # for idx, token in enumerate(doc):
        #     if idx in start_indices:
        #         res += start_indices[idx] + ' '
        #     if idx in end_indices:
        #         res += end_indices[idx] + ' '
        #     # normalizing the numbers
        #     if token.like_num:
        #         try:
        #             val = float(token.text)
        #             if val == round(val):
        #                 res += '%d ' % (int(val))
        #             else:
        #                 res += '%.2f ' % (val)
        #         except:
        #             res += token.text + ' '
        #     elif len(token.text) >= 7 and \
        #          any([ch.isdigit() for ch in token.text]):
        #         res += 'ID ' + token.text + ' '
        #     else:
        #         res += token.text + ' '
        # return res.strip()





class GeneralDKInjector(DKInjector):
    def initialize(self):
        """Initialize spacy"""
        self.nlp = spacy.load('en_core_web_lg')

    def transform(self, entry):
        res = ''
        doc = self.nlp(entry, disable=['tagger', 'parser'])
        ents = doc.ents
        start_indices = {}
        end_indices = {}

        for ent in ents:
            start, end, label = ent.start, ent.end, ent.label_
            if label in ['PERSON', 'ORG', 'LOC', 'PRODUCT', 'DATE', 'QUANTITY', 'TIME']:
                start_indices[start] = label
                end_indices[end] = label

        for idx, token in enumerate(doc):
            if idx in start_indices:
                res += start_indices[idx] + ' '

            # normalizing the numbers
            if token.like_num:
                try:
                    val = float(token.text)
                    if val == round(val):
                        res += '%d ' % (int(val))
                    else:
                        res += '%.2f ' % (val)
                except:
                    res += token.text + ' '
            elif len(token.text) >= 7 and \
                 any([ch.isdigit() for ch in token.text]):
                res += 'ID ' + token.text + ' '
            else:
                res += token.text + ' '
        return res.strip()


def get_tag(entry, dk):
    tag = {}
    if dk == 'product':
        try:
            span = re.search('\d+\.?\d*\s?x\s?\d+\.?\d*', entry).span()
            tag['SIZE'] = span
        except:
            pass

        try:
            span = re.search('\d+\.?\d*\s?(g[b]*|G[B]*)', entry).span()
            tag['CAPA'] = span
        except:
            pass
        # try:
        #     span = re.search('[a-zA-Z]+\s?\d+\.?\d*', entry).span()
        #     tag['VER'] = span
        # except:
        #     pass
    return tag

# if __name__ == '__main__':
#     config = "x"
#     name = "x"
#     input ="COL title VAL sandisk 2p2322322 iphone COL brand VAL sandisk COL modelno VAL sdsdq008ga11m COL price VAL 24.88	1972"
#     injector = ProductDKInjector()
#     res = injector.transform(input)
#     print(res)
