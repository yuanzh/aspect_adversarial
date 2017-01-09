import sys
sys.path.append('../')

import numpy as np
import utils
from utils import say, load_embedding_iterator
from nn import EmbeddingLayer
from collections import Counter
import operator
import string

printable = set(string.printable)
REL_PAD = -2
REL_UNK = -1
REL_POS = 1
REL_NEG = 0

def read_syn_data(filename):
    x = []
    y = []
    rel = []    # relevance score
    cnt = 0
    lines = utils.file_line_iterator(filename)
    line = next(lines, None)
    while line is not None:
        data = line.split()
        r_y = int(data[0])
        n = int(data[1])
        r_x = []
        r_rel = []
        for _ in xrange(n):
            data = next(lines, None).split("\t")
            r = int(data[0])
            tokens = data[1].split()
            if len(tokens) == 0:
                continue
            r_x.append(tokens)
            r_rel.append(r)
        line = next(lines, None)
        line = next(lines, None)
        if len(r_x) == 0:
            continue
        y.append(r_y)
        x.append(r_x)
        rel.append(r_rel)
        cnt += 1
        if cnt % 50000 == 0: print cnt

    print "get from", filename, cnt
    return x, y, rel

            
def create_embedding_layer(emb_filename, n_d=100, vocab_dict=None,
        unk="<unk>", padding="<padding>", fix_init_embs=True):
    
    embs = load_embedding_iterator(emb_filename, vocab_dict, skip_head=True) if emb_filename else None
    embedding_layer = EmbeddingLayer(
            n_d = n_d,
            vocab = [ padding, unk ] + (vocab_dict.keys() if not embs else []),
            embs = embs,
            fix_init_embs = fix_init_embs
        )
    return embedding_layer
    
def get_vocab_dict(data):
    cnt = Counter()
    for x in data:
        cnt.update(x)
    vocab_dict = {}
    for w in cnt:
        if cnt[w] < 3: continue
        vocab_dict[w] = len(vocab_dict)
    return vocab_dict

def create_doc_array(inst, padding_id, max_doc_len, max_sent_len):
    doc_emb_ids = inst[0]
    # create max_doc_len*max_sent_len matrix
    if len(doc_emb_ids) < max_doc_len:
        doc_emb_ids = [np.array([padding_id], dtype="int32") for _ in xrange(max_doc_len - len(doc_emb_ids))] + doc_emb_ids 
    arr = np.column_stack([ np.pad(x, (max_sent_len-len(x), 0), 'constant',
                            constant_values=padding_id) for x in doc_emb_ids])
    assert arr.shape == (max_sent_len, max_doc_len)
    return arr

def create_input(data, padding_id):
    doc_length = [len(d[0]) for d in data]
    sent_length = [len(x) for d in data for x in d[0]]
    if len(sent_length) == 0: sent_length.append(0)
    
    max_doc_len = max(1, max(doc_length))
    max_sent_len = max(1, max(sent_length))
    
    idxs = np.column_stack(
            [create_doc_array(d, padding_id, max_doc_len, max_sent_len).ravel() for d in data]
            )
    idxs = idxs.reshape(max_sent_len, max_doc_len, len(data))
    idys = np.array([d[1] for d in data], dtype="int32")
    
    # relevance
    gold_rels = np.column_stack([np.array([REL_PAD] * (max_doc_len-len(d[2])) + d[2], dtype="int32") for d in data])
    assert gold_rels.shape == (max_doc_len, len(data))
    
    for d in data: assert len(d[2]) == len(d[0])
    input_lst = [idxs, idys, gold_rels]
        
    return input_lst

def create_batches(data, batch_size, padding_id, label=True, sort=True, shuffle=True):
    if label:
        for d in data:
            assert d[1] != -1
    if sort:
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)
    batches = []
    for i in xrange(0, len(data), batch_size):
        #idxs, idys
        input_lst = create_input(data[i:i+batch_size], padding_id)
        batches.append(input_lst)
    if shuffle:
        idx = np.random.permutation(len(batches))
        new_batches = [batches[i] for i in idx]
        new_data = reduce(operator.add, [data[i*batch_size:(i+1)*batch_size] for i in idx])
        batches, data = new_batches, new_data
        assert len(new_data) == len(data)
    if not label:
        # set all label to 0
        for b in batches:
            b[1][:] = 0
        
    return batches, data

def map_doc_to_ids(d, embedding_layer, max_sent_len):
    ids = [ embedding_layer.map_to_ids(x)[:max_sent_len] for x in d ]
    return ids
    
def map_to_id(data, vocab_dict):
    ids = [vocab_dict[x] if x in vocab_dict else -1 for x in data]
    assert min(ids) >= 0, "Unknown label"
    return ids

def shuffle_data(data):
    return np.random.permutation(data)
