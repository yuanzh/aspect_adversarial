import sys
import gzip

import numpy as np

rng = np.random.RandomState(0)

def say(s, stream=sys.stderr):
    stream.write("{}".format(s))
    stream.flush()
    
def stop():
    sys.stdin.readline()
    
def assertion(cond):
    if not cond:
        raise Exception()
    
def load_embedding_iterator(filename, vocab=None, skip_head=True):
    line_idx = 0
    fopen = gzip.open if filename.endswith(".gz") else open
    with fopen(filename) as fin:
        for line in fin:
            line_idx += 1
            if skip_head and line_idx == 1:
                continue    # skip head
            
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                if vocab and word not in vocab: # skip useless vec
                    continue
                
                vals = np.array([ float(x) for x in parts[1:] ])
                yield word, vals

def file_line_iterator(filename):
    with open(filename) as fin:
        for line in fin:
            yield line.strip()