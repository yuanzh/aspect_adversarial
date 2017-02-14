import numpy as np
import argparse
import sys
from subprocess import call

np.random.seed(0)

POS_LAB = 1
NEG_LAB = 0
POS_REL = 1
NEG_REL = 0
UNK_REL = -1

n_sent_range = [8, 12]
n_gold_sent_range = [1, 1]
positive_ratio = 0.5
aspect_ratio = 0.9  # how many sentences have aspect
aspect_range = [[0, 4], [4, 8]]
polarity_name_range = [[0, 3], [2, 5]]
context_name_range = [[0, 9], [3, 12]]
aspect_name_range = [0, 9]
word_range = [0, 5000]
len_sent_range = [15, 30]
len_context_range = [4, 6]
source_aspect = 0
target_aspect = 8
aspect_polarity_ratio = 0.90
neutral_polarity_context_ratio = 0.90

def gen_aspect_name(aspect, domain, size):
    assert aspect >= aspect_range[domain][0] and aspect <= aspect_range[domain][1]
    r = np.random.random_integers(aspect_name_range[0], aspect_name_range[1], size=size)
    return ["ASP%d_NAME%d" % (aspect, x) for x in r]

def gen_aspect_context(aspect, domain, size):
    assert aspect >= aspect_range[domain][0] and aspect <= aspect_range[domain][1]
    r = np.random.random_integers(context_name_range[domain][0], context_name_range[domain][1], size=size)
    return ["ASP%d_CTXT%d" % (aspect, x) for x in r]

def gen_word(size):
    r = np.random.randint(word_range[0], word_range[1], size=size)
    return ["WORD%d" % (x) for x in r]

def gen_polarity_name(polarity, domain, size):
    p_str = "NEG" if polarity == NEG_LAB else "POS"
    r = np.random.random_integers(polarity_name_range[domain][0], polarity_name_range[domain][1], size=size)
    return ["%s_NAME%d" % (p_str, x) for x in r]

def gen_aspect_polarity_name(aspect, polarity, domain, size):
    p_str = "NEG" if polarity == NEG_LAB else "POS"
    r = np.random.random_integers(polarity_name_range[domain][0], polarity_name_range[domain][1], size=size)
    return ["ASP%d_%s_NAME%d" % (aspect, p_str, x) for x in r]

def get_polarity_context_word(aspect, polarity, p_str, name, word):
    sa = "ASP%d_" % aspect if aspect >= 0 else ""
    sp = "%s_" % p_str if polarity == 1 else "NEU_"
    sn = "NAME%d_" % name if name >= 0 else ""
    return "%s%s%sCTXT%d" % (sa, sp, sn, word)

def gen_polarity_context(polarity, domain, size):
    p_str = "NEG" if polarity == NEG_LAB else "POS"
    r = np.random.random_integers(context_name_range[domain][0], context_name_range[domain][1], size=size)
    rp = np.random.binomial(n=1, p=1-neutral_polarity_context_ratio, size=size)
    return [get_polarity_context_word(aspect=-1, polarity=xp, p_str=p_str, name=-1, word=x) for x, xp in zip(r, rp)]

def gen_aspect_polarity_context(aspect, polarity, domain, size):
    p_str = "NEG" if polarity == NEG_LAB else "POS"
    r = np.random.random_integers(context_name_range[domain][0], context_name_range[domain][1], size=size)
    rp = np.random.binomial(n=1, p=1-neutral_polarity_context_ratio, size=size)
    return [get_polarity_context_word(aspect=aspect, 
                                      polarity=xp, p_str=p_str, name=-1, word=x) 
            for x, xp in zip(r, rp)]

def generate_sent_wo_aspect(domain):
    # sentence without aspect
    len_sent = np.random.random_integers(len_sent_range[0], len_sent_range[1])
    x = gen_word(len_sent)
    return x

def generate_sent_wo_polarity(aspect, y, domain):
    # positive: ... ASP_NAME ...
    # negative: aspect name doesn't appear
    if y == NEG_LAB:
        x = generate_sent_wo_aspect(domain)
    else:
        # generate context
        len_context = np.random.random_integers(len_context_range[0], len_context_range[1])
        context = gen_aspect_context(aspect, domain, len_context)
        
        # insert aspect
        aspect = gen_aspect_name(aspect, domain, 1)
        insert_position = len_context / 2
        aspect = context[:insert_position] + aspect + context[insert_position:]
        
        # generate random words
        len_sent = np.random.random_integers(len_sent_range[0], len_sent_range[1])
        x = gen_word(max(2, len_sent-len(aspect)))
        insert_position = np.random.random_integers(len(x)-1)
        x = x[:insert_position] + aspect + x[insert_position:]
        
    return x

def generate_sent_wo_common_polarity(aspect_id, gold_aspect, y, domain):
    # positive: ... ASP_NAME ... POS_NAME ...
    # negative: ... ASP_NAME ... NEG_NAME ...

    # generate aspect
    len_context = np.random.random_integers(len_context_range[0], len_context_range[1])
    context = gen_aspect_context(aspect_id, domain, len_context)
    aspect = gen_aspect_name(aspect_id, domain, 1)
    insert_position = len_context / 2
    aspect = context[:insert_position] + aspect + context[insert_position:]
    
    # generate POS/NEG
    len_context = np.random.random_integers(len_context_range[0], len_context_range[1])

    # polarity with aspect
    context = gen_aspect_polarity_context(aspect_id, y, domain, len_context)
    polarity = gen_aspect_polarity_name(aspect_id, y, domain, 1)
    insert_position = len_context / 2
    polarity = context[:insert_position] + polarity + context[insert_position:]
    
    # generate random words
    len_sent = np.random.random_integers(len_sent_range[0], len_sent_range[1])
    x = gen_word(max(2, len_sent-len(aspect)-len(polarity)))
    insert_position = np.random.random_integers(len(x)-1, size=2)
    p1, p2 = min(insert_position), max(insert_position)
    x = x[:p1] + aspect + x[p1:p2] + polarity + x[p2:]

    return x

def generate_sent(aspect_id, gold_aspect, y, domain):
    # positive: ... ASP_NAME ... POS_NAME ...
    # negative: ... ASP_NAME ... NEG_NAME ...

    # generate aspect
    len_context = np.random.random_integers(len_context_range[0], len_context_range[1])
    context = gen_aspect_context(aspect_id, domain, len_context)
    aspect = gen_aspect_name(aspect_id, domain, 1)
    insert_position = len_context / 2
    aspect = context[:insert_position] + aspect + context[insert_position:]
    
    # generate POS/NEG
    len_context = np.random.random_integers(len_context_range[0], len_context_range[1])
    if np.random.rand() < aspect_polarity_ratio or gold_aspect != aspect_id:
        # polarity with aspect
        context = gen_aspect_polarity_context(aspect_id, y, domain, len_context)
        polarity = gen_aspect_polarity_name(aspect_id, y, domain, 1)
        insert_position = len_context / 2
        polarity = context[:insert_position] + polarity + context[insert_position:]
    else:
        # polarity without aspect
        context = gen_polarity_context(y, domain, len_context)
        polarity = gen_polarity_name(y, domain, 1)
        insert_position = len_context / 2
        polarity = context[:insert_position] + polarity + context[insert_position:]
    
    # generate random words
    len_sent = np.random.random_integers(len_sent_range[0], len_sent_range[1])
    x = gen_word(max(2, len_sent-len(aspect)-len(polarity)))
    insert_position = np.random.random_integers(len(x)-1, size=2)
    p1, p2 = min(insert_position), max(insert_position)
    x = x[:p1] + aspect + x[p1:p2] + polarity + x[p2:]

    return x

def generate_doc(gold_aspect, domain, mode):
    global aspect_polarity_ratio
    global neutral_polarity_context_ratio
    if mode == 0:
        aspect_polarity_ratio = 0.9
        neutral_polarity_context_ratio = 0.9
    elif mode == 1:
        # common polarity words: positive 20%, negative 0%
        aspect_polarity_ratio = 0.8
        neutral_polarity_context_ratio = 0.95
    elif mode == 2:
        # common polarity words: 50%
        aspect_polarity_ratio = 0.8
        neutral_polarity_context_ratio = 0.9
    elif mode == 3:
        # common polarity words: 20%
        aspect_polarity_ratio = 0.8
        neutral_polarity_context_ratio = 0.95
    else:
        raise Exception(), "mode must be integers in [0, 3]"
    
    y = POS_LAB if np.random.rand() < positive_ratio else NEG_LAB
    data = []
    n_gold_sent = np.random.random_integers(n_gold_sent_range[0], n_gold_sent_range[1])
    for _ in xrange(n_gold_sent):
        rel = POS_REL
        if mode == 0:
            sent = generate_sent_wo_polarity(gold_aspect, y, domain)
        elif mode == 1:
            if y == NEG_LAB:
                sent = generate_sent_wo_common_polarity(gold_aspect, gold_aspect, y, domain)
            else:
                sent = generate_sent(gold_aspect, gold_aspect, y, domain)
        else:
            sent = generate_sent(gold_aspect, gold_aspect, y, domain)
        data.append([sent, rel])
    
    n_sent = np.random.randint(n_sent_range[0], n_sent_range[1])
    for _ in xrange(n_sent-n_gold_sent):
        if np.random.rand() < aspect_ratio:
            # sentence with aspect
            aspect = np.random.random_integers(aspect_range[domain][0], aspect_range[domain][1])
            if aspect == gold_aspect:
                rel = POS_REL 
                if mode == 0:
                    sent = generate_sent_wo_polarity(aspect, y, domain)
                elif mode == 1:
                    if y == NEG_LAB:
                        sent = generate_sent_wo_common_polarity(aspect, gold_aspect, y, domain)
                    else:
                        sent = generate_sent(aspect, gold_aspect, y, domain)
                else:
                    sent = generate_sent(aspect, gold_aspect, y, domain)
            else:
                rel = NEG_REL
                lab = POS_LAB if np.random.rand() < positive_ratio else NEG_LAB
                if mode == 0:
                    sent = generate_sent_wo_polarity(aspect, lab, domain)
                elif mode == 1:
                    if lab == NEG_LAB:
                        sent = generate_sent_wo_common_polarity(aspect, gold_aspect, lab, domain)
                    else:
                        sent = generate_sent(aspect, gold_aspect, lab, domain)
                else:
                    sent = generate_sent(aspect, gold_aspect, lab, domain)
            
            data.append([sent, rel])
        else:
            # sentence without aspect
            rel = UNK_REL
            sent = generate_sent_wo_aspect(domain)
            data.append([sent, rel])
            
    # shuffle
    idx = np.random.permutation(len(data))
    data = [data[i] for i in idx]
    return data, y

def output_doc(f, data, y):
    f.write("%d %d\n" % (y, len(data)))
    for d in data:
        f.write("%d\t%s\n" % (d[1], " ".join(d[0])))
    f.write("\n")
    
def output_raw(f, data):
    for d in data:
        f.write("%s\n" % " ".join(d[0]))

def generate_data(mode):
    print "Generating data in mode", mode
    raw = []
    fn = "syn" + str(mode)
    
    # source unlabel
    n = 100000
    f = open(fn + ".source.ul", "w")
    for _ in xrange(n):
        data, y = generate_doc(source_aspect, 0, mode)
        output_doc(f, data, NEG_LAB)
        raw.append(data)
    f.close()
    
    # target unlabel
    n = 100000
    f = open(fn + ".target.ul", "w")
    for _ in xrange(n):
        data, y = generate_doc(target_aspect, 1, mode)
        output_doc(f, data, NEG_LAB)
        raw.append(data)
    f.close()
    
    # source train
    n = 50000
    f = open(fn + ".source.train", "w")
    for _ in xrange(n):
        data, y = generate_doc(source_aspect, 0, mode)
        output_doc(f, data, y)
        raw.append(data)
    f.close()
    
    # source dev
    n = 2000
    f = open(fn + ".dev", "w")
    for _ in xrange(n):
        data, y = generate_doc(source_aspect, 0, mode)
        output_doc(f, data, y)
        raw.append(data)
    f.close()
    
    # target label
    n = 2000
    f = open(fn + ".target.train", "w")
    for _ in xrange(n):
        data, y = generate_doc(target_aspect, 1, mode)
        output_doc(f, data, y)
        raw.append(data)
    f.close()

    # target test
    n = 2000
    f = open(fn + ".test", "w")
    for _ in xrange(n):
        data, y = generate_doc(target_aspect, 1, mode)
        output_doc(f, data, y)
        raw.append(data)
    f.close()
    
    # raw
    f = open(fn + ".raw", "w")
    for data in raw:
        output_raw(f, data)
    n = 50000
    for _ in xrange(n):
        data, y = generate_doc(source_aspect, 0, mode)
        output_raw(f, data)
        data, y = generate_doc(target_aspect, 1, mode)
        output_raw(f, data)
    f.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--mode",
            type = int,
            default = 0
        )
    args = argparser.parse_args()
    generate_data(args.mode)
    
    print "Generate word embeddings using word2vec"
    in_name = "syn" + str(args.mode) + ".raw"
    out_name = "syn" + str(args.mode) + ".emb.100"
    call(["../../word2vec/word2vec", "-train", in_name, "-output", out_name, "-size", "100", "-binary", "0", "-cbow",  "0"])
    