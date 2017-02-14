import sys
sys.path.append('../')

import io_util
import argparse
import utils
from utils import say
import theano
import theano.tensor as T
import numpy as np
from nn import get_activation_by_name, create_optimization_updates
from nn.optimization import create_accumulators
from nn import Layer, LeCNN, apply_dropout, sigmoid, tanh, logsoftmax
import time
from collections import Counter

from io_util import REL_PAD, REL_UNK

np.random.seed(0)

class Model(object):
    '''
    classdocs
    '''


    def __init__(self, args, embedding_layer, n_class):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = n_class
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        
    def get_l2_cost(self, params):
        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p**2)
            else:
                l2_cost = l2_cost + T.sum(p**2)
        return l2_cost
    
    def get_params(self, layers):
        params = [ ]
        for l in layers:
            params += l.params
        return params
    
    def get_accumulators(self, layers, accum_dict):
        accum = [[], []]
        for l in layers:
            a = accum_dict[l]
            accum[0] += a[0]
            accum[1] += a[1]
        return accum
    
    def add_accumulators(self, accums, layer, accum_dict):
        a = accum_dict[layer]
        accums[0] += a[0]
        accums[1] += a[1]
    
    def create_accumulators(self, layer):
        print "create accumulators"
        return [create_accumulators(layer.params), create_accumulators(layer.params)]
    
    def get_recon_loss(self, idxs, sent_output):
        len_sent, len_doc_batch, n_d = sent_output.shape
        recon_layer = self.recon_layer
        padding_id = self.padding_id
        dropout = self.dropout
        
        # (len(sent)*len(doc)*batch)*n_e
        input_flat = idxs.ravel()
        true_recon = self.embedding_layer.recon_forward(input_flat)
        sent_output = apply_dropout(sent_output, dropout)
        pred_recon = recon_layer.forward(sent_output.reshape((len_sent*len_doc_batch, n_d)))
        
        # (len(sent)*len(doc)*batch)
        mask = T.cast(T.neq(input_flat, padding_id), theano.config.floatX)
        n = T.sum(mask)
        loss = T.sum((true_recon - pred_recon) ** 2, axis=1) * mask
        loss = T.sum(loss) / n
        return loss
    
    def ready_one_domain(self, idxs, idys, dom_ids, gold_rels, \
                         cnn_layer, rel_hid_layer, rel_out_layer, trans_layer, \
                         dom_hid_layer, dom_out_layer, lab_hid_layer, lab_out_layer):
        dropout = self.dropout
        embedding_layer = self.embedding_layer
        n_d = self.n_d
        n_e = self.n_e
        len_sentence, len_document, batch = idxs.shape
        dw = theano.shared(np.float64(0.4).astype(theano.config.floatX))

        # (len(sent)*len(doc)*batch)*n_e
        sent_input_flat = embedding_layer.forward(idxs.ravel())
        sent_input_flat = apply_dropout(sent_input_flat, dropout)
        
        # len(sent)*(len(doc)*batch)*n_e
        sent_input = sent_input_flat.reshape((len_sentence, len_document*batch, n_e))

        # len(sent) * (len(doc)*batch) * n_d
        sent_output = cnn_layer.forward_all(sent_input)
        
        # reconstruction loss
        recon_loss = self.get_recon_loss(idxs, sent_output)
        
        # max pooling, (len(doc)*batch) * n_d
        sent_embedding = T.max(sent_output, axis=0)
        sent_embedding = apply_dropout(sent_embedding, dropout)

        # relevance score, (len(doc)*batch) * 2
        rel_score = rel_out_layer.forward(rel_hid_layer.forward(sent_embedding)).ravel()
        
        # relevance loss
        gold_rels = gold_rels.ravel()
        rel_mask = T.cast(T.neq(gold_rels, REL_PAD), theano.config.floatX)
        gold_rel_mask = rel_mask * T.cast(T.neq(gold_rels, REL_UNK), theano.config.floatX)
        rel_loss = T.sum((gold_rels - rel_score) ** 2 * gold_rel_mask)
        n_rel_loss = batch
        rel_loss = rel_loss / n_rel_loss
        
        # document embedding via weighted combination, batch * n_d
        rel_score = (rel_score * rel_mask).reshape((len_document, batch))
        weighted_sent_embedding = sent_embedding.reshape((len_document, batch, n_d)) * rel_score.dimshuffle((0, 1, 'x'))
        n = T.sum(rel_score, axis=0) + 1e-8 * T.sum(rel_mask, axis=0)
        orig_doc_embedding = T.sum(weighted_sent_embedding, axis=0) / n.dimshuffle((0, 'x'))
        
        # transform document embedding, batch * n_d
        doc_embedding = trans_layer.forward(orig_doc_embedding)
        
        # domain prediction. batch * 2 
        dom_logprob = dom_out_layer.forward(apply_dropout(dom_hid_layer.forward(doc_embedding), dropout))

        # domain loss
        dom_loss = -dom_logprob[T.arange(batch), dom_ids]
        dom_loss = dw * T.mean(dom_loss)

        # domain adv loss
        adv_loss = self.rho * (-dom_loss)

        # label prediction. batch * n_c
        lab_logprob = lab_out_layer.forward(apply_dropout(lab_hid_layer.forward(doc_embedding), dropout))
        lab_prob = T.exp(lab_logprob)
        
        # label loss
        lab_loss = -lab_logprob[T.arange(batch), idys]
        lab_loss = T.mean(lab_loss)
        
        return lab_loss, rel_loss, dom_loss, adv_loss, lab_prob, recon_loss
    
    def ready(self):
        args = self.args
        #n_domain = 2
        accum_dict = self.accum_dict = {}
        
        # len(sent) * len(doc) * batch
        s_idxs = self.s_idxs = T.itensor3()
        t_idxs = self.t_idxs = T.itensor3()

        # batch
        s_idys = self.s_idys = T.ivector()
        t_idys = self.t_idys = T.ivector()
        
        # batch
        s_dom_ids = self.s_dom_ids = T.ivector()
        t_dom_ids = self.t_dom_ids = T.ivector()
        
        # len(doc) * batch, 0: negative, 1: positive, -1: REL_UNK, -2, REL_PAD
        s_gold_rels = self.s_gold_rels = T.imatrix() 
        t_gold_rels = self.t_gold_rels = T.imatrix() 
        
        # has label flag, 0: no, 1: yes
        s_has_lab = self.s_has_lab = T.iscalar()
        t_has_lab = self.t_has_lab = T.iscalar()
        
        self.dropout = theano.shared(np.float64(args.dropout).astype(
                            theano.config.floatX))

        embedding_layer = self.embedding_layer
        if not embedding_layer.fix_init_embs:
            accum_dict[embedding_layer] = self.create_accumulators(embedding_layer)

        activation = get_activation_by_name(args.activation)
        n_d = self.n_d = args.hidden_dim
        n_e = self.n_e = embedding_layer.n_d
        n_c = self.nclasses
        self.rho = theano.shared(np.float64(0.0).astype(theano.config.floatX))

        self.source_k = 2

        # CNN to encode sentence into embedding
        cnn_layer = self.cnn_layer = LeCNN(
                n_in = n_e,
                n_out = n_d,
                activation=activation,
                order = args.cnn_window_size,
                BN = True,
            )
        accum_dict[cnn_layer] = self.create_accumulators(cnn_layer)
        
        # softmax layer to predict the label of the document
        self.lab_hid_layer = lab_hid_layer = Layer(
                n_in = n_d,
                n_out = n_d,
                activation = activation,
            )
        accum_dict[lab_hid_layer] = self.create_accumulators(lab_hid_layer)
        self.lab_out_layer = lab_out_layer = Layer(
                n_in = n_d,
                n_out = n_c,
                activation = logsoftmax,
            )
        accum_dict[lab_out_layer] = self.create_accumulators(lab_out_layer)
        
        # hidden layer to predict the domain of the document
        dom_hid_layer = self.dom_hid_layer = Layer(
                n_in = n_d,
                n_out = n_d,
                activation = activation,
            )
        accum_dict[dom_hid_layer] = self.create_accumulators(dom_hid_layer)

        # softmax layer to predict the domain of the document
        dom_out_layer = self.dom_out_layer = Layer(
                n_in = n_d,
                n_out = 2,
                activation = logsoftmax,
            )
        accum_dict[dom_out_layer] = self.create_accumulators(dom_out_layer)

        # for each domain, a vector parameter to compute the relevance score
        rel_hid_layer = self.rel_hid_layer =  Layer(
                n_in = n_d,
                n_out = n_d,
                activation = activation,
            )
        accum_dict[rel_hid_layer] = self.create_accumulators(rel_hid_layer)
        s_rel_out_layer = self.s_rel_out_layer =  Layer(
                n_in = n_d,
                n_out = 1,
                activation = sigmoid,
            )
        accum_dict[s_rel_out_layer] = self.create_accumulators(s_rel_out_layer)
        t_rel_out_layer = self.t_rel_out_layer =  Layer(
                n_in = n_d,
                n_out = 1,
                activation = sigmoid,
            )
        accum_dict[t_rel_out_layer] = self.create_accumulators(t_rel_out_layer)
        
        # transformation to domain independent layer
        trans_layer = self.trans_layer = Layer(
                n_in = n_d,
                n_out = n_d,
                activation = activation,
                has_bias=False,
                init_zero=True,
            )
        accum_dict[trans_layer] = self.create_accumulators(trans_layer)
        val = np.eye(n_d, dtype=theano.config.floatX)
        identity_mat = theano.shared(val)
        trans_layer.W.set_value(val)
        
        # reconstruction layer
        recon_layer = self.recon_layer = Layer(
                n_in = n_d,
                n_out = n_e,
                activation = tanh,
            )
        accum_dict[recon_layer] = self.create_accumulators(recon_layer)
        
        # construct network
        s_lab_loss, s_rel_loss, s_dom_loss, s_adv_loss, s_lab_prob, s_recon_loss = self.ready_one_domain(
                         s_idxs, s_idys, s_dom_ids, s_gold_rels, \
                         cnn_layer, rel_hid_layer, s_rel_out_layer, trans_layer, \
                         dom_hid_layer, dom_out_layer, lab_hid_layer, lab_out_layer)
        self.s_lab_loss, self.s_rel_loss, self.s_dom_loss, self.s_adv_loss, self.s_lab_prob, self.s_recon_loss = \
                        s_lab_loss, s_rel_loss, s_dom_loss, s_adv_loss, s_lab_prob, s_recon_loss
        
        t_lab_loss, t_rel_loss, t_dom_loss, t_adv_loss, t_lab_prob, t_recon_loss = self.ready_one_domain(
                         t_idxs, t_idys, t_dom_ids, t_gold_rels, \
                         cnn_layer, rel_hid_layer, t_rel_out_layer, trans_layer, \
                         dom_hid_layer, dom_out_layer, lab_hid_layer, lab_out_layer)
        self.t_lab_loss, self.t_rel_loss, self.t_dom_loss, self.t_adv_loss, self.t_lab_prob, self.t_recon_loss = \
                        t_lab_loss, t_rel_loss, t_dom_loss, t_adv_loss, t_lab_prob, t_recon_loss
        
        # transformation regularization
        trans_reg = self.trans_reg = args.trans_reg * T.sum((trans_layer.W - identity_mat) ** 2)
        
        # domain cost
        layers = [ dom_out_layer, dom_hid_layer ]
        self.dom_params = self.get_params(layers)
        self.dom_accums = self.get_accumulators(layers, accum_dict)
        self.dom_cost = s_dom_loss + t_dom_loss + args.l2_reg * self.get_l2_cost(self.dom_params)
        
        # label cost
        lab_layers = [ lab_out_layer, lab_hid_layer ]
        lab_params = self.get_params(lab_layers)
        lab_cost = s_has_lab * self.source_k * s_lab_loss + t_has_lab * t_lab_loss \
                    + args.l2_reg * (s_has_lab + t_has_lab) * self.get_l2_cost(lab_params)
            
        # total cost
        other_layers = [ cnn_layer, s_rel_out_layer, t_rel_out_layer, rel_hid_layer, trans_layer, recon_layer ]
        other_params = self.get_params(other_layers)
        self.other_cost_except_dom = lab_cost + s_rel_loss + t_rel_loss + s_adv_loss + t_adv_loss + trans_reg \
                     + s_recon_loss + t_recon_loss \
                     + args.l2_reg * self.get_l2_cost(other_params)
        self.other_params_except_dom = lab_params + other_params
        self.other_accums_except_dom = self.get_accumulators(lab_layers + other_layers, accum_dict)
        if not embedding_layer.fix_init_embs:
            self.other_params_except_dom += embedding_layer.params
            self.add_accumulators(self.other_accums_except_dom, embedding_layer, accum_dict)
        
        # info
        layers = lab_layers + other_layers + [ dom_out_layer, dom_hid_layer ]
        params = self.params = self.get_params(layers)
        if not embedding_layer.fix_init_embs:
            self.params += embedding_layer.params
        say("num of parameters: {}\n".format(
            sum(len(x.get_value(borrow=True).ravel()) for x in params)
        ))

    def train(self, source_train, target_train, source_ul, target_ul, dev, test):
        args = self.args
        n_domain = 2
        padding_id = self.padding_id

        start_time = time.time()
        if source_train is not None:
            s_train_batches, source_train = io_util.create_batches(
                                source_train, args.batch, padding_id)
            for b in s_train_batches:
                b.append(self.get_domain_ids(domain_id=0, n_domain=n_domain, batch=len(b[1])))
                
        if target_train is not None:
            t_train_batches, target_train = io_util.create_batches(
                                target_train, args.batch, padding_id)
            for b in t_train_batches:
                b.append(self.get_domain_ids(domain_id=1, n_domain=n_domain, batch=len(b[1])))

        if dev is not None:
            dev_batches, dev = io_util.create_batches(
                            dev, args.batch, padding_id
                        )
            for b in dev_batches:
                b.append(self.get_domain_ids(domain_id=0, n_domain=n_domain, batch=len(b[1])))
            tot = 0
            for b in dev_batches:
                tot += len(b[0].T)
            print "dev size:", tot, len(dev)
            
        if test is not None:
            test_batches, test = io_util.create_batches(
                            test, args.batch, padding_id
                        )
            for b in test_batches:
                b.append(self.get_domain_ids(domain_id=1, n_domain=n_domain, batch=len(b[1])))
            tot = 0
            for b in test_batches:
                tot += len(b[0].T)
            print "test size:", tot, len(test)

        print 'load source unlabeled data'        
        s_ul_batches, source_ul = io_util.create_batches(
                            source_ul, args.batch, padding_id, label=False)
        for b in s_ul_batches:
            b.append(self.get_domain_ids(domain_id=0, n_domain=n_domain, batch=len(b[1])))

        print 'load target unlabeled data'        
        t_ul_batches, target_ul = io_util.create_batches(
                            target_ul, args.batch, padding_id, label=False)
        for b in t_ul_batches:
            b.append(self.get_domain_ids(domain_id=1, n_domain=n_domain, batch=len(b[1])))

        say("{:.2f}s to create training batches\n\n".format(
                time.time()-start_time
            ))

        dom_updates, dom_lr, dom_gnorm = create_optimization_updates(
                               cost = self.dom_cost,
                               params = self.dom_params,
                               method = args.learning,
                               lr = args.learning_rate,
                               gsums = self.dom_accums[0],
                               xsums = self.dom_accums[1],
                        )[:3]
                        
        other_updates, other_lr, other_gnorm = create_optimization_updates(
                               cost = self.other_cost_except_dom,
                               params = self.other_params_except_dom,
                               method = args.learning,
                               lr = args.learning_rate,
                               gsums = self.other_accums_except_dom[0],
                               xsums = self.other_accums_except_dom[1],
                        )[:3]
        
        BNupdates = self.cnn_layer.get_updates()
        train_func = theano.function(
                inputs = [ self.s_idxs, self.t_idxs, self.s_idys, self.t_idys, self.s_gold_rels, self.t_gold_rels, \
                          self.s_dom_ids, self.t_dom_ids, self.s_has_lab, self.t_has_lab ],
                outputs = [ self.dom_cost, self.other_cost_except_dom, dom_gnorm, other_gnorm, \
                            self.s_lab_loss, self.t_lab_loss, self.s_rel_loss, self.t_rel_loss, \
                            self.s_dom_loss, self.t_dom_loss, self.s_adv_loss, self.t_adv_loss, self.trans_reg, \
                            self.s_recon_loss, self.t_recon_loss ],
                updates = dom_updates.items() + other_updates.items() + BNupdates,
            )

        s_get_loss_and_pred = theano.function(
                inputs = [ self.s_idxs, self.s_idys, self.s_gold_rels, self.s_dom_ids ],
                outputs = [ self.s_lab_prob, self.s_lab_loss, self.s_rel_loss, self.s_dom_loss, self.s_adv_loss, self.s_recon_loss ]
            )
        t_get_loss_and_pred = theano.function(
                inputs = [ self.t_idxs, self.t_idys, self.t_gold_rels, self.t_dom_ids ],
                outputs = [ self.t_lab_prob, self.t_lab_loss, self.t_rel_loss, self.t_dom_loss, self.t_adv_loss, self.t_recon_loss ]
            )

        unchanged = 0
        best_dev = 0
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)
        
        s_ul_batch_ptr = 0
        t_ul_batch_ptr = 0
        s_train_ptr = 0
        t_train_ptr = 0
        test_ptr = 0
        
        print 'Training'
        say("\t"+str([ "{:.1f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                        for x in self.params ])+"\n")
        for epoch in xrange(args.epochs):
            unchanged += 1
            if unchanged > 100: break
                
            s_avg_lab_loss, s_avg_rel_loss, s_avg_dom_loss, s_avg_adv_loss, s_avg_recon_loss = 0.0, 0.0, 0.0, 0.0, 0.0
            t_avg_lab_loss, t_avg_rel_loss, t_avg_dom_loss, t_avg_adv_loss, t_avg_recon_loss = 0.0, 0.0, 0.0, 0.0, 0.0
            avg_dom_cost, avg_other_cost, dom_g, other_g, avg_trans_reg = 0.0, 0.0, 0.0, 0.0, 0.0
            start_time = time.time()

            source_k = self.source_k
            if source_train is not None:
                N = len(s_train_batches) * source_k
            else:
                raise Exception(), "no source training data?"
                
            N_s_ul = len(s_ul_batches)
            N_t_ul = len(t_ul_batches)
            n_s_lab, n_t_lab, n_s_ul, n_t_ul = 0, 0, 0, 0
                
            for t in xrange(N):
                progress = epoch + (t+0.0)/N
                rho_t = 2.0 / (1.0 + np.exp(-0.5*progress)) - 1.0
                rho_t = np.float64(rho_t * args.rho).astype(theano.config.floatX)
                self.rho.set_value(rho_t)
                
                lr_t = args.learning_rate / (1.0 + 0.5*progress) ** 0.75
                lr_t = np.float64(lr_t).astype(theano.config.floatX)
                other_lr.set_value(lr_t)

                s_task = t % source_k
                
                if s_task == 0 and source_train is not None:
                    s_bx, s_by, s_brel, s_bid = s_train_batches[s_train_ptr]
                    s_has_lab = 1
                    s_train_ptr = (s_train_ptr+1)%len(s_train_batches)
                    n_s_lab += 1
                else:
                    s_bx, s_by, s_brel, s_bid = s_ul_batches[s_ul_batch_ptr]
                    s_has_lab = 0
                    s_ul_batch_ptr = (s_ul_batch_ptr+1)%N_s_ul
                    n_s_ul += 1
                    
                t_bx, t_by, t_brel, t_bid = t_ul_batches[t_ul_batch_ptr]
                t_has_lab = 0
                t_ul_batch_ptr = (t_ul_batch_ptr+1)%N_t_ul
                n_t_ul += 1
                    
                dom_cost, other_cost, dom_g, other_g, \
                s_lab_loss, t_lab_loss, s_rel_loss, t_rel_loss, \
                s_dom_loss, t_dom_loss, s_adv_loss, t_adv_loss, trans_reg, \
                s_recon_loss, t_recon_loss = train_func( \
                        s_bx, t_bx, s_by, t_by, s_brel, t_brel, s_bid, t_bid, s_has_lab, t_has_lab)
                    
                avg_dom_cost += dom_cost
                avg_other_cost += other_cost
                avg_trans_reg += trans_reg
                if s_has_lab: s_avg_lab_loss += s_lab_loss
                if t_has_lab: t_avg_lab_loss += t_lab_loss
                s_avg_rel_loss += s_rel_loss
                t_avg_rel_loss += t_rel_loss
                s_avg_dom_loss += s_dom_loss
                t_avg_dom_loss += t_dom_loss
                s_avg_adv_loss += s_adv_loss
                t_avg_adv_loss += t_adv_loss
                s_avg_recon_loss += s_recon_loss
                t_avg_recon_loss += t_recon_loss
                
                say("\r{}/{}/{} {}/{}/{} {}/{}/{} {}/{}/{}/{}    ".format(n_s_lab,s_train_ptr,N, \
                                                                          n_t_lab,t_train_ptr,N, \
                                                                          n_s_ul,s_ul_batch_ptr,N_s_ul, \
                                                                          n_t_ul,t_ul_batch_ptr,test_ptr,N_t_ul))
                    
            say(("Epoch {:.2f}  [{:.2f}m]\n").format(
                        epoch,
                        (time.time()-start_time)/60.0,
                    ))
            say("Source:\t")
            if source_train is not None:
                say(("lab_loss={:.4f}  ").format(s_avg_lab_loss / n_s_lab,))
            say(("rel_loss={:.4f}  dom_loss={:.4f}  adv_loss={:.4f}  recon_loss={:.4f}\n").format(
                    s_avg_rel_loss / N,
                    s_avg_dom_loss / N,
                    s_avg_adv_loss / N,
                    s_avg_recon_loss / N,
                ))
            
            say("Target:\t")
            if target_train is not None:
                say(("lab_loss={:.4f}  ").format(t_avg_lab_loss / n_t_lab,))
            say(("rel_loss={:.4f}  dom_loss={:.4f}  adv_loss={:.4f}  recon_loss={:.4f}\n").format(
                    t_avg_rel_loss / N,
                    t_avg_dom_loss / N,
                    t_avg_adv_loss / N,
                    t_avg_recon_loss / N,
                ))

            say(("Domain cost={:.4f}  |g|={:.4f}  Other cost={:.4f}  |g|={:.4f}  trans_reg={:.4f}\n").format(
                    avg_dom_cost / N,
                    float(dom_g),
                    avg_other_cost / N,
                    float(other_g),
                    avg_trans_reg / N,
                ))
            say("\t"+str([ "{:.1f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                            for x in self.params ])+"\n")

            if dev:
                self.dropout.set_value(0.0)
                self.cnn_layer.set_runmode(1)
                dev_lab_loss, dev_rel_loss, dev_dom_loss, dev_adv_loss, dev_recon_loss, dev_acc, dev_f1 = self.evaluate_data(dev_batches, s_get_loss_and_pred)
                self.dropout.set_value(dropout_prob)
                self.cnn_layer.set_runmode(0)

                if dev_acc > best_dev:
                    best_dev = dev_acc
                    unchanged = 0

                say(("\tdev_lab_loss={:.4f}  dev_rel_loss={:.4f}  dom_loss={:.4f}  adv_loss={:.4f}  recon_loss={:.4f}  dev_acc={:.4f}  dev_f1={}" +
                            "  best_dev={:.4f}\n").format(
                    dev_lab_loss,
                    dev_rel_loss,
                    dev_dom_loss,
                    dev_adv_loss,
                    dev_recon_loss,
                    dev_acc,
                    " ".join(['{:.4f}'.format(x) for x in dev_f1]),
                    best_dev,
                ))
                

            if test:
                self.dropout.set_value(0.0)
                self.cnn_layer.set_runmode(1)
                test_lab_loss, test_rel_loss, test_dom_loss, test_adv_loss, test_recon_loss, test_acc, test_f1 = self.evaluate_data(test_batches, t_get_loss_and_pred)
                self.dropout.set_value(dropout_prob)
                self.cnn_layer.set_runmode(0)
                say(("\ttest_lab_loss={:.4f}  test_rel_loss={:.4f}  dom_loss={:.4f}  adv_loss={:.4f}  recon_loss={:.4f}  test_acc={:.4f}  test_f1={}\n").format(
                    test_lab_loss,
                    test_rel_loss,
                    test_dom_loss,
                    test_adv_loss,
                    test_recon_loss,
                    test_acc,
                    " ".join(['{:.4f}'.format(x) for x in test_f1]),
                ))

    def get_domain_ids(self, domain_id, n_domain, batch):
        arr = np.ones((batch,), dtype="int32") * domain_id
        return arr
    
    def evaluate_data(self, batches, eval_func):
        tot_loss,  tot_acc, tot = 0.0, 0.0, 0.0
        n_c = self.nclasses
        tot_pred, tot_gold, tot_corr = [0.0]*n_c, [0.0]*n_c, [0.0]*n_c
        tot_rel_loss, tot_dom_loss, tot_adv_loss = 0.0, 0.0, 0.0
        tot_recon_loss = 0.0
        for b in batches:
            bx, by, brel, bid = b
            prob, lab_loss, rel_loss, dom_loss, adv_loss, recon_loss = eval_func(bx, by, brel, bid)
            tot_loss += lab_loss
            tot_rel_loss += rel_loss
            tot_dom_loss += dom_loss
            tot_adv_loss += adv_loss
            tot_recon_loss += recon_loss
            for gold_y, p in zip(by, prob):
                pred = np.argmax(p)
                tot += 1
                tot_pred[pred] += 1
                tot_gold[gold_y] += 1
                if pred == gold_y: 
                    tot_acc += 1
                    tot_corr[pred] += 1
        n = len(batches)
        f1 = []
        for p, g, c in zip(tot_pred, tot_gold, tot_corr):
            pre = c / p if p > 0 else 0.0
            rec = c / g if g > 0 else 0.0
            f1.append((2*pre*rec)/(pre+rec+1e-8))
            
        tot_data = sum([len(b[1]) for b in batches])
        say(("\tEvaluate data: {}\n").format(str(tot_data)))
        return tot_loss/n, tot_rel_loss/n, tot_dom_loss/n, tot_adv_loss/n/self.rho.get_value(), tot_recon_loss/n, tot_acc/tot, f1
            
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--source_train",
            type = str,
            default = ""
        )
    argparser.add_argument("--target_train",
            type = str,
            default = ""
        )
    argparser.add_argument("--dev",
            type = str,
            default = ""
        )
    argparser.add_argument("--test",
            type = str,
            default = ""
        )
    argparser.add_argument("--embeddings",
            type = str,
            default = ""
        )
    argparser.add_argument("--source_unlabel",
            type = str,
            default = ""
        )
    argparser.add_argument("--target_unlabel",
            type = str,
            default = ""
        )
    argparser.add_argument("--batch",
            type = int,
            default = 8
        )
    argparser.add_argument("--epochs",
            type = int,
            default = 20
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.2
        )
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 150
        )
    argparser.add_argument("--word_emb_dim",
            type = int,
            default = 100
        )
    argparser.add_argument("--max_doc_len",
            type = int,
            default = 64
        )
    argparser.add_argument("--max_sent_len",
            type = int,
            default = 64
        )
    argparser.add_argument("--cnn_window_size",
            type = int,
            default = 3
        )
    argparser.add_argument("--fix_init_embs",
            type = int,
            default = 0
        )
    argparser.add_argument("--activation", "-act",
            type = str,
            default = "relu"
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 1e-5
        )
    argparser.add_argument("--trans_reg",
            type = float,
            default = 1.0
        )
    argparser.add_argument("--rho",
            type = float,
            default = 1.0
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adam"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = 0.001
        )
    args = argparser.parse_args()
    print args
    
    all_data = []
    assert args.source_train or args.target_train
    
    if args.source_train:
        train_x, train_y, train_rel = io_util.read_syn_data(args.source_train) 
        all_data += [sent for doc in train_x for sent in doc]
        
    if args.dev:
        dev_x, dev_y, dev_rel = io_util.read_syn_data(args.dev)
        all_data += [sent for doc in dev_x for sent in doc]
        
    if args.target_train:
        t_train_x, t_train_y, t_train_rel = io_util.read_syn_data(args.target_train)
        all_data += [sent for doc in t_train_x for sent in doc]
        
    if args.test:
        test_x, test_y, test_rel = io_util.read_syn_data(args.test)
        all_data += [sent for doc in test_x for sent in doc]
        
    s_ul_x, s_ul_y, s_ul_rel = io_util.read_syn_data(args.source_unlabel)
    t_ul_x, t_ul_y, t_ul_rel = io_util.read_syn_data(args.target_unlabel)
    all_data += [sent for doc in s_ul_x for sent in doc]
    all_data += [sent for doc in t_ul_x for sent in doc]

    x_vocab_dict = io_util.get_vocab_dict(all_data)
    print "Vocab size:", len(x_vocab_dict)
    all_data = []

    embedding_layer = io_util.create_embedding_layer(
                args.embeddings,
                vocab_dict = x_vocab_dict,
                fix_init_embs=False if args.fix_init_embs == 0 else True,
            )

    max_doc_len, max_sent_len = args.max_doc_len, args.max_sent_len
        
    if args.test:
        test_x = [ io_util.map_doc_to_ids(d, embedding_layer, max_sent_len)[:max_doc_len] for d in test_x ]
        test_rel = [ d[:max_doc_len] for d in test_rel]
    
    s_ul_x = [ io_util.map_doc_to_ids(d, embedding_layer, max_sent_len)[:max_doc_len] for d in s_ul_x ]
    s_ul_rel = [ d[:max_doc_len] for d in s_ul_rel]
    
    t_ul_x = [ io_util.map_doc_to_ids(d, embedding_layer, max_sent_len)[:max_doc_len] for d in t_ul_x ]
    t_ul_rel = [ d[:max_doc_len] for d in t_ul_rel]
    
    if args.source_train:
        train_x = [ io_util.map_doc_to_ids(d, embedding_layer, max_sent_len)[:max_doc_len] for d in train_x ]
        train_rel = [ d[:max_doc_len] for d in train_rel]
        
    if args.dev:
        dev_x = [ io_util.map_doc_to_ids(d, embedding_layer, max_sent_len)[:max_doc_len] for d in dev_x ]
        dev_rel = [ d[:max_doc_len] for d in dev_rel]

    if args.target_train:
        t_train_x = [ io_util.map_doc_to_ids(d, embedding_layer, max_sent_len)[:max_doc_len] for d in t_train_x ]
        t_train_rel = [ d[:max_doc_len] for d in t_train_rel]
    
    model = Model(
                args = args,
                embedding_layer = embedding_layer,
                n_class=2,
            )
    model.ready()

    model.train(
            zip(train_x, train_y, train_rel) if args.source_train else None,
            zip(t_train_x, t_train_y, t_train_rel) if args.target_train else None,
            zip(s_ul_x, s_ul_y, s_ul_rel),
            zip(t_ul_x, t_ul_y, t_ul_rel),
            zip(dev_x, dev_y, dev_rel) if dev_x is not None else None,
            zip(test_x, test_y, test_rel) if args.test else None,
        )
