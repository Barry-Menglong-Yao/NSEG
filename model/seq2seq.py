import re
import torch
import numpy as np
import math
import torch.nn as nn
import time
import subprocess
import torch.nn.functional as F
from model.generator import Beam
from data.data import DocField, DocDataset, DocIter
from util.config import is_permutation_task

from model.model import * 

def beam_search_pointer(args, model, src_and_len, doc_num, ewords_and_len, elocs):
    sentences, _, dec_init, keys = model.encode(src_and_len, doc_num,  ewords_and_len, elocs)

    document = sentences.squeeze(0)
    T, H = document.size()

    W = args.beam_size

    prev_beam = Beam(W)
    prev_beam.candidates = [[]]
    prev_beam.scores = [0]

    target_t = T - 1

    f_done = (lambda x: len(x) == target_t)

    valid_size = W
    hyp_list = []

    for t in range(target_t):
        candidates = prev_beam.candidates
        if t == 0:
            # start
            dec_input = sentences.new_zeros(1, 1, H)
            pointed_mask = sentences.new_zeros(1, T).byte()
        else:
            index = sentences.new_tensor(list(map(lambda cand: cand[-1], candidates))).long()
            # beam 1 H
            dec_input = document[index].unsqueeze(1)

            pointed_mask[torch.arange(index.size(0)), index] = 1

        dec_h, dec_c, log_prob = model.step(dec_input, dec_init, keys, pointed_mask)

        next_beam = Beam(valid_size)
        done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
        hyp_list.extend(done_list)
        valid_size -= len(done_list)

        if valid_size == 0:
            break

        beam_remain_ix = src_and_len[0].new_tensor(remain_list)
        dec_h = dec_h.index_select(1, beam_remain_ix)
        dec_c = dec_c.index_select(1, beam_remain_ix)
        dec_init = (dec_h, dec_c)

        pointed_mask = pointed_mask.index_select(0, beam_remain_ix)

        prev_beam = next_beam

    score = dec_h.new_tensor([hyp[1] for hyp in hyp_list])
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix][0], score[ix].item()))
    best_output = output[0][0]

    the_last = list(set(list(range(T))).difference(set(best_output)))
    best_output.append(the_last[0])

    return best_output




def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))



def train(args, train_iter, dev, fields, checkpoint,permutation_file,permutation_length):
    if is_permutation_task():
        model=PermutationPredictor(args)
    else:
        model = PointerNet(args)
    model.cuda()

    DOC, ORDER, GRAPH = fields
    print('1:', DOC.vocab.itos[1])
    model.load_pretrained_emb(DOC.vocab.vectors)

    print_params(model)
    print(model)

    
    wd = 1e-5
    opt = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, weight_decay=wd)

    best_score = -np.inf
    best_iter = 0
    offset = 0

    criterion = nn.NLLLoss(reduction='none')
    model.equip(criterion)

    start = time.time()

    early_stop = args.early_stop

    test_data = DocDataset(path=args.test, text_field=DOC, order_field=ORDER, graph_field=GRAPH,
    permutation_length=permutation_length,permutation_number=args.permutation_number)
    test_real = DocIter(test_data, 1,  permutation_file, permutation_length,device='cuda', batch_size_fn=None,
                        train=False, repeat=False, shuffle=False, sort=False)
    
    history={"acc":[],"loss":[],"val_acc":[], "epoch":[]}
    val_loss = 0.0
    val_steps = 0
    for epc in range(args.maximum_steps):
        
        for iters, batch in enumerate(train_iter):
            model.train()

            model.zero_grad()

            t1 = time.time()

            loss = model(batch.doc, batch.order, batch.doc_len, batch.e_words, batch.elocs,batch.permutation_lables)
           
            loss.backward()
            opt.step()

            t2 = time.time()
            print('epc:{} iter:{} loss:{:.2f} t:{:.2f} lr:{:.1e}'.format(epc, iters + 1, loss, t2 - t1,
                                                                         opt.param_groups[0]['lr']))
            val_loss += loss.detach().cpu().numpy()
            val_steps += 1
        if epc < 5:
            continue

        with torch.no_grad():
            print('valid..............')
            if args.loss:
                score = valid_model(args, model, dev, DOC, 'loss')
                print('epc:{}, loss:{:.2f} best:{:.2f}\n'.format(epc, score, best_score))
            else:
                score, pmr, ktau, _ = valid_model(args, model, dev, DOC)
                history["epoch"].append(epc)
                history["acc"].append(score)
                print('epc:{}, val acc:{:.4f} best:{:.4f} pmr:{:.2f} ktau:{:.4f}'.format(epc, score, best_score,
                                                                                              pmr, ktau))

            if score > best_score:
                best_score = score
                best_iter = epc

                print('save best model at epc={}'.format(epc))
                checkpoint = {'model': model.state_dict(),
                              'args': args,
                              'loss': best_score}
                torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))

            if early_stop and (epc - best_iter) >= early_stop:
                print('early stop at epc {}'.format(epc))
                break

    print('\n*******Train Done********{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    minutes = (time.time() - start) // 60
    if minutes < 60:
        print('best:{:.2f}, iter:{}, time:{} mins, lr:{:.1e}, '.format(best_score, best_iter, minutes,
                                                                       opt.param_groups[0]['lr']))
    else:
        hours = minutes / 60
        print('best:{:.2f}, iter:{}, time:{:.1f} hours, lr:{:.1e}, '.format(best_score, best_iter, hours,
                                                                            opt.param_groups[0]['lr']))

    checkpoint = torch.load('{}/{}.best.pt'.format(args.model_path, args.model), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        acc, pmr, ktau, pm = valid_model(args, model, test_real, DOC, shuflle_times=1)
        print('test acc:{:.4%} pmr:{:.2%} ktau:{:.4f} pm:{:.2%}'.format(acc, pmr, ktau, pm))

    loss=val_loss/val_steps
    return acc,loss


def valid_model(args, model, dev, field, dev_metrics=None, shuflle_times=1):
    model.eval()

    if dev_metrics == 'loss':
        total_score = []
        number = 0

        for iters, dev_batch in enumerate(dev):
            loss = model(dev_batch.doc, dev_batch.order, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs)
            n = dev_batch.order[0].size(0)
            batch_loss = -loss.item() * n
            total_score.append(batch_loss)
            number += n

        return sum(total_score) / number
    else:
        if is_permutation_task():
            return valid_permutation_model(args, model, dev, field, dev_metrics , shuflle_times)
        else:
            return valid_sentence_ordering_model(args, model, dev, field, dev_metrics , shuflle_times)


def valid_permutation_model(args, model, dev, field, dev_metrics , shuflle_times):
    f = open(args.writetrans, 'w')
    best_acc = []
    for epc in range(shuflle_times):
        truth = []
        predicted = []

        for j, dev_batch in enumerate(dev):
            tru = dev_batch.permutation_lables[0]
            truth.append(int(tru.cpu().detach() ) )
            pred = model.predict(dev_batch.doc, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs)
            predicted.append(int(pred.cpu().detach() ))
        right, total = 0, 0
        pmr_right = 0
        taus = []
        # pm
        pm_p, pm_r = [], []
        import itertools

        from sklearn.metrics import accuracy_score

        for t, p in zip(truth, predicted):
       

            eq = np.equal(t , p )
            right += eq.sum()
            total += 1

            pmr_right += eq.all()

            # pm
            # s_t = set([i for i in itertools.combinations(t, 2)])
            # s_p = set([i for i in itertools.combinations(p, 2)])
            # pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
            # pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

            # cn_2 = len(p) * (len(p) - 1) / 2
            # pairs = len(s_p) - len(s_p.intersection(s_t))
            # tau = 1 - 2 * pairs / cn_2
            # taus.append(tau)

        # acc = right / total

        acc = accuracy_score( truth ,
                                 predicted   )

        best_acc.append(acc)

        # pmr = pmr_right / len(truth)
        # taus = np.mean(taus)

        # pm_p = np.mean(pm_p)
        # pm_r = np.mean(pm_r)
        # pm = 2 * pm_p * pm_r / (pm_p + pm_r)

        print('acc:', acc)

    f.close()
    acc = max(best_acc)
    return acc ,0,0,0

def valid_sentence_ordering_model(args, model, dev, field, dev_metrics , shuflle_times):
    f = open(args.writetrans, 'w')

    if args.beam_size != 1:
        print('beam search with beam', args.beam_size)

    best_acc = []
    for epc in range(shuflle_times):
        truth = []
        predicted = []

        for j, dev_batch in enumerate(dev):
            tru = dev_batch.order[0].view(-1).tolist()
            truth.append(tru)
    
            if len(tru) == 1:
                pred = tru
            else:
                pred = beam_search_pointer(args, model, dev_batch.doc, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs)
            print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
                file=f)
            predicted.append(pred)
        right, total = 0, 0
        pmr_right = 0
        taus = []
        # pm
        pm_p, pm_r = [], []
        import itertools

        from sklearn.metrics import accuracy_score

        for t, p in zip(truth, predicted):
            if len(p) == 1:
                right += 1
                total += 1
                pmr_right += 1
                taus.append(1)
                continue

            eq = np.equal(t, p)
            right += eq.sum()
            total += len(t)

            pmr_right += eq.all()

            # pm
            s_t = set([i for i in itertools.combinations(t, 2)])
            s_p = set([i for i in itertools.combinations(p, 2)])
            pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
            pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

            cn_2 = len(p) * (len(p) - 1) / 2
            pairs = len(s_p) - len(s_p.intersection(s_t))
            tau = 1 - 2 * pairs / cn_2
            taus.append(tau)

        # acc = right / total

        acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                                list(itertools.chain.from_iterable(predicted)))

        best_acc.append(acc)

        pmr = pmr_right / len(truth)
        taus = np.mean(taus)

        pm_p = np.mean(pm_p)
        pm_r = np.mean(pm_r)
        pm = 2 * pm_p * pm_r / (pm_p + pm_r)

        print('acc:', acc)

    f.close()
    acc = max(best_acc)
    return acc, pmr, taus, pm

def decode(args, test_real, fields, model_state_dir):
    with torch.no_grad():
        model = PermutationPredictor(args)
        model.cuda()

        print('load parameters')
        model.load_state_dict(model_state_dir)
        DOC, ORDER = fields
        acc, pmr, ktau, _ = valid_model(args, model, test_real, DOC)
        print('test acc:{:.2%} pmr:{:.2%} ktau:{:.2f}'.format(acc, pmr, ktau))
    return acc


