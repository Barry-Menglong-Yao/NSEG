import torch
import numpy as np
from torchtext import data
import logging
import random
import argparse

from data.data import DocField, DocDataset, DocIter, GraphField
import time
import re
from model.seq2seq import train, decode
from pathlib import Path
import json
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer / FastTransformer.')
    parser.add_argument('--gpu', type=str,default='0,1,2,3', help='gpu')
    # dataset settings
    parser.add_argument('--corpus', type=str, nargs='+',default=['/home/barry/workspace/code/referredModels/NSEG/nips/train.lower','/home/barry/workspace/code/referredModels/NSEG/nips/train.eg'])
    parser.add_argument('--lang', type=str, nargs='+', help="the suffix of the corpus, translation language")
    parser.add_argument('--valid', type=str, nargs='+',default=['/home/barry/workspace/code/referredModels/NSEG/nips/val.lower', '/home/barry/workspace/code/referredModels/NSEG/nips/val.eg'])

    parser.add_argument('--writetrans', type=str,default='decoding/gdp0.5_gl2.devorder', help='write translations for to a file')
    parser.add_argument('--ref', type=str, help='references, word unit')

    parser.add_argument('--vocab', type=str,default='/home/barry/workspace/code/referredModels/NSEG/nips/vocab.new.100d.lower.pt')
    parser.add_argument('--vocab_size', type=int, default=40000)

    parser.add_argument('--load_vocab', action='store_true', help='load a pre-computed vocabulary')
    # parser.add_argument('--load_corpus', action='store_true', default=False, help='load a pre-processed corpus')
    # parser.add_argument('--save_corpus', action='store_true', default=False, help='save a pre-processed corpus')
    parser.add_argument('--max_len', type=int, default=None, help='limit the train set sentences to this many tokens')
    # parser.add_argument('--max_train_data', type=int, default=None,
    #                     help='limit the train set sentences to this many sentences')
    parser.add_argument('--pool', type=int, default=100, help='shuffle batches in the pool')

    # model name
    parser.add_argument('--model', type=str, default='gdp0.5_gl2', help='prefix to denote the model, nothing or [time]')

    # network settings
    parser.add_argument('--share_embed', action='store_true', default=False,
                        help='share embeddings and linear out weight')
    parser.add_argument('--share_vocab', action='store_true', default=False,
                        help='share vocabulary between src and target')

    # parser.add_argument('--ffw_block', type=str, default="residual", choices=['residual', 'highway', 'nonresidual'])

    # parser.add_argument('--posi_kv', action='store_true', default=False,
    #                     help='incorporate positional information in key/value')

    parser.add_argument('--params', type=str, default='user', choices=['user', 'small', 'middle', 'big'],
                        help='Defines the dimension size of the parameter')
    parser.add_argument('--n_layers', type=int, default=5, help='number of layers')
    parser.add_argument('--n_heads', type=int, default=2, help='number of heads')

    parser.add_argument('--d_emb', type=int, default=100, help='dimention size for hidden states')
    parser.add_argument('--d_rnn', type=int, default=300, help='dimention size for FFN')
    parser.add_argument('--d_mlp', type=int, default=300, help='dimention size for FFN')
    parser.add_argument('--senenc', default='bow', help='sentence encoder')

    parser.add_argument('--gnnl', default=2, type=int, help='stacked layer number')
    parser.add_argument('--gnndp', default=0.5, type=float, help='self-att dropout')
    
    parser.add_argument('--labeldim', default=50, type=int, help='label dim')
    parser.add_argument('--agg',default='gate', choices=['gate', 'att'], help='node agg method')

    parser.add_argument('--reglamb', default=0, type=float)
    parser.add_argument('--loss', default=0, type=int)

    parser.add_argument('--entityemb', default='glove',choices=['glove', 'lstm'])
    parser.add_argument('--ehid', default=150, type=int)

    parser.add_argument('--initnn', default='standard', help='parameter init')
    parser.add_argument('--early_stop', type=int, default=5)

    # running setting
    parser.add_argument('--mode', type=str, default='hyper_search',
                        choices=['train', 'test',
                                 'distill','hyper_search'])  # distill : take a trained AR model and decode a training set
    parser.add_argument('--seed', type=int, default=1234, help='seed for randomness')

    parser.add_argument('--keep_cpts', type=int, default=1, help='save n checkpoints, when 1 save best model only')

    # training
    # parser.add_argument('--tqdm', action="store_true", default=False) #???
    parser.add_argument('--eval_every', type=int, default=100, help='validate every * step')
    parser.add_argument('--save_every', type=int, default=50, help='save model every * step (5000)')

    parser.add_argument('--batch_size', type=int, default=16, help='# of tokens processed per batch')
    parser.add_argument('--delay', type=int, default=1, help='gradiant accumulation for delayed update for large batch')

    parser.add_argument('--optimizer', type=str, default='Noam')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    # parser.add_argument('--lr_schedule', type=str, default='transformer', choices=['transformer', 'anneal', 'fixed'])
    parser.add_argument('--warmup', type=int, default=4000, help='maximum steps to linearly anneal the learning rate')

    # lr decay
    parser.add_argument('--lrdecay', type=float, default=0, help='learning rate decay')
    parser.add_argument('--patience', type=int, default=0, help='learning rate decay 0.5')

    # parser.add_argument('--anneal_steps', type=int, default=250000,
    #                     help='maximum steps to linearly anneal the learning rate')
    parser.add_argument('--maximum_steps', type=int, default=100, help='maximum steps you take to train a model')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--input_drop_ratio', type=float, default=0.5, help='dropout ratio only for inputs')

    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping')
    parser.add_argument('--smoothing', type=float, default=0.0, help='label smoothing')

    # decoding
    parser.add_argument('--length_ratio', type=float, default=2, help='maximum lengths of decoding')
    parser.add_argument('--beam_size', type=int, default=64,
                        help='beam-size used in Beamsearch, default using greedy decoding')
    parser.add_argument('--alpha', type=float, default=0.6, help='length normalization weights')
    # parser.add_argument('--T', type=float, default=1, help='softmax temperature when decoding')

    parser.add_argument('--test', type=str, nargs='+', help='test src file',default=['/home/barry/workspace/code/referredModels/NSEG/nips/test.lower', 
    '/home/barry/workspace/code/referredModels/NSEG/nips/test.eg'])

    # model saving/reloading, output translations
    parser.add_argument('--load_from', nargs='+', default=None, help='load from 1.modelname, 2.lastnumber, 3.number')

    parser.add_argument('--resume', action='store_true',
                        help='when resume, need other things besides parameters')
    # save path
    parser.add_argument('--main_path', type=str, default="./")
    parser.add_argument('--model_path', type=str, default="models")
    parser.add_argument('--decoding_path', type=str, default="decoding")


    file='/home/barry/workspace/code/referredModels/NSEG/data/hamming_perms_4_patches_4_max_avg.npy'
 
    temp = re.findall(r'\d+', file)
    digits = list(map(int, temp))
    parser.add_argument("--permutation_number", type=int, default=digits[0], help="Number of  permuations")
    parser.add_argument("--permutation_length", type=int, default=digits[1], help="Number of sentences in permuations")
    parser.add_argument("--permutation_file", type=str, default=file, help=" permuations file")

    
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def curtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def override(args, load_dict, except_name):
    for k in args.__dict__:
        if k not in except_name:
            args.__dict__[k] = load_dict[k]


'''
You can call `torch.load(.., map_location='cpu')`
and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint
'''
# def extract_permutation_args(args):
    



def train_with_hyper_search(args,num_samples=10,  gpus_per_trial=1):
    max_num_epochs=args.maximum_steps
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    # config = {
    #     "lr": tune.loguniform(1e-1, 1e-1),
    #     "batch_size": tune.choice([ 16])
    # }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_one_trail, args=args),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
 
    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # test_acc = test(args,model_state)
    # print("Best trial test set accuracy: {}".format(test_acc))

def updateHyper(config,args):
    if args!=None and config!=None:
        args.__dict__['lr']=config['lr']
        args.__dict__['batch_size']=config['batch_size']


def train_one_trail(config, checkpoint_dir=None,args=None):
    updateHyper(config,args)
    if args.load_from is not None and len(args.load_from) == 1:
            load_from = args.load_from[0]
            print('{} load the checkpoint from {} for initilize or resume'.
                    format(curtime(), load_from))
            checkpoint = torch.load(load_from, map_location='cpu')
    else:
        checkpoint = None

    # if not resume(initilize), only need model parameters
    if args.resume:
        print('update args from checkpoint')
        load_dict = checkpoint['args'].__dict__
        except_name = ['mode', 'resume', 'maximum_steps']
        override(args, load_dict, tuple(except_name))

    main_path = Path(args.main_path)
    model_path = main_path / args.model_path
    decoding_path = main_path / args.decoding_path

    for path in [model_path, decoding_path]:
        path.mkdir(parents=True, exist_ok=True)

    args.model_path = str(model_path)
    args.decoding_path = str(decoding_path)

    if args.model == '[time]':
        args.model = time.strftime("%m.%d_%H.%M.", time.gmtime())

    # setup random seeds
    set_seeds(args.seed)

    # special process, shuffle each document
    # DOC = DocField(batch_first=True, include_lengths=True, eos_token='<eos>', init_token='<bos>')
    DOC = DocField(batch_first=True, include_lengths=True)
    ORDER = data.Field(batch_first=True, include_lengths=True, pad_token=0, use_vocab=False,
                        sequential=True)

    GRAPH = GraphField(batch_first=True)



    train_data = DocDataset(path=args.corpus, text_field=DOC, order_field=ORDER, 
    graph_field=GRAPH,permutation_length=args.permutation_length,permutation_number=args.permutation_number)

    dev_data = DocDataset(path=args.valid, text_field=DOC, order_field=ORDER, 
    graph_field=GRAPH,permutation_length=args.permutation_length,permutation_number=args.permutation_number)

    DOC.vocab = torch.load(args.vocab)
    print('vocab {} loaded'.format(args.vocab))
    args.__dict__.update({'doc_vocab': len(DOC.vocab)})

    train_flag = True
    train_real = DocIter(train_data, args.batch_size, args.permutation_file,args.permutation_length,device='cuda',
                            train=train_flag,
                            shuffle=train_flag,
                            sort_key=lambda x: len(x.doc))

    devbatch = 1
    dev_real = DocIter(dev_data, devbatch, args.permutation_file,args.permutation_length,device='cuda', batch_size_fn=None,
                        train=False, repeat=False, shuffle=False, sort=False)

    args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
    print(args_str)

    print('{} Start training'.format(curtime()))
    acc,loss=train(args, train_real, dev_real, (DOC, ORDER, GRAPH), checkpoint,args.permutation_file,args.permutation_length)
    if args.mode=='hyper_search':
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss, accuracy=acc)

def test(args,model_state_dir):
    print('{} Load test set'.format(curtime()))

    DOC = DocField(batch_first=True, include_lengths=True)
    ORDER = data.Field(batch_first=True, include_lengths=True, pad_token=0, use_vocab=False,
                        sequential=True)
    GRAPH = GraphField(batch_first=True)

    DOC.vocab = torch.load(args.vocab)
    print('vocab {} loaded'.format(args.vocab))
    args.__dict__.update({'doc_vocab': len(DOC.vocab)})

    args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
    print(args_str)

    test_data = DocDataset(path=args.test, text_field=DOC, order_field=ORDER, graph_field=GRAPH,
    permutation_length=args.permutation_length,permutation_number=args.permutation_number)
    test_real = DocIter(test_data, 1,args.permutation_file,args.permutation_length, device='cuda', batch_size_fn=None,
                        train=False, repeat=False, shuffle=False, sort=False)

    print('{} Load data done'.format(curtime()))
    start = time.time()
    acc=decode(args, test_real, (DOC, ORDER), model_state_dir)
    print('{} Decode done, time {} mins'.format(curtime(), (time.time() - start) / 60))
    return acc

if __name__ == '__main__':
    args = parse_args()
    # extract_permutation_args(args)
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    if args.mode == 'train':
        train_one_trail(None,args =args)
    elif args.mode=='hyper_search':
        train_with_hyper_search(args,num_samples=10)
    else:
        if len(args.load_from) == 1:
            load_from = '{}.best.pt'.format(args.load_from[0])
            print('{} load the best checkpoint from {}'.format(curtime(), load_from))
            checkpoint = torch.load(load_from, map_location='cpu')
        else:
            raise RuntimeError('must load model')

        # when translate load_dict update args except some
        print('update args from checkpoint')
        load_dict = checkpoint['args'].__dict__
        except_name = ['mode', 'load_from', 'test', 'writetrans', 'beam_size', 'batch_size']
        override(args, load_dict, tuple(except_name))

        test(args,checkpoint['model'])
    


