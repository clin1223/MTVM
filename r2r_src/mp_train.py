import torch
import torch.multiprocessing as mp
import os
import time
import json
import random
import numpy as np
from collections import defaultdict

from utils import read_vocab, write_vocab, build_vocab, padding_idx, timeSince, read_img_features, print_progress
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args
import model_PREVALENT
from shared_optim import ensure_shared_grads



import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

from vlnbert.vlnbert_init import get_tokenizer

def shared_save(models, shared_optimizers, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        bert, critic = models
        bert_optimizer, critic_optimizer = shared_optimizers
        all_tuple = [("vln_bert", bert, bert_optimizer),
                     ("critic", critic, critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

def train_agent(rank, num_process, update_rank, shared_models, shared_optimizers, feat_dict, n_iters, start_iter, mp_iter):
    with torch.cuda.device(rank+1):
        last_time = time.time()
        print("rank:",str(rank),"\n")

        torch.manual_seed(1 + rank)
        torch.cuda.manual_seed(1 + rank)

        tok_bert = get_tokenizer(args)
        aug_path = args.aug
        feature_size = args.feature_size 
        train_env = R2RBatch(feat_dict, batch_size=args.batchSize, seed=(1+rank), splits=['train'], tokenizer=tok_bert)
        aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize, seed=(1+rank), splits=[aug_path], tokenizer=tok_bert, name='aug')
        listner = Seq2SeqAgent(None, "", tok_bert, args.maxAction)
        scaler = torch.cuda.amp.GradScaler()
        while time.time()-last_time<90:
            continue
        for _ in range(n_iters // 2):

            # Train with GT data
            listner.env = train_env
            listner.train(1, scaler, shared_models, shared_optimizers, feedback=args.feedback)
            while update_rank != rank:
                continue

            ensure_shared_grads(listner.models, shared_models)
            torch.nn.utils.clip_grad_norm(shared_models[0].parameters(), 40.)
            scaler.step(shared_optimizers[0])
            scaler.step(shared_optimizers[1])
            scaler.update()

            update_rank += 1
            if int(update_rank) == num_process:
                update_rank -= num_process

            # Train with Augmented data
            listner.env = aug_env
            listner.train(1, scaler, shared_models, shared_optimizers, feedback=args.feedback)
            while update_rank != rank:
                continue
            ensure_shared_grads(listner.models, shared_models)
            torch.nn.utils.clip_grad_norm(shared_models[0].parameters(), 40.)
            scaler.step(shared_optimizers[0])
            scaler.step(shared_optimizers[1])
            scaler.update()

            update_rank += 1
            if int(update_rank) == num_process:
                update_rank -= num_process           

            mp_iter += 1

def test_agent(shared_models, optimizers, feat_dict, n_iters, start_iter, mp_iter):
    with torch.cuda.device(0):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        tok_bert = get_tokenizer(args)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        # prepare val env
        #val_env_names = ['val_unseen']
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
        from collections import OrderedDict
        val_envs = OrderedDict(
                ((split,
                    (R2RBatch(feat_dict, batch_size=args.TestbatchSize, splits=[split], tokenizer=tok_bert),
                    Evaluation([split], featurized_scans, tok_bert))
                )
                for split in val_env_names
                )
            )

        listner = Seq2SeqAgent(None, "", tok_bert, args.maxAction)
        start = time.time()
        best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", 'update':False}}

        while True:
            while ((mp_iter % 100) != 0) or (mp_iter == 0):
                continue
            loss_str = ""
            # load model
            test_iter = int(mp_iter)
            for model, shared_model in zip(listner.models, shared_models):
                model.load_state_dict(shared_model.state_dict())

            for env_name, (env, evaluator) in val_envs.items():
                listner.env = env
                iters = None 
                listner.test(use_dropout=False, feedback='argmax', iters=iters)
                result = listner.get_results()
                score_summary, _ = evaluator.score(result)
                loss_str += ", %s " % env_name
                for metric,val in score_summary.items():
                    if metric in ['spl']:
                        if env_name in best_val:
                            if val > best_val[env_name]['spl']:
                                best_val[env_name]['spl'] = val
                                best_val[env_name]['update'] = True
                            elif (val == best_val[env_name]['spl']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
                                best_val[env_name]['spl'] = val
                                best_val[env_name]['update'] = True
                    loss_str += ', %s: %.3f' % (metric, val)

            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = 'Iter %d %s' % (test_iter, loss_str)
                    best_val[env_name]['update'] = False
                    shared_save(listner.models, optimizers, test_iter, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

            print(('%s (%d %d%%) %s' % (timeSince(start, float(test_iter)/n_iters), test_iter, float(test_iter)/n_iters*100, loss_str)))

            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])
            #time.sleep(100)

        