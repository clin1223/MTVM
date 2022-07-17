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
from mp_train import train_agent, test_agent
import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter
from shared_optim import SharedAdamW
from vlnbert.vlnbert_init import get_tokenizer
import sys

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features == 'places365':
    features = PLACE365_FEATURES

feedback_method = args.feedback  # teacher or sample
args.adj_data_path = 'data/'

print(args); print('')


''' train the listener '''
def train(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    start_iter = 0
    if args.load is not None:
        if args.aug is None:
            start_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))
        else:
            load_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, load_iter))

    start = time.time()
    print('\nListener training starts, start iteration: %s' % str(start_iter))

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", 'update':False}}

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=feedback_method)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method)

                # Train with Augmented data
                listner.env = aug_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method)

                #print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        # print("total_actions", total, ", max_length", length)

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                if metric in ['spl']:
                    writer.add_scalar("spl/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['spl']:
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                        elif (val == best_val[env_name]['spl']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.4f' % (metric, val)

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
            else:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "latest_dict"))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)

def train_val(test_only=False):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    tok = get_tokenizer(args)

    feat_dict = read_img_features(features, test_only=test_only)

    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    if args.submit:
        val_env_names.append('test')
    else:
        pass

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        valid(train_env, tok, val_envs=val_envs)
    else:
        assert False

def train_val_augment(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()
    feature_size = args.feature_size
    start_iter = 0
    n_iters = args.iters
    # Load the env img features
    feat_dict = read_img_features(features, test_only=test_only)
    with torch.cuda.device(0):
        if args.vlnbert == 'prevalent':
            vln_bert = model_PREVALENT.VLNBERT(feature_size=feature_size + args.angle_feat_size).cuda()
            critic = model_PREVALENT.Critic().cuda()
    vln_bert.share_memory()
    critic.share_memory()
    shared_models = (vln_bert, critic)

    mp_iter = torch.zeros(1)
    mp_iter.share_memory_()
    update_rank = torch.zeros(1)
    update_rank.share_memory_()
    
    vln_bert_optimizer = SharedAdamW(vln_bert.parameters(), lr=args.lr)
    critic_optimizer = SharedAdamW(critic.parameters(), lr=args.lr)
    vln_bert_optimizer.share_memory()
    critic_optimizer.share_memory()
    shared_optimizers = (vln_bert_optimizer, critic_optimizer)

    mp.set_start_method('spawn', force=True)
    num_processes = args.num_process

    processes = []
    p = mp.Process(target=test_agent, args=(shared_models, shared_optimizers, feat_dict,
                                      n_iters*num_processes, start_iter, mp_iter))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, num_processes):
        last_time = time.time()
        p = mp.Process(
            target=train_agent, args=(rank, num_processes, update_rank, shared_models, shared_optimizers, feat_dict,
                                      n_iters, start_iter, mp_iter))
        p.start()
        processes.append(p)
        time.sleep(0.1)
        while time.time()-last_time<20:
            continue
    for p in processes:
        time.sleep(0.1)
        p.join()

if __name__ == "__main__":
    if args.train in ['listener', 'validlistener']:
        train_val(test_only=args.test_only)
    elif args.train == 'auglistener':
        train_val_augment(test_only=args.test_only)
    else:
        assert False
