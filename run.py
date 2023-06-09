from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os

import numpy as np
import torch

from torch.utils.data import DataLoader

from models.RelaGraph import KGEModel

from collections import defaultdict
from tqdm import tqdm
import time

from tensorboardX import SummaryWriter

from processors import *
from dataloader import TrainDataset, TestDataset
from dataloader import BidirectionalOneShotIterator
import torch.nn.functional as F

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)

    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_valid', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=True)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--data_name', type=str, default='KMHEO', help='dataset name, default to KMHEO')
    parser.add_argument('--use_RelaGraph', default=True, type=bool)
    parser.add_argument('--SF', default='TripleRE', type=str)
    
    parser.add_argument('-n', '--negative_sample_size', default=32, type=int)
    parser.add_argument('-d', '--hidden_dim', default=64, type=int)
    parser.add_argument('-g', '--gamma', default=6.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=2.0, type=float)
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=2, type=int)
    parser.add_argument('-randomSeed', default=0, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=5000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=100000, type=int)
    parser.add_argument('--valid_steps', default=500, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=500, type=int, help='valid/test log every xx steps')

    parser.add_argument('--print_on_screen', action='store_true', default=True, help='log on screen or not')
    parser.add_argument('--ntriples_eval_train', type=int, default=200000,
                        help='number of training triples to evaluate eventually')
    parser.add_argument('--neg_size_eval_train', type=int, default=500,
                        help='number of negative samples when evaluating training triples')

    parser.add_argument('--true_negative', action='store_true', default=True, help='whether to remove existing triples from negative sampling')
    parser.add_argument('--inverse', action='store_true', help='whether to add inverse edges')
    parser.add_argument('--val_inverse', action='store_true', help='whether to add inverse edges to the validation set')
    parser.add_argument('--drop', type=float, default=0.05, help='Dropout in layers')
    
    parser.add_argument('-u', '--triplere_u', default=1.0, type=float)
    parser.add_argument('--anchor_size', default=0.1, type=float, help='size of the anchor set, i.e. |A|')
    parser.add_argument('-ancs', '--sample_anchors', default=10, type=int)
    parser.add_argument('-path', '--use_anchor_path', default=True)
    parser.add_argument('--sample_neighbors', default=5, type=int)
    parser.add_argument('--max_relation', default=3, type=int)
    parser.add_argument('-center', '--sample_center', default=True)
    parser.add_argument('--node_dim', default=0, type=int)
    parser.add_argument('-merge', '--merge_strategy', default='mean_pooling', type=str,
                        help='how to merge information from anchors, chosen between [ mean_pooling, linear_proj ]')
    parser.add_argument('-layers', '--attn_layers_num', default=1, type=int)
    parser.add_argument('--mlp_ratio', default=2, type=int)
    parser.add_argument('--head_dim', default=8, type=int)
    parser.add_argument('-type', '--add_type_embedding', default=True)
    parser.add_argument('-share', '--anchor_share_embedding', default=True)
    parser.add_argument('-skip', '--anchor_skip_ratio', default=0.2, type=float)


    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.dataset = argparse_dict['dataset']
    args.model = argparse_dict['model']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    if hasattr(model, 'entity_embedding'):
        entity_embedding = model.entity_embedding.weight.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'entity_embedding'),
            entity_embedding
        )
    elif hasattr(model, 'anchor_embedding'):
        anchor_embedding = model.anchor_embedding.weight.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'anchor_embedding'),
            anchor_embedding
        )

    relation_embedding = model.relation_embedding.weight.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )

def train(model, processor, args):
    logging.info('Calculating sample weights...')
    if args.uni_weight:
        triple_weights = None
    else:
        train_count = defaultdict(lambda: 3)
        for h,r,t in processor.train_triples[:,:3]:
            train_count[(h,r)] += 1
            train_count[(t,-r-1)] += 1
        triple_weights = [train_count[(h,r)]+train_count[(t,-r-1)] for h,r,t in processor.train_triples[:,:3]]
        triple_weights = 1 / np.sqrt(np.array(triple_weights))

    logging.info('Creating train dataloader...')
    # Set training dataloader iterator
    train_dataloader_head = DataLoader(
        TrainDataset(processor.train_triples, processor.nentity,
                        args.negative_sample_size, mode='head-batch',
                        filter_negative=args.true_negative,
                        weights=triple_weights,
                        type_offset=processor.type_offset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    train_dataloader_tail = DataLoader(
        TrainDataset(processor.train_triples, processor.nentity,
                        args.negative_sample_size, mode='tail-batch',
                        filter_negative=args.true_negative,
                        weights=triple_weights,
                        type_offset=processor.type_offset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    # Set training configuration
    current_learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=current_learning_rate
    )
    if args.warm_up_steps:
        warm_up_steps = args.warm_up_steps
    else:
        warm_up_steps = args.max_steps // 2
        
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        init_step = 0
            
    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    logging.info('learning_rate = %d' % current_learning_rate)
    training_logs = []
    max_val_mrr = 0
    best_val_metrics = None
    best_test_metrics = None
    best_metrics_step = 0
    print('-----------------------------------------')
    print(model)
    print('-----------------------------------------')
    print(optimizer)
    print('-----------------------------------------')
    # Training Loop
    for step in tqdm(range(init_step, args.max_steps)):

        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator)

        if args.cuda:
            batch = [data.cuda() for data in batch]
            
        head, relation, tail, weight = batch
        score = model(head, relation, tail)
        positive_score, negative_score = score[:,0], score[:,1:]
        '''
        if args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                            * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        '''

        positive_score = F.logsigmoid(positive_score)
        positive_sample_loss = - (weight * positive_score).sum() / weight.sum()
        multicls_score=F.softmax(score,dim=1)[:,0]
        multicls_loss=-(weight * torch.log(multicls_score)).sum() / weight.sum()
        #negative_sample_loss = - (weight * negative_score).sum() / weight.sum()
        #loss = (positive_sample_loss + negative_sample_loss) / 2
        loss=positive_sample_loss+multicls_loss
            
        loss.backward()
        optimizer.step()


        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            #'negative_sample_loss': negative_sample_loss.item(),
            'multi-class_loss': multicls_loss.item(),
            'loss': loss.item()
        }

        training_logs.append(log)

        if step >= warm_up_steps:
            current_learning_rate = current_learning_rate / 10
            logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            for p in optimizer.param_groups:
                p['lr'] = current_learning_rate
            warm_up_steps = warm_up_steps * 3

        if step % args.save_checkpoint_steps == 0 and step > 0:
            save_variable_list = {
                'step': step,
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            }
            save_model(model, optimizer, save_variable_list, args)

        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
            log_metrics('Train', step, metrics)
            training_logs = []

        if args.do_valid and step % args.valid_steps == 0 and step > 0:
            logging.info('Evaluating on Valid Dataset...')
            metrics = test(model, processor, args, 'valid')
            log_metrics('Valid', step, metrics)
            val_mrr = metrics['mrr_list']

            if val_mrr > max_val_mrr:
                max_val_mrr = val_mrr
                best_val_metrics = metrics
                best_metrics_step = step

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    metrics = test(model, processor, args, 'test')
                    log_metrics('Test', step, metrics)
                    best_test_metrics = metrics

    if args.do_valid and best_val_metrics != None:
        log_metrics('Best Val  Metrics', best_metrics_step, best_val_metrics)
    if args.do_test and best_test_metrics != None:
        log_metrics('Best Test Metrics', best_metrics_step, best_test_metrics)

    save_variable_list = {
        'step': step,
        'current_learning_rate': current_learning_rate,
        'warm_up_steps': warm_up_steps
    }
    save_model(model, optimizer, save_variable_list, args)

def test(model, processor, args, dataset):
    assert dataset in ['train', 'valid', 'test']
    if dataset == 'train':
        triples = processor.train_triples
        neg_head, neg_tail = None, None
        neg_size = args.negative_sample_size
    elif dataset == 'valid':
        triples = processor.valid_triples
        neg_head, neg_tail = processor.valid_neg_head, processor.valid_neg_tail
        neg_size = 0
    elif dataset == 'test':
        triples = processor.test_triples
        neg_head, neg_tail = processor.test_neg_head, processor.test_neg_tail
        neg_size = 0
    else:
        raise NotImplementedError

    model.eval()

    test_dataloader_head = DataLoader(
        TestDataset(triples, processor.nentity, 'head-batch', neg_head, neg_size),
        batch_size=args.test_batch_size,
        num_workers=0
    )

    test_dataloader_tail = DataLoader(
        TestDataset(triples, processor.nentity, 'tail-batch', neg_tail, neg_size),
        batch_size=args.test_batch_size,
        num_workers=0
    )

    test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    test_logs = defaultdict(list)

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            if test_dataset.dataset.neg_mode == 'all':
                model.cache_entity_embedding()
            for batch in test_dataset:
                if args.cuda:
                    batch = [data.cuda() for data in batch]
                head, relation, tail = batch

                score = model(head, relation, tail)

                batch_results = processor.evaluate(head, relation, tail, score)
                for metric in batch_results:
                    test_logs[metric].append(batch_results[metric])

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1
            if test_dataset.dataset.neg_mode == 'all':
                model.detach_entity_embedding()

        metrics = {}
        for metric in test_logs:
            metrics[metric] = np.concatenate(test_logs[metric]).mean()

    return metrics

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    logging.info('\n')
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


if __name__ == '__main__':
    args=parse_args()
    
    # one of train/val/test mode must be choosed
    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)

    # 参数保存路径
    args.save_path = 'log/%s/%s/%s-%s/%s' % (
    args.data_name, args.SF, args.hidden_dim, args.gamma, time.time()) if args.save_path == None else args.save_path
    writer = SummaryWriter(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)
    logging.info('Random seed: {}'.format(args.randomSeed))
    torch.manual_seed(args.randomSeed)
    np.random.seed(args.randomSeed)
    processor = DataProcessor(data_name=args.data_name,
                              inverse=args.inverse, val_inverse=args.val_inverse)

    logging.info('Saving: %s' % args.save_path)
    logging.info('Scoring Function: %s' % args.SF)
    logging.info('Dataset: %s' % args.data_name)
    logging.info('#entity: %d' % processor.nentity)
    logging.info('#relation: %d' % processor.nrelation)

    logging.info('#train: %d' % len(processor.train_triples))
    logging.info('#valid: %d' % len(processor.valid_triples))
    logging.info('#test: %d' % len(processor.test_triples))

    model = KGEModel(
        processor=processor,
        args=args,
    )
    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    logging.info(f"Total number of params: {sum(p.numel() for p in model.parameters())}")

    if args.cuda:
        model = model.cuda()


    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.SF)

    if args.do_train:
        train(model, processor, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = test(model, processor, args, 'valid')
        log_metrics('Valid', -1, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = test(model, processor, args, 'test')
        log_metrics('Test', -1, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = test(model, processor, args, 'train')
        log_metrics('Train', -1, metrics)
