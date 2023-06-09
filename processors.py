from functools import partial
import numpy as np
import torch
from collections import defaultdict
import os

class Processor:
    def __init__(self, data_name,
                 filter_unknown=True, filter_true=True, inverse=False, val_inverse=False) -> None:
        # defined as numpy array
        self.data_name=data_name
        self.data_dir = 'datasets/'+self.data_name
        self.train_triples = None
        self.valid_triples, self.valid_neg_head, self.valid_neg_tail = None, None, None
        self.test_triples, self.test_neg_head, self.test_neg_tail = None, None, None
        # either dict or list, visited by key or index
        self.e2i, self.i2e, self.r2i, self.i2r = {}, [], {}, []
        self.nentity = 0
        self.nrelation = 0
        self.type_offset = None
        # self.all_as_negative = True
        self._init()

        self._nr = self.nrelation
        self._true_triples = defaultdict(list)
        if filter_true:
            for h,r,t in self.train_triples[:,:3]:
                self._true_triples[(h,r)].append(t)
                self._true_triples[(t, r + self._nr)].append(h)
            for h,r,t in self.valid_triples[:,:3]:
                self._true_triples[(h,r)].append(t)
                self._true_triples[(t, r + self._nr)].append(h)
            for h,r,t in self.test_triples[:,:3]:
                self._true_triples[(h,r)].append(t)
                self._true_triples[(t, r + self._nr)].append(h)
        if filter_unknown:
            self.filter_unknown()
        assert inverse or not val_inverse, 'Should not set val_inverse without setting inverse.'
        if inverse or val_inverse:
            self.r2i.update({'inversed-'+r: self.r2i[r]+self._nr for r in self.r2i})
            self.i2r.extend(['inversed-'+self.i2r[i] for i in range(len(self.i2r))])
            self.nrelation *= 2
            self.set_inverse(inverse, val_inverse)

    def _init(self):
        raise NotImplementedError
    
    def filter_unknown(self):
        if self.train_triples is None:
            return
        unknow_entities = set(i for i in range(self.nentity)) - set(self.train_triples[:,0].tolist() + self.train_triples[:,2].tolist())
        unknow_relations = set(i for i in range(self.nrelation)) - set(self.train_triples[:,1])
        if self.valid_triples is not None:
            keep = [h not in unknow_entities and t not in unknow_entities and r not in unknow_relations for h,r,t in self.valid_triples[:,:3]]
            keep = np.array(keep)
            self.valid_triples = self.valid_triples[keep]
            if self.valid_neg_head is not None:
                self.valid_neg_head = self.valid_neg_head[keep]
            if self.valid_neg_tail is not None:
                self.valid_neg_tail = self.valid_neg_tail[keep]
        if self.test_triples is not None:
            keep = [h not in unknow_entities and t not in unknow_entities and r not in unknow_relations for h,r,t in self.test_triples[:,:3]]
            keep = np.array(keep)
            self.test_triples = self.test_triples[keep]
            if self.test_neg_head is not None:
                self.test_neg_head = self.test_neg_head[keep]
            if self.test_neg_tail is not None:
                self.test_neg_tail = self.test_neg_tail[keep]
    
    def set_inverse(self, inverse, val_inverse):
        if inverse and self.train_triples is not None:
            inv_triples = np.concatenate([self.train_triples[:,2::-1], self.train_triples[:,4:2:-1]], axis=1).copy()
            inv_triples[:,1] += self._nr
            self.train_triples = np.concatenate([self.train_triples, inv_triples])
        if not val_inverse:
            return
        if self.valid_triples is not None:
            inv_triples = np.concatenate([self.valid_triples[:,2::-1], self.valid_triples[:,4:2:-1]], axis=1).copy()
            inv_triples[:,1] += self._nr
            self.valid_triples = np.concatenate([self.valid_triples, inv_triples])
            if self.valid_neg_head is not None and self.valid_neg_tail is not None:
                self.valid_neg_head = np.concatenate([self.valid_neg_head, self.valid_neg_tail])
                self.valid_neg_tail = np.concatenate([self.valid_neg_tail, self.valid_neg_head])
            elif self.valid_neg_head is not None or self.valid_neg_tail is not None:
                raise NotImplementedError
        if self.test_triples is not None:
            inv_triples = np.concatenate([self.test_triples[:,2::-1], self.test_triples[:,4:2:-1]], axis=1).copy()
            inv_triples[:,1] += self._nr
            self.test_triples = np.concatenate([self.test_triples, inv_triples])
            if self.test_neg_head is not None and self.test_neg_tail is not None:
                self.test_neg_head = np.concatenate([self.test_neg_head, self.test_neg_tail])
                self.test_neg_tail = np.concatenate([self.test_neg_tail, self.test_neg_head])
            elif self.test_neg_head is not None or self.test_neg_tail is not None:
                raise NotImplementedError
    
    def get_original_train_triples(self):
        if self._nr == self.nrelation:
            return self.train_triples
        ntriples = self.train_triples.shape[0]
        assert ntriples % 2 == 0
        return self.train_triples[:ntriples//2]
    
    def evaluate(self, head, relation, tail, score):
        if type(score) is torch.Tensor:
            head = head.cpu().numpy()
            relation = relation.cpu().numpy()
            tail = tail.cpu().numpy()
            score = score.cpu().numpy()
        elif type(score) is not np.ndarray:
            raise TypeError('The score should be of type torch.Tensor or numpy.ndarray')
        head = head.squeeze()
        relation = relation.squeeze()
        tail = tail.squeeze()
        score = score.squeeze()
        assert len(score.shape) == 2 and score.shape[-1] > 1

        if self._true_triples is not None:
            if len(head.shape) > len(tail.shape): # head-batch
                head, relation, tail = tail, relation, head
                mask = relation < self._nr
                relation[ mask] += self._nr
                relation[~mask] -= self._nr
            elif len(head.shape) < len(tail.shape): # tail-batch
                pass
            else:
                raise ValueError('Something wrong, neither head-batch nor tail-batch.')
            if score.shape[-1] == self.nentity + 1: # the negative samples is all the entities
                for i in range(len(head)):
                    true_tails = self._true_triples[(head[i], relation[i])]
                    score[i,1:][true_tails] -= 99999
            else:
                for i in range(len(head)):
                    true_tails = set(self._true_triples[(head[i], relation[i])])
                    for j in range(1, score.shape[-1]):
                        if tail[i,j] in true_tails:
                            score[i,j] -= 99999
        # the first column is the positive samples, the larger score the better
        return self._eval(score)

    def _eval(self, score) -> dict:
        # put the positive sample at the bottom of equal scores
        ranking = (score >= score[:,0:1]).sum(axis=-1) # 0:1 is necessary to preserve the shape
        batch_results = {
            'mrr_list': 1.0/ranking,
            'mr_list': ranking,
            'hits@1_list': ranking <= 1,
            'hits@3_list': ranking <= 3,
            'hits@10_list': ranking <= 10,
        }
        return batch_results
    
    def entities_to_ids(self, keys):
        if not self.e2i:
            raise NotImplementedError
        if type(keys) is not list:
            return self.e2i[keys]
        return [self.e2i[k] for k in keys]

    def ids_to_entities(self, keys):
        if not self.i2e:
            raise NotImplementedError
        if type(keys) is not list:
            return self.i2e[keys]
        return [self.i2e[k] for k in keys]

    def relations_to_ids(self, keys):
        if not self.r2i:
            raise NotImplementedError
        if type(keys) is not list:
            return self.r2i[keys]
        return [self.r2i[k] for k in keys]

    def ids_to_relations(self, keys):
        if not self.i2r:
            raise NotImplementedError
        if type(keys) is not list:
            return self.i2r[keys]
        return [self.i2r[k] for k in keys]


class DataProcessor(Processor):
    def _init(self) -> None:
        with open(os.path.join(self.data_dir, 'train.txt'), 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]
        entities = sorted({line[0] for line in lines} | {line[2] for line in lines})
        relations = sorted({line[1] for line in lines})
        self.e2i = {e:i for i,e in enumerate(entities)}
        self.r2i = {r:i for i,r in enumerate(relations)}
        
        self.nentity = len(self.e2i)
        self.nrelation = len(self.r2i)
        
        def read_triple(file_path):
            triples = []
            with open(file_path, 'r') as f:
                for line in f:
                    h, r, t = line.strip().split('\t')
                    if h not in self.e2i or t not in self.e2i or r not in self.r2i:
                        continue
                    triples.append([self.e2i[h], self.r2i[r], self.e2i[t]])
            return np.array(triples, dtype=np.int64)

        self.train_triples = read_triple(os.path.join(self.data_dir, 'train.txt'))
        self.valid_triples = read_triple(os.path.join(self.data_dir, 'valid.txt'))
        self.test_triples = read_triple(os.path.join(self.data_dir, 'test.txt'))


