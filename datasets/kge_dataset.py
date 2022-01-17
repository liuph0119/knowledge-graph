import random
import numpy  as np
import torch
from torch.utils import data
from torch.utils.data import Dataset


class KGETrainDataset(Dataset):
    def __init__(self, triples, num_entities, num_relations, mode, negative_sample_size):
        self.triples = triples
        self.triple_set = set(triples)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.mode = mode
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_tail(triples)
        if self.mode not in ("head-batch", "tail-batch"):
            raise ValueError(f"`mode` {mode} is not valid, should be one of ('head-batch', 'tail-batch')")

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        
        negative_samples = []
        negative_sample_size= 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = random.randint(self.num_entities, size=self.negative_sample_size * 2)

            # judge where the negative sample is not in the true pairs
            if self.mode == "head-batch":
                mask = np.in1d(negative_sample, self.true_head[(relation, tail)], assume_unique=True, invert=True)
            elif self.mode == "tail-batch":
                mask = np.in1d(negative_sample, self.true_tail[(head, relation)], assume_unique=True, invert=True)

            negative_sample = negative_sample[mask]
            negative_samples.append(negative_sample)
            negative_sample_size += len(negative_sample)
        negative_samples = np.concatenate(negative_samples)[: self.negative_sample_size]

        subsampling_weight = self.count[(head, tail)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(torch.tensor([1 / subsampling_weight]))

        positive_sample = torch.LongTensor(positive_sample)
        negative_samples = torch.from_numpy(negative_samples)
        return positive_sample, negative_samples, subsampling_weight, self.mode

    def __len__(self):
        return len(self.triples)
    
    @staticmethod
    def collate_fn(batch):
        positive_samples = torch.stack([x[0] for x in batch], dim=0)
        negative_samples = torch.stack([x[1] for x in batch], dim=0)
        subsampling_weight = torch.cat([x[2] for x in batch], dim=0)
        mode = batch[0][3]
        return positive_samples, negative_samples, subsampling_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[((head, relation))] += 1
            
            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        true_head, true_tail = {}, {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = [] 
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_tail[(head, relation)].append(tail)
            true_head[(relation, tail)].append(head)
        
        for k, v in true_head.items():
            true_head[k] = np.array(list(set(v)))
        for k, v in true_tail.items():
            true_tail[k] = np.array(list(set(v)))

        return true_head, true_tail


class MultiDataLoaderIterator:
    def __init__(self, dataloaders):
        self.num_dataloaders = len(dataloaders)
        self.iterators = []
        for dataloader in dataloaders:
            self.iterators.append(self.oneshot_iterator(dataloader))
        self.step = 0

    def __next__(self):
        self.step += 1
        idx = self.step % self.num_dataloaders
        return next(self.iterators[idx])
    
    def oneshot_iterator(self, dataloader):
        while True:
            for data in dataloader:
                yield data