import os
import io
import sys
import datetime
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.append(os.getcwd())
from translations.model import KGEModel
from utils.common_utils import read_lines
from datasets.kge_dataset import KGETrainDataset, MultiDataLoaderIterator

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", 
                    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.StreamHandler()
logging.getLogger().addHandler(logger)


def read_item_info(file, bucket, sep="\t"):
    """ read item info file

    the item(entity/relation) file contains id and text
    """
    item2id = {}
    lines = read_lines(file, bucket)
    for line in lines:
        slist = line.strip().split(sep)
        if len(slist) == 2:
            id, text = slist
            item2id[text] = id
    return item2id


def read_triples(file, bucket, sep="\t", ent2id=None, rel2id=None):
    """ read triples file

    each line contains head_id, rel_id, tail_id
    """
    triples = []
    lines = read_lines(file, bucket)
    for line in lines:
        slist = line.strip().split(sep)
        if len(slist) == 3:
            h, r, t = slist
            triples.append((ent2id[h], rel2id[r], ent2id[t]))
    return triples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_val", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--negative_sample_size", type=int, default=1)
    parser.add_argument
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=2.0, help="the embedding range")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--save_checkpoint_steps", type=int, default=10)


def train(args):
    # load darasets
    triples = []        # TODO://
    collate_fn = KGETrainDataset.collate_fn
    train_dataset_head = KGETrainDataset(triples, num_entities=args.num_entities, 
                                    num_relations=args.num_relations, mode="head-batch", 
                                    negative_sample_size=args.negative_sample_size)
    train_dataset_tail = KGETrainDataset(triples, num_entities=args.num_entities, 
                                    num_relations=args.num_relations, mode="tail-batch", 
                                    negative_sample_size=args.negative_sample_size)
    train_loader_head = DataLoader(train_dataset_head, batch_size=args.batch_size, 
                                    shuffle=True, collate_fn=collate_fn)
    train_loader_tail = DataLoader(train_dataset_tail, batch_size=args.batch_size, 
                                    shuffle=True, collate_fn=collate_fn)
    train_iterator = MultiDataLoaderIterator([train_loader_head, train_loader_tail])
    model = KGEModel(model_name=args.model_name, 
                    num_entities=args.num_entities, 
                    num_relations=args.num_relations,
                    hidden_dim=args.hidden_dim, 
                    gamma=args.gamma).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    init_step = 0
    for step in range(init_step, args.max_train_steps):
        loss = model.train_step(args, model, optimizer, train_iterator)


def main(args):
    # create directory to save checkpoint and embeddings
    if args.save_path and args.bucket is None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_path)
    
    # load dataset
    entity2id = read_item_info(args.data_dir + "/entities.dict", args.bucket)
    relation2id = read_item_info(args.data_dir + "/relations.dict", args.bucket)
    args.num_entities = len(entity2id)
    args.num_relations = len(relation2id)
    logging.info(f"# entity: {args.num_entities}, # relation: {args.num_relations}")

    if args.do_train:
        train_triples = read_triples(args.data_dir + "/train.tsv", args.bucket, 
                        ent2id=entity2id, rel2id=relation2id)
        logging.info(f"# train samples: {len(train_triples)}")
    if args.do_val:
        val_triples = read_triples(args.data_dir + "/val.tsv", args.bucket, 
                        ent2id=entity2id, rel2id=relation2id)
        logging.info(f"# train samples: {len(val_triples)}")
    if args.do_test:
        test_triples = read_triples(args.data_dir + "/test.tsv", args.bucket, 
                        ent2id=entity2id, rel2id=relation2id)
        logging.info(f"# train samples: {len(test_triples)}")
    
    model = KGEModel(model_name=args.model_name, 
                    num_entities=args.num_entities, 
                    num_relations=args.num_relations,
                    hidden_dim=args.hidden_dim, 
                    gamma=args.gamma).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    init_step = 0
    collate_fn = KGETrainDataset.collate_fn
    if args.do_train:
        train_loader_head = DataLoader(
            KGETrainDataset(train_triples, args.num_entities, args.num_relations, 
                            "head-batch", args.negative_sample_size),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        train_loader_tail = DataLoader(
            KGETrainDataset(train_triples, args.num_entities, args.num_relations, 
                            "tail-batch", args.negative_sample_size),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn))
        train_iterator = MultiDataLoaderIterator([train_loader_head, train_loader_tail])

        logging.info(f"# batch_size: {args.batch_size}")
        logging.info(f"# max train steps: {args.max_train_steps}")
        losses = []
        for step in range(init_step, args.max_train_steps):
            loss = model.train_step(args, model, optimizer, train_iterator)
            losses.append(loss)
            if step % args.save_checkpoint_steps == 0:
                torch_save(args.save_dir + "model.pth", args.bucket, model=model)
            if step % args.log_steps == 0:
                pass        # TODO://

            if args.do_val and step % args.val_steps == 0:
                pass
        torch_save(args.save_dir + "model.pth", args.bucket, model=model)
    if args.do_val:
        pass
    if args.do_test:
        pass



if __name__ == "__main__":
    args = parse_args()
    train(args)