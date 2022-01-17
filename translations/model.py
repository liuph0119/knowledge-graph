import torch
from torch._C import Value
import torch.nn as nn


class KGEModel(nn.Module):
    def __init__(self, model_name: str, num_entities:int, 
            num_relations: int, hidden_dim:int, gamma:float,
            epsilon: float) -> None:
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.entity_dim = self.relation_dim = hidden_dim
        self.gamma = gamma

        embedding_range = 0.31 if self.model_name == "RESCAL" else gamma + epsilon
        # initialization
        self.entity_embedding = nn.Parameter(torch.zeros(num_entities, self.entity_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relations, self.entity_dim))
        nn.init.uniform_(self.entity_embedding, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(self.relation_embedding, a=-embedding_range, b=embedding_range)

        self.model_func_dict = {
            "DistMult": self.dist_mult
        }
        if model_name not in self.model_func_dict:
            raise ValueError(f"model `{model_name}` is not supported, valid models: {', '.join(list(self.model_func_dict.keys()))}")
    
    def forward(self, samples, mode="single"):
        """ calculate the score of a batch of triples

        if mode is "single", the sample is a batch of samples
        if mode is "head-batch" or "tail-batch", the samples contain 2 parts: 
            the positive samples and the negative entities 
        """
        if mode == "single":
            head_indices, rel_indices, tail_indices = samples 
            head = torch.index_select(self.entity_embedding, dim=0, index=head_indices).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=rel_indices).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_indices).unsqueeze(1)
        else:
            pass

        score = self.model_func_dict[self.model_name](head, relation, tail, mode)
    
    @staticmethod
    def dist_mult(head, relation, tail, mode):
        if mode == "head-batch":
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        return score.sum(dim=-1)
    

    @staticmethod
    def train_step(args, model, optimizer, data_loader):
        model.train()
        optimizer.zero_grad()
        positive_samples, negative_samples, subsampling_weight, mode = next(data_loader)

        positive_samples = positive_samples.to(args.device)
        negative_samples = negative_samples.to(args.device)
        subsampling_weight = subsampling_weight.to(args.device)

        negative_score = model((positive_samples, negative_samples), mode)
        positive_score = model(positive_samples, mode)

        pos_loss = -torch.log(torch.sigmoid(positive_score) + 1e-8).squeeze(1)
        neg_loss = -(torch.log(1 - torch.sigmoid(negative_score) + 1e-8) * (1 - args.uncertainty) + args.uncertainty * (torch.log(torch.sigmoid(negative_score) + 1e-8))).mean(dim=1)
        reg = args.reg_lambda * (model.entity_embedding.norm(p=3) ** 3 + model.relation_embedding.norm(p=3) ** 3)
        loss = (pos_loss + neg_loss) / 2 + reg

        loss.backward()
        optimizer.step()
        return loss.item()