import numpy as np
import torch
import torch.nn as nn
import pdb

class TransE(nn.Module):

    def __init__(self, entity_count, relation_count, device, norm=1, dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim = dim
        # self.entities_emb = self._init_enitity_emb()
        self.relations_emb = self._init_relation_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    # def _init_enitity_emb(self):
    #     entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
    #                                 embedding_dim=self.dim,
    #                                 padding_idx=self.entity_count)
    #     uniform_range = 6 / np.sqrt(self.dim)
    #     entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
    #     return entities_emb

    # def _init_relation_emb(self):
    #     relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
    #                                  embedding_dim=self.dim,
    #                                  padding_idx=self.relation_count)
    #     uniform_range = 6 / np.sqrt(self.dim)
    #     relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
    #     # -1 to avoid nan for OOV vector
    #     relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
    #     return relations_emb
    
    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        import torch.nn.init as init

        # Assuming relations_emb is already initialized
        init.constant_(relations_emb.weight.data, 1)
        return relations_emb

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        """Return model losses based on the input.

        :param positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param negative_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        # -1 to avoid nan for OOV vector
        # self.entities_emb.weight.data[:-1, :].div_(self.entities_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        # pdb.set_trace()
        # assert positive_triplets.size()[1] == 3
        positive_distances = self._distance(positive_triplets, 0)

        # assert negative_triplets.size()[1] == 3
        negative_distances = self._distance(negative_triplets, 1)

        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances

    def predict(self, triplets: torch.LongTensor):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(triplets)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, triplets, flag = 0):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        # assert triplets.size()[1] == 3
        # pdb.set_trace()
        if flag==0:
            heads = triplets[:-1, :]
            relations = torch.zeros(len(triplets)-1).int().to(heads.device)
            tails = triplets[1:, :]
            
            return  (heads+self.relations_emb(relations)-tails).norm(p=self.norm, dim=1)
        elif flag==1:
            heads = triplets[1:, :]
            relations = torch.zeros(len(triplets)-1).int().to(heads.device)
            tails = triplets[:-1, :]
            
            return  (heads+self.relations_emb(relations)-tails).norm(p=self.norm, dim=1)
        # return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm,
                                                                                                        #   dim=1)