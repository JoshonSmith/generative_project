from packaging import version
import torch
from torch import nn



class PatchNCELoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, weighted = None):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        if self.cfg.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.cfg.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        if weighted is not None:
            if self.cfg.prob_weighted:
                l_neg_curbatch.masked_fill_(diagonal, -10.0)
                l_neg = l_neg_curbatch.view(-1, npatches)
                weighted = weighted.view(-1,npatches)
                weighted = torch.cat((torch.ones([l_pos.shape[0],1],device=l_pos.device),weighted),dim=1)

                out = torch.cat((l_pos, l_neg), dim=1) / self.cfg.nce_T

                out = out.exp()
                out = out*weighted
                out = out/torch.sum(out,dim=1,keepdim=True)
                loss = -torch.log(out[:,0])
                return loss
            else:
                l_neg_curbatch = l_neg_curbatch*weighted
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.cfg.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                            device=feat_q.device))

        return loss

class PatchNCELoss_maxcut(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, sample_id):
        dev = feat_q.device
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[-1]
        feat_k = feat_k.detach()
        #print(feat_q.shape,feat_k.shape)

        sim_qk = torch.bmm(feat_q, feat_k.transpose(2, 1)) # b x num_patch x (h x w)
        l_pos, pos_idx = sim_qk.max(dim=-1) # b, num_patch

        if self.cfg.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.cfg.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_sample = feat_k[:, sample_id, :]
        feat_k_sample = feat_k_sample.view(batch_dim_for_bmm, -1, dim)

        npatches = feat_q.size(1)
        l_curbatch = torch.bmm(feat_q, feat_k_sample.transpose(2, 1))

        idx = torch.arange(npatches, dtype=torch.long, device=dev)
        l_max = l_curbatch.clone()
        l_max[:, idx, idx] = l_pos
        l_max = l_max.view(-1, npatches) # nce loss with max cosine similarity as positives
        l_same = l_curbatch.view(-1, npatches) # nce loss with embeddings from the same location as positives

        labels = torch.tensor(list(range(npatches)) * batchSize, dtype=torch.long, device=dev)
        loss_max = self.cross_entropy_loss(l_max / self.cfg.nce_T, labels)
        loss_same = self.cross_entropy_loss(l_same / self.cfg.nce_T, labels)

        loss = 0.5 * loss_same + 0.5 * loss_max
        return loss

class PatchNCELoss_bicut(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.similarity_function = self._get_similarity_function()
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def _get_similarity_function(self):

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()
        l_pos = self.cos(feat_q,feat_k)
        l_pos = l_pos.view(batchSize, 1)
        l_neg_curbatch = self.similarity_function(feat_q.view(batchSize,1,-1),feat_k.view(1,batchSize,-1))
        l_neg_curbatch = l_neg_curbatch.view(1,batchSize,-1)
        diagonal = torch.eye(batchSize, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, batchSize)
        out = torch.cat((l_pos, l_neg), dim=1) / self.cfg.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss

class PatchNCELoss_v2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, sample_id):
        dev = feat_q.device
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[-1]
        feat_k = feat_k.detach()

        sim_qk = torch.bmm(feat_q, feat_k.transpose(2, 1)) # b x num_patch x (h x w)
        l_pos, pos_idx = sim_qk.max(dim=-1) # b, num_patch

        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_sample = feat_k[:, sample_id, :]
        feat_k_sample = feat_k_sample.view(batch_dim_for_bmm, -1, dim)

        npatches = feat_q.size(1)
        l_curbatch = torch.bmm(feat_q, feat_k_sample.transpose(2, 1))

        idx = torch.arange(npatches, dtype=torch.long, device=dev)
        l_max = l_curbatch.clone()
        l_max[:, idx, idx] = l_pos
        l_max = l_max.view(-1, npatches)
        l_same = l_curbatch.view(-1, npatches)

        labels = torch.tensor(list(range(npatches)) * batchSize, dtype=torch.long, device=dev)
        loss_max = self.cross_entropy_loss(l_max / self.opt.nce_T, labels)
        loss_same = self.cross_entropy_loss(l_same / self.opt.nce_T, labels)

        loss = 0.7 * loss_same + 0.3 * loss_max
        return loss