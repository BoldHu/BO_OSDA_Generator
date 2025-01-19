import torch
import torch.nn as nn
import torch.nn.functional as F

# loss function which smooth the label
class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=None, class_num=None):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-2

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot

            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.sum()

def InfoNCELoss(sm, sm_positive, sm_negative, trfm, temperature=0.07):
    '''
    Function: Compute the InfoNCE loss using transformer encoder
    
    Args:
        sm: Tensor of shape (batch_size, seq_len) - origin SMILES sequences
        sm_positive: Tensor of shape (batch_size, P, seq_len) - each SMILES has P positive samples
        sm_negative: Tensor of shape (batch_size, N, seq_len) - each SMILES has N negative samples
        trfm: Transformer 
        temperature: scale factor for logits
    
    Returns:
        loss: Tensor scalar
    '''
    batch_size = sm.size(0)
    P = sm_positive.size(1)  # the number of positive samples for each sample
    N = sm_negative.size(1)  # the number of negative samples for each sample

    device = sm.device

    # encode anchor
    # trfm.encode (seq_len, batch_size)
    sm_encoded = sm.transpose(0, 1)  # (seq_len, batch_size)
    anchors = trfm.encode(sm_encoded)  # (batch_size, embed_dim)

    # encode positive samples
    sm_positive_flat = sm_positive.view(batch_size * P, sm_positive.size(2))  # (batch_size * P, seq_len)
    sm_positive_flat = sm_positive_flat.transpose(0, 1)  # (seq_len, batch_size * P)
    positives = trfm.encode(sm_positive_flat)  # (batch_size * P, embed_dim)

    # encode negative samples
    sm_negative_flat = sm_negative.view(batch_size * N, sm_negative.size(2))  # (batch_size * N, seq_len)
    sm_negative_flat = sm_negative_flat.transpose(0, 1)  # (seq_len, batch_size * N)
    negatives = trfm.encode(sm_negative_flat)  # (batch_size * N, embed_dim)

    # calculate cosine similarity between anchor and positive samples
    # anchors: (batch_size, embed_dim) -> (batch_size * P, embed_dim)
    anchors_exp = anchors.repeat_interleave(P, dim=0)  # (batch_size * P, embed_dim)
    sim_p = F.cosine_similarity(anchors_exp, positives, dim=1) / temperature  # (batch_size * P)

    # calculate cosine similarity between anchor and negative samples
    # negatives: (batch_size * N, embed_dim) -> (batch_size, N, embed_dim)
    negatives = negatives.view(batch_size, N, -1)  # (batch_size, N, embed_dim)
    # repeat anchor for N times
    negatives_exp = negatives.repeat_interleave(P, dim=0)  # (batch_size * P, N, embed_dim)
    # anchors_exp: (batch_size * P, embed_dim) -> (batch_size * P, 1, embed_dim)
    anchors_exp_neg = anchors_exp.unsqueeze(1)  # (batch_size * P, 1, embed_dim)
    # calculate cosine similarity
    sim_n = F.cosine_similarity(anchors_exp_neg, negatives_exp, dim=2) / temperature  # (batch_size * P, N)

    # build logits
    # concatenate positive and negative samples
    logits = torch.cat([sim_p.unsqueeze(1), sim_n], dim=1)  # (batch_size * P, 1 + N)

    # 创建标签，正样本为第一个类别
    labels = torch.zeros(batch_size * P).long().to(device)  # (batch_size * P)

    # calcyulate loss
    loss = F.cross_entropy(logits, labels)

    return loss
