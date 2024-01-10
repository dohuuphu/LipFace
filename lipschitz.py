import torch
import math

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Lipschitz_loss(torch.nn.Module):
    def __init__(self, 
                 embedding_size: int,
                 num_classes: int,
                 max_id_size: int,
                 valid_step: int,
                 squared,
                 lip,
                 detach_HR,
                 p = 0.2,
                 fp16: bool = False,
                 lip_negative: bool = True):
        super(Lipschitz_loss, self).__init__()
        self.eps = 1e-10
        self.p = p
        self.detach_HR = detach_HR
        self.lip = lip
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.squared = squared
        self.fp16 = fp16
        self.max_id_size = max_id_size
        self.valid_step = valid_step
        self.lip_negative = lip_negative

        self.memory_bank_HR = torch.zeros(self.num_classes, self.max_id_size, self.embedding_size).cuda()
        self.validness = torch.zeros(self.num_classes, self.max_id_size).cuda()

    
    def cos_linearity_lipschitz_positive(self, inp, out):
        # inp: [batchsz, 1+num_img_lip, 3, 112, 112]
        # out: [batchsz, 1+num_img_lip, 512]
        batchsz = inp.shape[0]
        num_img_lip = inp.shape[1] - 1
        
        penalty = Variable(torch.zeros([int(batchsz * (num_img_lip))], dtype=torch.float).cuda())
        
        idx = 0
        for batch_ind in range(batchsz):
            for img_lip_ind in range(1, 1+num_img_lip): # 1, 2, 3
                # if np.random.random() < self.p:
                if not torch.equal(inp[batch_ind, img_lip_ind], inp[batch_ind, 0]):
                    # if two input samples are the same, nan would occur, even not divide by zero with additional eps
                    inp_diff = (((inp[batch_ind, img_lip_ind] - inp[batch_ind, 0]) ** 2).sum()) ** (0.5)# + self.eps
                    
                    # L2
                    # out_diff = (((out[batch_ind, img_lip_ind] - out[batch_ind, 0]) ** 2).sum()) ** (0.5)
                    # cosine dist
                    norms_HR = torch.norm(out[batch_ind, 0]).detach()  # [batchsz, 3, 1]
                    norms_LR = torch.norm(out[batch_ind, img_lip_ind]).detach()  # [batchsz, 3, 1]
                    
                    # out[batch_ind, 0] = out[batch_ind, 0] / norms_HR # [batchsz, 3, 512]
                    # out[batch_ind, img_lip_ind] = out[batch_ind, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                    out_normalized_HR = out[batch_ind, 0] / norms_HR # [batchsz, 3, 512]
                    out_normalized_LR = out[batch_ind, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                    
                    # print('norms_HR: ', norms_HR.shape)
                    # print('out[batch_ind, 0:1]: ', out[batch_ind, 0:1].shape)
                    # print('out[batch_ind, img_lip_ind:img_lip_ind+1]: ', out[batch_ind, img_lip_ind:img_lip_ind+1].shape)
                    # print('inp_diff: ', inp_diff)
                    
                    if self.detach_HR:
                        out_diff = 1 - (F.cosine_similarity(out_normalized_HR.unsqueeze(0).detach(), out_normalized_LR.unsqueeze(0)))
                    else:
                        out_diff = 1 - (F.cosine_similarity(out_normalized_HR.unsqueeze(0), out_normalized_LR.unsqueeze(0)))
                    # print(out_diff)
    
                    if self.squared == False:
                        
                        penalty[idx] = torch.maximum(out_diff / inp_diff - self.lip, torch.tensor(0.).cuda())
    
                        # print(penalty[idx])
                    else:
                        penalty[idx] = (out_diff / inp_diff - self.lip) ** 2
                # print(out_diff / inp_diff)
                #assert not torch.isnan(out_diff / inp_diff)

                idx += 1

        return torch.mean(penalty)


    def cos_linearity_lipschitz_negative(self, inp, out, labels):
        # inp: [batchsz, 1+num_img_lip, 3, 112, 112]
        # out: [batchsz, 1+num_img_lip, 512]
        batchsz = inp.shape[0]
        num_img_lip = inp.shape[1] - 1
        
        penalty = Variable(torch.zeros([int(batchsz * (num_img_lip))], dtype=torch.float).cuda())
        
        idx = 0
        for batch_ind in range(batchsz):
            current_label = labels[batch_ind]
            for img_lip_ind in range(1, 1+num_img_lip): # 1, 2, 3
                #if np.random.random() < self.p:
                if not torch.equal(inp[batch_ind, img_lip_ind], inp[batch_ind, 0]):
                    # if two input samples are the same, nan would occur, even not divide by zero with additional eps
                    penalty_sum = torch.tensor(0.).cuda()
                    for batch_ind_neg in range(batchsz):
                        if batch_ind != batch_ind_neg:
                            label_neg = labels[batch_ind_neg]
                            for i in range(self.max_id_size):
                                if self.validness[label_neg][i] > 0:
                                    inp_diff = (((inp[batch_ind, img_lip_ind] - inp[batch_ind_neg, 0]) ** 2).sum()) ** (0.5)# + self.eps
                                    
                                    # L2
                                    # out_diff = (((out[batch_ind, img_lip_ind] - out[batch_ind, 0]) ** 2).sum()) ** (0.5)
                                    # cosine dist
                                    norms_HR = torch.norm(out[batch_ind_neg, 0]).detach()  # [batchsz, 3, 1]
                                    norms_LR = torch.norm(out[batch_ind, img_lip_ind]).detach()  # [batchsz, 3, 1]
                                    
                                    # out[batch_ind, 0] = out[batch_ind, 0] / norms_HR # [batchsz, 3, 512]
                                    # out[batch_ind, img_lip_ind] = out[batch_ind, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                                    out_normalized_HR = out[batch_ind_neg, 0] / norms_HR # [batchsz, 3, 512]
                                    out_normalized_LR = out[batch_ind, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                                    
                                    # print('norms_HR: ', norms_HR.shape)
                                    # print('out[batch_ind, 0:1]: ', out[batch_ind, 0:1].shape)
                                    # print('out[batch_ind, img_lip_ind:img_lip_ind+1]: ', out[batch_ind, img_lip_ind:img_lip_ind+1].shape)
                                    # print('inp_diff: ', inp_diff)
                                    
                                    if detach_HR:
                                        out_diff = 1 - (F.cosine_similarity(out_normalized_HR.unsqueeze(0).detach(), out_normalized_LR.unsqueeze(0)))
                                    else:
                                        out_diff = 1 - (F.cosine_similarity(out_normalized_HR.unsqueeze(0), out_normalized_LR.unsqueeze(0)))
                                    # print(out_diff)
                    
                                    if squared == False:
                                        
                                        penalty_sum = torch.maximum(out_diff / inp_diff - lip, torch.tensor(0.).cuda()) + penalty_sum
                    
                                        # print(penalty[idx])
                                    else:
                                        penalty_sum = (out_diff / inp_diff - lip) ** 2 + penalty_sum
                        
                    penalty[idx] = penalty_sum
                # print(out_diff / inp_diff)
                #assert not torch.isnan(out_diff / inp_diff)

                idx += 1

        return torch.mean(penalty) * -1.0
    

    def forward(self, inp, out, labels):
        assert inp.shape[0] == labels.shape[0]
        batchsz = inp.shape[0]
        num_img_lip = inp.shape[1] - 1
        for batch_ind in range(batchsz):
            embedding_now = out[batch_ind, 0]
            label_now = labels[batch_ind]
            valid_row = self.validness[label_now].unsqueeze(0)
            min_index = torch.argmin(valid_row)
            self.memory_bank_HR[label_now][min_index] = embedding_now
            self.validness[label_now][min_index] = self.valid_step
        self.validness = self.validness - 1

        if self.lip_negative:
            positive = self.cos_linearity_lipschitz_positive(inp, out)
            negative = self.cos_linearity_lipschitz_negative(inp, out, labels)
            return positive + negative
        else:
            return self.cos_linearity_lipschitz_positive(inp, out)