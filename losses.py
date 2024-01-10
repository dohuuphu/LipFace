import torch
import math

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, norms, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            if self.easy_margin:
                final_target_logit = torch.where(
                    target_logit > 0, cos_theta_m, target_logit)
            else:
                final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        
        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise        

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, norms: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, norms: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits
        

class AdaFace(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.40):
        super(AdaFace, self).__init__()
        self.scale = s
        self.easy_margin = False
        self.eps = 1e-3
        self.h = 0.333
        self.m = margin

        # ema prep
        self.t_alpha = 1.0
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('AdaFace with the following property')
        print('margin: ', margin)
        print('h: ', self.h)
        print('scale: ', self.scale)
        print('t_alpha: ', self.t_alpha)

    def forward(self, logits: torch.Tensor, norms: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        
        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(labels.size()[0], target_logit.size()[1], device=target_logit.device)
        m_arc.scatter_(1, labels.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = target_logit.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        target_logit = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(labels.size()[0], target_logit.size()[1], device=target_logit.device)
        m_cos.scatter_(1, labels.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        target_logit = target_logit - m_cos

        logits[index, labels[index].view(-1)] = target_logit
        logits = logits * self.scale
        return logits
    

def cos_linearity_lipschitz(img, feat, feat_norm, tao, lip, squared, detach_HR, p=0.2):
    # img: [batchsz, 1+num_img_lip, 3, 112, 112]
    # feat: [batchsz, 1+num_img_lip, 512]
    batchsz = img.shape[0]
    num_img_lip = img.shape[1] - 1
    eps = 1e-10
    
    penalty = Variable(torch.zeros([int(batchsz * (num_img_lip))], dtype=torch.float).cuda())
    
    idx = 0
    for batch_ind in range(batchsz):
        # adaptive weights
        #print(feat_norm[batch_ind, 0])
        if feat_norm[batch_ind, 0] > -1.0*tao:
            adaptive_weight = 1.0 / (torch.exp(feat_norm[batch_ind, 0]) + eps)
            #print(adaptive_weight, torch.exp(feat_norm[batch_ind, 0]), torch.exp(feat_norm[batch_ind, 0]) + eps)
            for img_lip_ind in range(1, 1+num_img_lip): # 1, 2, 3
                if np.random.random() < p:
                    if not torch.equal(img[batch_ind, img_lip_ind], img[batch_ind, 0]):
                        # if two input samples are the same, nan would occur, even not divide by zero with additional eps
                        img_diff = (((img[batch_ind, img_lip_ind] - img[batch_ind, 0]) ** 2).sum()) ** (0.5)# + eps
                        
                        # L2
                        # feat_diff = (((feat[batch_ind, img_lip_ind] - feat[batch_ind, 0]) ** 2).sum()) ** (0.5)
                        # cosine dist
                        norms_HR = torch.norm(feat[batch_ind, 0]).detach()  # [batchsz, 3, 1]
                        norms_LR = torch.norm(feat[batch_ind, img_lip_ind]).detach()  # [batchsz, 3, 1]
                        
                        # feat[batch_ind, 0] = feat[batch_ind, 0] / norms_HR # [batchsz, 3, 512]
                        # feat[batch_ind, img_lip_ind] = feat[batch_ind, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                        feat_normalized_HR = feat[batch_ind, 0] / norms_HR # [batchsz, 3, 512]
                        feat_normalized_LR = feat[batch_ind, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                        
                        # print('norms_HR: ', norms_HR.shape)
                        # print('feat[batch_ind, 0:1]: ', feat[batch_ind, 0:1].shape)
                        # print('feat[batch_ind, img_lip_ind:img_lip_ind+1]: ', feat[batch_ind, img_lip_ind:img_lip_ind+1].shape)
                        #print('img_diff: ', img_diff)
                        
                        if detach_HR:
                            feat_diff = 1 - (F.cosine_similarity(feat_normalized_HR.unsqueeze(0).detach(), feat_normalized_LR.unsqueeze(0)))
                        else:
                            feat_diff = 1 - (F.cosine_similarity(feat_normalized_HR.unsqueeze(0), feat_normalized_LR.unsqueeze(0)))
                        #print('feat_diff: ', feat_diff)
        
                        if squared == False:
                            
                            penalty[idx] = torch.maximum(feat_diff / img_diff - lip, torch.tensor(0.).cuda()) * adaptive_weight[0].detach()
        
                            #print(penalty[idx])
                        else:
                            penalty[idx] = ((feat_diff / img_diff - lip) ** 2) * adaptive_weight[0].detach()

                # print(feat_diff / img_diff)
                #assert not torch.isnan(feat_diff / img_diff)

                idx += 1

    #print(torch.mean(penalty))
    return torch.mean(penalty)


class Lipschitz_loss(torch.nn.Module):
    def __init__(self,
                 squared: bool,
                 lip: float,
                 detach_HR: bool,
                 p: float,
                 tao: float,
                 fp16: bool,
                 lamb_lip: int):  
        super(Lipschitz_loss, self).__init__()
        self.eps = 1e-10
        self.p = p
        self.detach_HR = detach_HR
        self.lip = lip
        self.squared = squared
        self.fp16 = fp16
        self.tao = tao
        self.lamb_lip = lamb_lip

    
    def forward(self, img, feat, feat_norm):
        # img: [batchsz, 1+num_img_lip, 3, 112, 112]
        # feat: [batchsz, 1+num_img_lip, 512]
        batchsz = img.shape[0]
        num_img_lip = img.shape[1] - 1
        
        penalty = Variable(torch.zeros([int(batchsz)], dtype=torch.float).cuda())
        adaptive_weight_list = Variable(torch.zeros([int(batchsz)], dtype=torch.float).cuda())
        
        idx = 0
        for batch_ind in range(batchsz):
            if feat_norm[batch_ind, 0] > -1.0*self.tao:
                adaptive_weight = 1.0 / (torch.exp(feat_norm[batch_ind, 0]) + self.eps)
                adaptive_weight_list[idx] = adaptive_weight[0].detach()
                for img_lip_ind in range(1, 1+num_img_lip): # 1, 2, 3
                    if np.random.random() < self.p:
                        if not torch.equal(img[batch_ind, img_lip_ind], img[batch_ind, 0]):
                            # if two input samples are the same, nan would occur, even not divide by zero with additional eps
                            img_diff = (((img[batch_ind, img_lip_ind] - img[batch_ind, img_lip_ind-1]) ** 2).sum()) ** (0.5) + self.eps
                            
                            # L2
                            # feat_diff = (((feat[batch_ind, img_lip_ind] - feat[batch_ind, 0]) ** 2).sum()) ** (0.5)
                            # cosine dist
                            '''
                            norms_HR = torch.norm(feat[batch_ind, 0]).detach()  # [batchsz, 3, 1]
                            norms_LR = torch.norm(feat[batch_ind, img_lip_ind]).detach()  # [batchsz, 3, 1]
                            
                            feat_normalized_HR = feat[batch_ind, 0] / norms_HR # [batchsz, 3, 512]
                            feat_normalized_LR = feat[batch_ind, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                            '''
                            feat_normalized_HR = feat[batch_ind, img_lip_ind-1] # [batchsz, 3, 512]
                            feat_normalized_LR = feat[batch_ind, img_lip_ind] # [batchsz, 3, 512]                      
                            
                            if self.detach_HR:
                                feat_diff = 1 - (F.cosine_similarity(feat_normalized_HR.unsqueeze(0).detach(), feat_normalized_LR.unsqueeze(0)))
                            else:
                                feat_diff = 1 - (F.cosine_similarity(feat_normalized_HR.unsqueeze(0), feat_normalized_LR.unsqueeze(0)))
            
                            if self.squared == False:          
                                penalty[idx] = penalty[idx] + torch.maximum(feat_diff / img_diff - self.lip, torch.tensor(0.).cuda()) * adaptive_weight[0].detach()
                            else:
                                penalty[idx] = penalty[idx] + (feat_diff / img_diff - self.lip) ** 2 * adaptive_weight[0].detach()

            idx += 1
        '''
        for img_lip_ind in range(1, 1+num_img_lip): # 1, 2, 3
            if np.random.random() < self.p:
                if not torch.equal(img[:, img_lip_ind], img[:, 0]):
                    # if two input samples are the same, nan would occur, even not divide by zero with additional eps
                    img_diff = (((img[:, img_lip_ind] - img[:, 0]) ** 2).sum()) ** (0.5) + self.eps
                    
                    # L2
                    # feat_diff = (((feat[batch_ind, img_lip_ind] - feat[batch_ind, 0]) ** 2).sum()) ** (0.5)
                    # cosine dist
                    
                    norms_HR = torch.norm(feat[:, 0]).detach()  # [batchsz, 3, 1]
                    norms_LR = torch.norm(feat[:, img_lip_ind]).detach()  # [batchsz, 3, 1]
                    
                    feat_normalized_HR = feat[:, 0] / norms_HR # [batchsz, 3, 512]
                    feat_normalized_LR = feat[:, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                    
                    feat_normalized_HR = feat[:, 0] # [batchsz, 3, 512]
                    feat_normalized_LR = feat[:, img_lip_ind] # [batchsz, 3, 512]                      
                    
                    if self.detach_HR:
                        feat_diff = 1 - (F.cosine_similarity(feat_normalized_HR.unsqueeze(0).detach(), feat_normalized_LR.unsqueeze(0)))
                    else:
                        feat_diff = 1 - (F.cosine_similarity(feat_normalized_HR.unsqueeze(0), feat_normalized_LR.unsqueeze(0)))
    
                    if self.squared == False:          
                        penalty = torch.maximum(feat_diff / img_diff - self.lip, torch.tensor(0.).cuda()) #* adaptive_weight[0].detach()
                    else:
                        penalty = (feat_diff / img_diff - self.lip) ** 2 #* adaptive_weight[0].detach()
        '''
        return self.lamb_lip * torch.mean(penalty), adaptive_weight_list

