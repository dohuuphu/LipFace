import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Variable

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss, AdaFace, cos_linearity_lipschitz, cos_linearity_lipschitz_negative
from lipschitz import Lipschitz_loss
from lr_scheduler import PolyScheduler
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."
'''
try:
    distributed.init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )
'''

def main(args):
    local_rank = 2
    #print(world_size, rank, local_rank)
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )
    print("trainloader success.")

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone.eval()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error

    start_epoch = 0
    global_step = 0
    
    if cfg.pretrained != None:
        pretrained_dict = torch.load(cfg.pretrained)
        backbone.load_state_dict(pretrained_dict, strict=False)
        print("load backbone success.") 

    with torch.no_grad():
        # adaptive hyperparameter
        mean_all = torch.tensor(0.0).cuda()
        var_all = torch.tensor(0.0).cuda()

        for i, (img, local_labels) in enumerate(train_loader):
            # reshape
            img = torch.reshape(img, (cfg.batch_size*(1+cfg.num_img_lip), 3, 112, 112))

            # feature extractor forward
            global_step += 1
            local_embeddings, local_norms, _ = backbone(img)

            # mean & var
            with torch.no_grad():
                mean_all = cfg.alpha * torch.mean(torch.norm(local_embeddings, 2, 1, True)).detach() + (1-cfg.alpha) * mean_all
                var_all = cfg.alpha * torch.var(torch.norm(local_embeddings, 2, 1, True)).detach() + (1-cfg.alpha) * var_all
            local_embeddings_normalize = (local_norms - mean_all) / (var_all + 1e-10)
            #print(local_embeddings_normalize)

            # reshape
            local_embeddings = torch.reshape(local_embeddings, (cfg.batch_size, (1+cfg.num_img_lip), cfg.embedding_size))
            assert not torch.isnan(local_embeddings).any()
            local_norms = torch.reshape(local_norms, (cfg.batch_size, (1+cfg.num_img_lip), 1))
            assert not torch.isnan(local_norms).any()
            local_embeddings_normalize = torch.reshape(local_embeddings_normalize, (cfg.batch_size, (1+cfg.num_img_lip), 1))
            assert not torch.isnan(local_embeddings_normalize).any()
            img = torch.reshape(img, (cfg.batch_size, 1+cfg.num_img_lip, 3, 112, 112))

            batchsz = img.shape[0]
            num_img_lip = img.shape[1] - 1
            eps = 1e-10
            
            penalty = Variable(torch.zeros([int(batchsz * (num_img_lip))], dtype=torch.float).cuda())
            inp_all = Variable(torch.zeros([int(batchsz * (num_img_lip))], dtype=torch.float).cuda())
            out_all = Variable(torch.zeros([int(batchsz * (num_img_lip))], dtype=torch.float).cuda())
            adaptive_all = Variable(torch.zeros([int(batchsz * (num_img_lip))], dtype=torch.float).cuda())
            hr_feat = Variable(torch.zeros([int(batchsz * (num_img_lip))], dtype=torch.float).cuda())
            lr_feat = Variable(torch.zeros([int(batchsz * (num_img_lip))], dtype=torch.float).cuda())
            
            idx = 0
            for batch_ind in range(batchsz):
                # adaptive weights
                #print(out_norm[batch_ind, 0])
                if local_embeddings_normalize[batch_ind, 0] > -1.0*cfg.tao:
                    adaptive_weight = 1.0 / (torch.exp(local_embeddings_normalize[batch_ind, 0]) + eps)
                    #print(adaptive_weight, torch.exp(out_norm[batch_ind, 0]), torch.exp(out_norm[batch_ind, 0]) + eps)
                    for img_lip_ind in range(1, 1+num_img_lip): # 1, 2, 3
                        #if np.random.random() < cfg.p:
                        if not torch.equal(img[batch_ind, img_lip_ind], img[batch_ind, 0]):
                            # if two input samples are the same, nan would occur, even not divide by zero with additional eps
                            inp_diff = (((img[batch_ind, img_lip_ind] - img[batch_ind, 0]) ** 2).sum()) ** (0.5)# + eps
                            #print(torch.norm(local_embeddings[batch_ind, 0]))
                            # L2
                            # out_diff = (((out[batch_ind, img_lip_ind] - out[batch_ind, 0]) ** 2).sum()) ** (0.5)
                            # cosine dist
                            norms_HR = torch.norm(local_embeddings[batch_ind, 0]).detach()  # [batchsz, 3, 1]
                            norms_LR = torch.norm(local_embeddings[batch_ind, img_lip_ind]).detach()  # [batchsz, 3, 1]
                            
                            # out[batch_ind, 0] = out[batch_ind, 0] / norms_HR # [batchsz, 3, 512]
                            # out[batch_ind, img_lip_ind] = out[batch_ind, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                            out_normalized_HR = local_embeddings[batch_ind, 0] / norms_HR # [batchsz, 3, 512]
                            out_normalized_LR = local_embeddings[batch_ind, img_lip_ind] / norms_LR # [batchsz, 3, 512]
                            
                            # print('norms_HR: ', norms_HR.shape)
                            # print('out[batch_ind, 0:1]: ', out[batch_ind, 0:1].shape)
                            # print('out[batch_ind, img_lip_ind:img_lip_ind+1]: ', out[batch_ind, img_lip_ind:img_lip_ind+1].shape)
                            #print('inp_diff: ', inp_diff)
                            
                            if cfg.detach_HR:
                                out_diff = 1 - (F.cosine_similarity(out_normalized_HR.unsqueeze(0).detach(), out_normalized_LR.unsqueeze(0)))
                            else:
                                out_diff = 1 - (F.cosine_similarity(out_normalized_HR.unsqueeze(0), out_normalized_LR.unsqueeze(0)))
                            #print(out_diff)
            
                            if cfg.squared == False:
                                
                                penalty[idx] = torch.maximum(out_diff / inp_diff - cfg.lip, torch.tensor(0.).cuda()) * adaptive_weight[0].detach()
                                inp_all[idx] = inp_diff
                                out_all[idx] = out_diff
                                adaptive_all[idx] = adaptive_weight[0]
                                hr_feat[idx] = torch.norm(local_embeddings[batch_ind, 0])
                                lr_feat[idx] = torch.norm(local_embeddings[batch_ind, img_lip_ind])
            
                                #print(penalty[idx])
                            else:
                                penalty[idx] = ((out_diff / inp_diff - cfg.lip) ** 2) * adaptive_weight[0].detach()
                                inp_all[idx] = inp_diff
                                out_all[idx] = out_diff
                                adaptive_all[idx] = adaptive_weight[0]
                                hr_feat[idx] = torch.norm(local_embeddings[batch_ind, 0])
                                lr_feat[idx] = torch.norm(local_embeddings[batch_ind, img_lip_ind])

                        # print(out_diff / inp_diff)
                        #assert not torch.isnan(out_diff / inp_diff)

                        idx += 1

            print(i, torch.mean(inp_all), torch.mean(out_all), torch.mean(adaptive_all), torch.mean(penalty))
            print(i, torch.mean(hr_feat), torch.mean(lr_feat))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
