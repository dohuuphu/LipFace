import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss, Lipschitz_loss
from lr_scheduler import PolyScheduler
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

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

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(
    description="Distributed Arcface Training in Pytorch")
parser.add_argument("config", type=str, help="py config file")
parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
args = parser.parse_args()
print(world_size, rank, local_rank)
# get config
cfg = get_config(args.config)
# global control random seed
setup_seed(seed=cfg.seed, cuda_deterministic=False)

#torch.cuda.set_device(args.local_rank)
torch.cuda.set_device(local_rank)

backbone = get_model(
    cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

backbone = torch.nn.parallel.DistributedDataParallel(
    #module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
    module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
    find_unused_parameters=True)

##
backbone.eval()
for name, param in backbone.named_parameters():
    param.requires_grad = False
non_frozen_parameters = [p for p in backbone.parameters() if p.requires_grad]
# FIXME using gradient checkpoint if there are some unused parameters will cause error
backbone._set_static_graph()

backbone_eval = get_model(
    cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

backbone_eval = torch.nn.parallel.DistributedDataParallel(
    #module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
    module=backbone_eval, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
    find_unused_parameters=True)

backbone_eval.eval()
for name, param in backbone_eval.named_parameters():
    param.requires_grad = False
non_frozen_parameters = [p for p in backbone_eval.parameters() if p.requires_grad]
# FIXME using gradient checkpoint if there are some unused parameters will cause error
backbone_eval._set_static_graph()

if cfg.pretrained != None:
    pretrained_dict = torch.load(cfg.pretrained)
    pretrained_dict_1 = torch.load(os.path.join(cfg.output, f"model.pt"))
    backbone.module.load_state_dict(pretrained_dict, strict=False)
    backbone_eval.module.load_state_dict(pretrained_dict_1, strict=False)
    '''
    for name, para in backbone.module.named_parameters():
        if name == 'fc.weight':
            print("-"*20)
            print(f"name: {name}")
            print("values: ")
            print(para)
    '''
    params1 = backbone.module.named_parameters()
    params2 = backbone_eval.module.named_parameters()
    for p1, p2 in zip(params1, params2):
        if p1[0] != p2[0]:
            print("Error: models have different parameter names!")
            break
        if torch.equal(p1[1], p2[1]):
            pass
        else:
            print(f"Parameters {p1[0]} are different")
    del pretrained_dict
    del pretrained_dict_1
    