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


def freeze_all(net):
    for name, param in net.named_parameters():
        param.requires_grad = False
        
def unfreeze_all(net):
    for name, param in net.named_parameters():
        param.requires_grad = True


def main(args):
    print(world_size, rank, local_rank)
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    #torch.cuda.set_device(args.local_rank)
    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    train_loader = get_dataloader(
        cfg.rec,
        #args.local_rank,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers,
        cfg.num_img_lip
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        #module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    ##
    backbone.train()
    # backbone.eval()
    
    # for name, param in backbone.named_parameters():
    #     param.requires_grad = False
    
    #non_frozen_parameters = [p for p in backbone.parameters() if p.requires_grad]
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()
    
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    
    #margin_loss = AdaFace(s=64.0, margin=0.4)
    
    lipschitz_loss = Lipschitz_loss(
        cfg.squared,
        cfg.lip,
        cfg.detach_HR,
        cfg.p,
        cfg.tao,
        cfg.fp16,
        cfg.lamb_lip
    )

    if cfg.optimizer == "sgd":

        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        ##
        # module_partial_fc.train().cuda()
        module_partial_fc.eval().cuda()
        # for name, param in module_partial_fc.named_parameters():
        #     param.requires_grad = False
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            ##
            params=[{"params": backbone.parameters()}],
            # params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    
    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise
    

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    start_epoch = 0
    global_step = 0
    
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.ckpt_dir, f"checkpoint_gpu_{rank}.pt"))
        #start_epoch = dict_checkpoint["epoch"]
        #global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        #opt.load_state_dict(dict_checkpoint["state_optimizer"])
        #lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        print(f"load backbone success {local_rank}.") 
        del dict_checkpoint
    
    # if cfg.pretrained != None:
    #     pretrained_dict = torch.load(cfg.pretrained)
    #     backbone.module.load_state_dict(pretrained_dict, strict=False)
    #     print(f"load backbone success {local_rank}.") 
    #     del pretrained_dict
    
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    
    # adaptive hyperparameter
    mean_all = torch.tensor(0.0).cuda()
    var_all = torch.tensor(0.0).cuda()

    # train
    for epoch in range(start_epoch, cfg.num_epoch):
        #print(f"epoch: {epoch}")
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for idx, (img, local_labels) in enumerate(train_loader):
            # reshape
            img = torch.reshape(img, (cfg.batch_size*(1+cfg.num_img_lip), 3, 112, 112))

            # feature extractor forward
            global_step += 1
            local_embeddings, local_norms = backbone(img)
            local_embeddings_div_norm = torch.div(local_embeddings, local_norms)
            
            # # mean & var
            with torch.no_grad():
                mean_all = cfg.alpha * torch.mean(local_norms).detach() + (1-cfg.alpha) * mean_all
                var_all = cfg.alpha * torch.var(local_norms).detach() + (1-cfg.alpha) * var_all
            local_embeddings_normalize = (local_norms - mean_all) / (var_all + 1e-10)
            

            # reshape
            local_embeddings = torch.reshape(local_embeddings, (cfg.batch_size, (1+cfg.num_img_lip), cfg.embedding_size))
            assert not torch.isnan(local_embeddings).any()
            local_norms = torch.reshape(local_norms, (cfg.batch_size, (1+cfg.num_img_lip), 1))
            assert not torch.isnan(local_norms).any()
            local_embeddings_normalize = torch.reshape(local_embeddings_normalize, (cfg.batch_size, (1+cfg.num_img_lip), 1))
            assert not torch.isnan(local_embeddings_normalize).any()
            local_embeddings_div_norm = torch.reshape(local_embeddings_div_norm, (cfg.batch_size, (1+cfg.num_img_lip), cfg.embedding_size))
            assert not torch.isnan(local_embeddings_div_norm).any()

            # loss
            loss_arc: torch.Tensor = torch.tensor(0.0).cuda()
            loss_lip: torch.Tensor = torch.tensor(0.0).cuda()
            
            if global_step >= 0:
                # cos_linearity
                ## TODO: adaptive weight for lipface loss according to feature magnitude or data uncertainty
                img = torch.reshape(img, (cfg.batch_size, 1+cfg.num_img_lip, 3, 112, 112)) ##
                loss_lip, adaptive_weight_list = lipschitz_loss(img, local_embeddings_div_norm, local_embeddings_normalize)
                # margin loss
                loss_arc: torch.Tensor = module_partial_fc(local_embeddings[:, 0, :], local_norms[:, 0, :], local_labels, adaptive_weight_list)
                loss = loss_arc + loss_lip
            
            # else:
            # loss_arc: torch.Tensor = module_partial_fc(local_embeddings[:, 0, :], local_norms[:, 0, :], local_labels, None)
            # loss = loss_arc
            # loss_arc_1: torch.Tensor = module_partial_fc(local_embeddings[:, 0, :], local_norms[:, 0, :], local_labels, None)
            # loss_arc_2: torch.Tensor = module_partial_fc(local_embeddings[:, 1, :], local_norms[:, 1, :], local_labels, None)
            # loss_arc = loss_arc_1 + loss_arc_2
            # loss = loss_arc
            
            #print(f"loss_arc: {loss_arc}, loss_lip: {loss_lip}, loss_lip_negative: {loss_lip_negative}")
            
            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, loss_arc, loss_lip, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)
            # if idx == 3:
            #     print('breakkkk')
            #     break
        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, f"model_e{epoch}.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, f"model_e{epoch}.pt")
        torch.save(backbone.module.state_dict(), path_module)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local-rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
