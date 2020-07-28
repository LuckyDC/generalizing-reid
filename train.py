import logging
import os
import pprint

import torch
import torch.distributed as dist
import yaml
from apex import amp
from apex import parallel
from tensorboardX import SummaryWriter
from torch import optim

from data import get_cross_domain_train_loader
from data import get_test_loader
from data import get_train_loader
from engine import get_trainer
from models.model import Model


def train(cfg):
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        torch.distributed.init_process_group(backend="nccl", world_size=num_gpus)

    # set logger
    log_dir = os.path.join("logs/", cfg.source_dataset, cfg.prefix)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + "log.txt",
                        filemode="a")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # writer = SummaryWriter(log_dir, purge_step=0)

    if dist.is_initialized() and dist.get_rank() != 0:

        logger = writer = None
    else:
        logger.info(pprint.pformat(cfg))

    # training data loader
    if not cfg.joint_training:  # single domain
        train_loader = get_train_loader(root=os.path.join(cfg.source.root, cfg.source.train),
                                        batch_size=cfg.batch_size,
                                        image_size=cfg.image_size,
                                        random_flip=cfg.random_flip,
                                        random_crop=cfg.random_crop,
                                        random_erase=cfg.random_erase,
                                        color_jitter=cfg.color_jitter,
                                        padding=cfg.padding,
                                        num_workers=4)
    else:  # cross domain
        source_root = os.path.join(cfg.source.root, cfg.source.train)
        target_root = os.path.join(cfg.target.root, cfg.target.train)

        train_loader = get_cross_domain_train_loader(source_root=source_root,
                                                     target_root=target_root,
                                                     batch_size=cfg.batch_size,
                                                     random_flip=cfg.random_flip,
                                                     random_crop=cfg.random_crop,
                                                     random_erase=cfg.random_erase,
                                                     color_jitter=cfg.color_jitter,
                                                     padding=cfg.padding,
                                                     image_size=cfg.image_size,
                                                     num_workers=8)

    # evaluation data loader
    query_loader = None
    gallery_loader = None
    if cfg.eval_interval > 0:
        query_loader = get_test_loader(root=os.path.join(cfg.target.root, cfg.target.query),
                                       batch_size=512,
                                       image_size=cfg.image_size,
                                       num_workers=4)

        gallery_loader = get_test_loader(root=os.path.join(cfg.target.root, cfg.target.gallery),
                                         batch_size=512,
                                         image_size=cfg.image_size,
                                         num_workers=4)

    # model
    num_classes = cfg.source.num_id
    num_cam = cfg.source.num_cam + cfg.target.num_cam
    cam_ids = train_loader.dataset.target_dataset.cam_ids if cfg.joint_training else train_loader.dataset.cam_ids
    num_instances = len(train_loader.dataset.target_dataset) if cfg.joint_training else None

    model = Model(num_classes=num_classes,
                  drop_last_stride=cfg.drop_last_stride,
                  joint_training=cfg.joint_training,
                  num_instances=num_instances,
                  cam_ids=cam_ids,
                  num_cam=num_cam,
                  neighbor_mode=cfg.neighbor_mode,
                  neighbor_eps=cfg.neighbor_eps,
                  scale=cfg.scale,
                  mix=cfg.mix,
                  alpha=cfg.alpha)

    model.cuda()

    # optimizer
    ft_params = model.backbone.parameters()
    new_params = [param for name, param in model.named_parameters() if not name.startswith("backbone.")]
    param_groups = [{'params': ft_params, 'lr': cfg.ft_lr},
                    {'params': new_params, 'lr': cfg.new_params_lr}]

    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.wd)

    # convert model for mixed precision distributed training

    model, optimizer = amp.initialize(model, optimizer, enabled=cfg.fp16, opt_level="O2")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=cfg.lr_step,
                                                  gamma=0.1)

    if dist.is_initialized():
        model = parallel.DistributedDataParallel(model, delay_allreduce=True)

    # engine
    checkpoint_dir = os.path.join("checkpoints", cfg.source_dataset, cfg.prefix)
    engine = get_trainer(model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         logger=logger,
                         # writer=writer,
                         non_blocking=True,
                         log_period=cfg.log_period,
                         save_interval=10,
                         save_dir=checkpoint_dir,
                         prefix=cfg.prefix,
                         eval_interval=cfg.eval_interval,
                         query_loader=query_loader,
                         gallery_loader=gallery_loader)

    # training
    engine.run(train_loader, max_epochs=cfg.num_epoch)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    import argparse
    import random
    import numpy as np
    from yacs.config import CfgNode
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/market2duke.yml")
    parser.add_argument("--local_rank", type=int, default=None)
    args = parser.parse_args()

    # set random seed
    seed = 0 if args.local_rank is None else args.local_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # enable cudnn backend
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True

    # load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.SafeLoader)

    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    source_cfg = dataset_cfg.get(cfg.source_dataset)
    target_cfg = dataset_cfg.get(cfg.target_dataset)

    cfg.source = CfgNode()
    for k, v in source_cfg.items():
        cfg.source[k] = v

    cfg.target = CfgNode()
    for k, v in target_cfg.items():
        cfg.target[k] = v

    # split data batch into all devices evenly
    cfg.batch_size = cfg.batch_size // torch.cuda.device_count()

    cfg.freeze()

    train(cfg)
