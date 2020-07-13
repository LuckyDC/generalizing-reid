import logging

import numpy as np
import torch
import torch.distributed as dist
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer
from torch.nn.functional import normalize

from engine.engine import create_eval_engine
from engine.engine import create_train_engine
from engine.metric import AutoKVMetric
from utils.eval_cmc import eval_rank_list


def get_trainer(model, optimizer, lr_scheduler=None, logger=None, writer=None, non_blocking=False, log_period=10,
                save_interval=10, save_dir="checkpoints", prefix="model", query_loader=None, gallery_loader=None,
                eval_interval=None):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.WARN)

    # trainer
    trainer = create_train_engine(model, optimizer, non_blocking)

    # checkpoint handler
    if not dist.is_initialized() or dist.get_rank() == 0:
        handler = ModelCheckpoint(save_dir, prefix, save_interval=save_interval, n_saved=4, create_dir=True,
                                  save_as_state_dict=True, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler,
                                  {"model": model.module if dist.is_initialized() else model})

    # metric
    timer = Timer(average=True)

    kv_metric = AutoKVMetric()

    # evaluator
    evaluator = None
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if eval_interval > 0 and query_loader and gallery_loader:
        evaluator = create_eval_engine(model, non_blocking)

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
        epoch = engine.state.epoch

        if dist.is_initialized():
            engine.state.dataloader.sampler.set_epoch(epoch)

        engine.state.dataloader.batch_sampler.drop_last = True if epoch > 1 else False

        kv_metric.reset()
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % save_interval == 0:
            logger.info("Model saved at {}/{}_model_{}.pth".format(save_dir, prefix, epoch))

        if evaluator and epoch % eval_interval == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            torch.cuda.empty_cache()

            # extract query feature
            evaluator.run(query_loader)

            q_feats = torch.cat(evaluator.state.feat_list, dim=0)
            q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            q_cam = torch.cat(evaluator.state.cam_list, dim=0).numpy()

            # extract gallery feature
            evaluator.run(gallery_loader)

            g_feats = torch.cat(evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            g_cam = torch.cat(evaluator.state.cam_list, dim=0).numpy()

            distance = -torch.mm(normalize(q_feats), normalize(g_feats).transpose(0, 1)).numpy()
            rank_list = np.argsort(distance, axis=1)

            mAP, r1, r5, _ = eval_rank_list(rank_list, q_ids, q_cam, g_ids, g_cam)

            if writer is not None:
                writer.add_scalar('eval/mAP', mAP, epoch)
                writer.add_scalar('eval/r1', r1, epoch)
                writer.add_scalar('eval/r5', r5, epoch)

            torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        timer.step()

        kv_metric.update(engine.state.output)

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        iter_in_epoch = iteration - (epoch - 1) * len(engine.state.dataloader)

        if iter_in_epoch % log_period == 0:
            batch_size = engine.state.batch[0].size(0)
            speed = batch_size / timer.value()

            if dist.is_initialized():
                speed *= dist.get_world_size()

            msg = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed)

            metric_dict = kv_metric.compute()

            # log output information
            for k in sorted(metric_dict.keys()):
                msg += "\t%s: %.4f" % (k, metric_dict[k])

                if writer is not None:
                    writer.add_scalar('metric/{}'.format(k), metric_dict[k], iteration)

            logger.info(msg)

            kv_metric.reset()
            timer.reset()

    return trainer
