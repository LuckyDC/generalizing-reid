import torch
from apex import amp
from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad


def create_train_engine(model, optimizer, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.train()

        data, labels, cam_ids, img_paths, img_ids = batch
        epoch = engine.state.epoch

        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)
        img_ids = img_ids.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()

        loss, metric = model(data, labels,
                             cam_ids=cam_ids,
                             img_ids=img_ids,
                             epoch=epoch)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        return metric

    return Engine(_process_func)


def create_eval_engine(model, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()

        data, label, cam = batch[:3]

        data = data.to(device, non_blocking=non_blocking)

        with no_grad():
            feat = model(data, cam=cam.clone().to(device, non_blocking=non_blocking))

        return feat.data.float().cpu(), label, cam

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])

    return engine
