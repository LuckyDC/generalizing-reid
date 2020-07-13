from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from models.resnet import resnet50
from utils.calc_acc import calc_acc


class Model(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, joint_training=False, mix=False, neighbor_mode=1,
                 **kwargs):
        super(Model, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.joint_training = joint_training
        self.mix = mix
        self.neighbor_mode = neighbor_mode

        self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride)
        self.bn_neck = nn.BatchNorm1d(2048)
        self.bn_neck.bias.requires_grad_(False)

        if kwargs.get('eval'):
            return

        self.scale = kwargs.get('scale')

        # ----------- tasks for source domain --------------
        if num_classes is not None:
            self.classifier = nn.Linear(2048, num_classes, bias=False)
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)

        # ----------- tasks for target domain --------------
        if self.joint_training:
            cam_ids = kwargs.get('cam_ids')
            num_instances = kwargs.get('num_instances', None)
            self.neighbor_eps = kwargs.get('neighbor_eps')

            # identities captured by each camera
            uid2cam = zip(range(num_instances), cam_ids)
            self.cam2uid = defaultdict(list)
            for uid, cam in uid2cam:
                self.cam2uid[cam].append(uid)

            # components for neighborhood consistency
            self.exemplar_linear = ExemplarLinear(num_instances, 2048)
            self.nn_loss = NNLoss(dim=1)

            alpha = kwargs.get('alpha')
            self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)

            self.lambd_st = None

    @staticmethod
    def mix_source_target(inputs, beta_dist):
        half_batch_size = inputs.size(0) // 2
        source_input = inputs[:half_batch_size]
        target_input = inputs[half_batch_size:]

        lambd = beta_dist.sample().item()
        mixed_input = lambd * source_input + (1 - lambd) * target_input
        return mixed_input, lambd

    def forward(self, inputs, labels=None, **kwargs):
        if not self.training:
            global_feat = self.backbone(inputs)
            global_feat = self.bn_neck(global_feat)
            return global_feat
        else:
            batch_size = inputs.size(0)
            epoch = kwargs.get('epoch')

            if self.joint_training and self.mix and epoch > 1:
                mixed_st, self.lambda_st = self.mix_source_target(inputs, self.beta_dist)
                inputs = torch.cat([mixed_st, inputs[batch_size // 2:]], dim=0)

            return self.train_forward(inputs, labels, batch_size, **kwargs)

    def train_forward(self, inputs, labels, batch_size, **kwargs):
        epoch = kwargs.get('epoch')

        inputs = self.backbone(inputs)

        if not self.joint_training:  # single domain
            inputs = self.bn_neck(inputs)
            return self.source_train_forward(inputs, labels)
        else:  # cross domain
            half_batch_size = batch_size // 2
            label_s = labels[:half_batch_size]
            input_t = inputs[-half_batch_size:]

            # source task or mixed task
            input_s = inputs[:half_batch_size]
            feat_s = F.batch_norm(input_s, None, None, self.bn_neck.weight, self.bn_neck.bias, True)
            if not self.mix or epoch <= 1:
                loss, metric = self.source_train_forward(feat_s, label_s)
            else:
                loss, metric = self.mixed_st_forward(feat_s, label_s, **kwargs)

                # target task
            feat_t = self.bn_neck(input_t)
            target_loss, target_metric = self.target_train_forward(feat_t, **kwargs)

            # summarize loss and metric
            loss += target_loss
            metric.update(target_metric)

        return loss, metric

    # Tasks for source domain
    def source_train_forward(self, inputs, labels):
        metric_dict = {}

        cls_score = self.classifier(inputs)
        loss = self.id_loss(cls_score.float(), labels)

        metric_dict.update({'id_ce': loss.data,
                            'id_acc': calc_acc(cls_score.data, labels.data, ignore_index=-1)})

        return loss, metric_dict

    # Tasks for target domain
    def target_train_forward(self, inputs, **kwargs):
        metric_dict = {}

        target_batch_size = inputs.size(0)

        epoch = kwargs.get('epoch')
        img_ids = kwargs.get('img_ids')[-target_batch_size:]
        cam_ids = kwargs.get('cam_ids')[-target_batch_size:]

        # inputs = self.dropout(inputs)
        feat = F.normalize(inputs)

        # Set updating momentum of the exemplar memory.
        # Note the momentum must be 0 at the first iteration.
        mom = 0.6
        self.exemplar_linear.set_momentum(mom if epoch > 1 else 0)
        sim = self.exemplar_linear(feat, img_ids).float()

        # ----------------------Neighborhood Constraint------------------------- #

        # Camera-agnostic neighborhood loss
        if self.neighbor_mode == 0:
            loss = self.cam_agnostic_eps_nn_loss(sim, img_ids)
            metric_dict.update({'neighbor': loss.data})

            weight = 0.1 if epoch > 10 else 0
            loss = weight * loss

        # Camera-aware neighborhood loss (intra_loss and inter_loss)
        elif self.neighbor_mode == 1:
            intra_loss, inter_loss = self.cam_aware_eps_nn_loss(sim, cam_ids, img_ids=img_ids, epoch=epoch)
            metric_dict.update({'intra': intra_loss.data, 'inter': inter_loss.data})

            intra_weight = 1.0 if epoch > 10 else 0
            inter_weight = 0.5 if epoch > 30 else 0

            loss = intra_weight * intra_loss + inter_weight * inter_loss

        return loss, metric_dict

    def mixed_st_forward(self, inputs, labels, **kwargs):
        img_ids = kwargs.get('img_ids')[-inputs.size(0):]
        agent = self.exemplar_linear.memory[img_ids]

        cls_score = F.linear(inputs, self.classifier.weight)

        sim_agent = inputs.mul(agent).sum(dim=1, keepdim=True)
        sim_agent = sim_agent.mul(self.classifier.weight.data[labels].norm(dim=1, keepdim=True))
        cls_score = torch.cat([cls_score, sim_agent], dim=1).float()

        virtual_label = labels.clone().fill_(cls_score.size(1) - 1)
        loss = self.lambda_st * self.id_loss(cls_score, labels)
        loss += (1 - self.lambda_st) * self.id_loss(cls_score, virtual_label)

        metric = {'mix_st': loss.data}

        return loss, metric

    def cam_aware_eps_nn_loss(self, sim, cam_ids, **kwargs):
        img_ids = kwargs.get('img_ids')

        sim_exp = torch.exp(sim * self.scale)

        # calculate mask for intra-camera matching and inter-camera matching
        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(), img_ids, cam_ids, sim.device)

        # intra-camera neighborhood loss
        sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
        neighbor_mask_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        num_neighbor_intra = neighbor_mask_intra.sum(dim=1)

        sim_exp_intra = sim_exp * mask_intra
        score_intra = sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)
        score_intra = score_intra.clamp_min(1e-5)
        intra_loss = -score_intra.log().mul(neighbor_mask_intra).sum(dim=1).div(num_neighbor_intra).mean()
        intra_loss -= score_intra.masked_select(mask_instance.bool()).log().mean()

        # inter-camera neighborhood loss
        sim_inter = (sim.data + 1) * mask_inter - 1
        nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        neighbor_mask_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        num_neighbor_inter = neighbor_mask_inter.sum(dim=1)

        sim_exp_inter = mask_inter * sim_exp
        score_inter = sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True)
        score_inter = score_inter.clamp_min(1e-5)
        inter_loss = -score_inter.log().mul(neighbor_mask_inter).sum(dim=1).div(num_neighbor_inter).mean()

        return intra_loss, inter_loss

    def cam_agnostic_eps_nn_loss(self, sim, img_ids):
        mask_instance = torch.zeros_like(sim)
        mask_instance[torch.arange(sim.size(0)), img_ids] = 1

        sim_neighbor = (sim.data + 1) * (1 - mask_instance) - 1
        nearest = sim_neighbor.max(dim=1, keepdim=True)[0]
        neighbor_mask = torch.gt(sim_neighbor, nearest * self.neighbor_eps)
        num_neighbor = neighbor_mask.sum(dim=1)

        score = F.log_softmax(sim * self.scale, dim=1)
        loss = -score.mul(neighbor_mask).sum(dim=1).div(num_neighbor).mean()
        loss -= score.masked_select(mask_instance.bool()).mean()

        return loss

    def compute_mask(self, size, img_ids, cam_ids, device):
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1

        return mask_instance, mask_intra, mask_inter
