import argparse
import os
import numpy as np
import scipy.io as sio
import torch

from configs.default import dataset_cfg
from data import get_test_loader
from models.model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    parser.add_argument("model_path", type=str)  # TODO compatible for different models
    parser.add_argument("--img-h", type=int, default=256)
    parser.add_argument("--dataset", type=str, default=None)

    args = parser.parse_args()
    model_path = args.model_path
    fname = model_path.split("/")[-1]

    if args.dataset is not None:
        dataset = args.dataset
    else:
        dataset = model_path.split("/")[1]

    prefix = os.path.splitext(fname)[0]

    dataset_config = dataset_cfg.get(dataset)
    image_size = (args.img_h, 128)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model = Model(eval=True, drop_last_stride=True)

    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict, strict=False)
    model.float()
    model.eval()
    model.cuda()

    # extract query feature
    query = get_test_loader(root=os.path.join(dataset_config.root, dataset_config.query),
                            batch_size=512,
                            image_size=image_size,
                            num_workers=16)

    query_feat = []
    query_label = []
    query_cam_id = []
    query_img_path = []
    for data, label, cam_id, img_path, _ in query:
        with torch.autograd.no_grad():
            feat = model(data.cuda(non_blocking=True))

        query_feat.append(feat.data.cpu().numpy())
        query_label.append(label.data.cpu().numpy())
        query_cam_id.append(cam_id.data.cpu().numpy())
        query_img_path.extend(img_path)

    query_feat = np.concatenate(query_feat, axis=0)
    query_label = np.concatenate(query_label, axis=0)
    query_cam_id = np.concatenate(query_cam_id, axis=0)
    print(query_feat.shape)

    dir_name = "features/{}".format(dataset, prefix)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    save_name = "{}/query-{}.mat".format(dir_name, prefix)
    sio.savemat(save_name,
                {"feat": query_feat,
                 "ids": query_label,
                 "cam_ids": query_cam_id,
                 "img_path": query_img_path})

    # extract gallery feature
    gallery = get_test_loader(root=os.path.join(dataset_config.root, dataset_config.gallery),
                              batch_size=512,
                              image_size=image_size,
                              num_workers=16)

    gallery_feat = []
    gallery_label = []
    gallery_cam_id = []
    gallery_img_path = []
    for data, label, cam_id, img_path, _ in gallery:
        with torch.autograd.no_grad():
            feat = model(data.cuda(non_blocking=True))

        gallery_feat.append(feat.data.cpu().numpy())
        gallery_label.append(label)
        gallery_cam_id.append(cam_id)
        gallery_img_path.extend(img_path)

    gallery_feat = np.concatenate(gallery_feat, axis=0)
    gallery_label = np.concatenate(gallery_label, axis=0)
    gallery_cam_id = np.concatenate(gallery_cam_id, axis=0)
    print(gallery_feat.shape)

    save_name = "{}/gallery-{}.mat".format(dir_name, prefix)
    sio.savemat(save_name,
                {"feat": gallery_feat,
                 "ids": gallery_label,
                 "cam_ids": gallery_cam_id,
                 "img_path": gallery_img_path})
