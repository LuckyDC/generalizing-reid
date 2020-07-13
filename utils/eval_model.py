import os

import numpy as np

from data import get_test_loader
from utils.eval_cmc import eval_feature


def eval_model(model, dataset_config, image_size, device):
    # extract query feature
    query = get_test_loader(root=os.path.join(dataset_config.root, dataset_config.query),
                            batch_size=512,
                            image_size=image_size,
                            num_workers=16)

    query_feat = []
    query_label = []
    query_cam_id = []
    for data, label, cam_id, _ in query:
        feat = model(data.cuda(non_blocking=True))

        query_feat.append(feat.data.cpu().numpy())
        query_label.append(label.data.cpu().numpy())
        query_cam_id.append(cam_id.data.cpu().numpy())

    query_feat = np.concatenate(query_feat, axis=0)
    query_label = np.concatenate(query_label, axis=0)
    query_cam_id = np.concatenate(query_cam_id, axis=0)

    # extract gallery feature
    gallery = get_test_loader(root=os.path.join(dataset_config.root, dataset_config.gallery),
                              batch_size=512,
                              image_size=image_size,
                              num_workers=16)

    gallery_feat = []
    gallery_label = []
    gallery_cam_id = []
    for data, label, cam_id, _ in gallery:
        feat = model(data.cuda(non_blocking=True))

        gallery_feat.append(feat.data.cpu().numpy())
        gallery_label.append(label)
        gallery_cam_id.append(cam_id)

    gallery_feat = np.concatenate(gallery_feat, axis=0)
    gallery_label = np.concatenate(gallery_label, axis=0)
    gallery_cam_id = np.concatenate(gallery_cam_id, axis=0)

    mAP, r1, r5, r10 = eval_feature(query_feat, gallery_feat, query_label, query_cam_id, gallery_label, gallery_cam_id,
                                    device)
    print('mAP = %f , r1 precision = %f , r5 precision = %f , r10 precision = %f' % (mAP, r1, r5, r10))
