import logging

import numba
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@numba.jit
def compute_ap(good_index, junk_index, sort_index):
    cmc = np.zeros((sort_index.shape[0],))
    n_good = good_index.shape[0]

    old_recall = 0
    old_precision = 1.0
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    n_junk = 0
    for i in range(sort_index.shape[0]):
        flag = 0
        if np.any(good_index == sort_index[i]):
            cmc[i - n_junk:] = 1
            flag = 1
            good_now = good_now + 1

        if np.any(junk_index == sort_index[i]):
            n_junk = n_junk + 1
            continue

        if flag == 1:
            intersect_size = intersect_size + 1

        recall = intersect_size / n_good
        precision = intersect_size / (j + 1)
        ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        j = j + 1

        if good_now == n_good:
            break

    return ap, cmc


def eval_feature(query_features, gallery_features, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, device):
    if isinstance(gallery_features, np.ndarray):
        gallery_features = torch.from_numpy(gallery_features)

    if isinstance(query_features, np.ndarray):
        query_features = torch.from_numpy(query_features)

    gallery_features = gallery_features.to(device)
    query_features = query_features.to(device)

    num_query = query_ids.shape[0]
    num_gallery = gallery_ids.shape[0]

    gallery_features = F.normalize(gallery_features, p=2, dim=1)
    query_features = F.normalize(query_features, p=2, dim=1)

    dist_array = -torch.mm(query_features, gallery_features.transpose(0, 1)).cpu().numpy()

    ap = np.zeros((num_query,))  # average precision
    cmc = np.zeros((num_query, num_gallery))

    index = np.arange(num_gallery)
    for i in tqdm(range(num_query)):
        good_flag = np.logical_and(np.not_equal(gallery_cam_ids, query_cam_ids[i]), np.equal(gallery_ids, query_ids[i]))
        junk_flag_1 = np.equal(gallery_ids, -1)
        junk_flag_2 = np.logical_and(np.equal(gallery_cam_ids, query_cam_ids[i]),
                                     np.equal(gallery_ids, query_ids[i]))

        good_index = index[good_flag]
        junk_index = index[np.logical_or(junk_flag_1, junk_flag_2)]

        dist = dist_array[i]

        sort_index = np.argsort(dist)

        ap[i], cmc[i, :] = compute_ap(good_index, junk_index, sort_index)

    mAP = np.mean(ap)
    r1 = np.mean(cmc, axis=0)[0]
    r5 = np.mean(np.clip(np.sum(cmc[:, :5], axis=1), 0, 1), axis=0)
    r10 = np.mean(np.clip(np.sum(cmc[:, :10], axis=1), 0, 1), axis=0)

    logging.info('mAP = %f , r1 precision = %f , r5 precision = %f , r10 precision = %f' % (mAP, r1, r5, r10))

    return mAP, r1, r5, r10


def eval_rank_list(rank_list, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    num_query = len(query_ids)
    num_gallery = len(gallery_ids)

    ap = np.zeros((num_query,))  # average precision
    cmc = np.zeros((num_query, num_gallery))
    for i in tqdm(range(num_query)):
        index = np.arange(num_gallery)
        good_flag = np.logical_and(np.not_equal(gallery_cam_ids, query_cam_ids[i]), np.equal(gallery_ids, query_ids[i]))
        junk_flag_1 = np.equal(gallery_ids, -1)
        junk_flag_2 = np.logical_and(np.equal(gallery_cam_ids, query_cam_ids[i]),
                                     np.equal(gallery_ids, query_ids[i]))

        good_index = index[good_flag]
        junk_index = index[np.logical_or(junk_flag_1, junk_flag_2)]

        sort_index = rank_list[i]

        ap[i], cmc[i, :] = compute_ap(good_index, junk_index, sort_index)

    mAP = np.mean(ap)
    r1 = np.mean(cmc, axis=0)[0]
    r5 = np.mean(np.clip(np.sum(cmc[:, :5], axis=1), 0, 1), axis=0)
    r10 = np.mean(np.clip(np.sum(cmc[:, :10], axis=1), 0, 1), axis=0)

    logging.info('mAP = %f , r1 precision = %f , r5 precision = %f , r10 precision = %f' % (mAP, r1, r5, r10))

    return mAP, r1, r5, r10
