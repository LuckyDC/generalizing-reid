import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

mpl.use('agg')


def rank_vis(query_mat, gallery_mat, n_match, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    query_feat = query_mat['feat']
    query_ids = query_mat['ids'].squeeze()
    query_cam_ids = query_mat['cam_ids'].squeeze()
    query_img_paths = query_mat['img_path'].tolist()

    gallery_feat = gallery_mat['feat']
    gallery_ids = gallery_mat['ids'].squeeze()
    gallery_cam_ids = gallery_mat['cam_ids'].squeeze()
    gallery_img_paths = gallery_mat['img_path'].tolist()

    dist = -np.dot(normalize(query_feat), normalize(gallery_feat).T)

    rank_list = np.argsort(dist, axis=1)

    num_query = len(query_ids)
    num_gallery = len(gallery_ids)

    for i in tqdm(range(num_query)):
        index = np.arange(num_gallery)

        intra_flag = np.equal(gallery_cam_ids, query_cam_ids[i])
        inter_flag = np.not_equal(gallery_cam_ids, query_cam_ids[i])
        junk_flag = np.logical_or(np.equal(gallery_ids, -1),
                                  np.logical_and(intra_flag, np.equal(gallery_ids, query_ids[i])))

        junk_index = index[junk_flag]
        # intra_index = index[intra_flag]
        # inter_index = index[inter_flag]

        sort_index = rank_list[i]
        flag = np.logical_not(np.isin(sort_index, junk_index))
        # intra_flag = np.isin(sort_index, intra_index)
        # inter_flag = np.isin(sort_index, inter_index)

        sort_index = sort_index[flag]
        # intra_sort_index = sort_index[np.logical_and(flag, intra_flag)]
        # inter_sort_index = sort_index[np.logical_and(flag, inter_flag)]

        plt.figure(figsize=(4 * n_match, 10))

        q_img_path = query_img_paths[i].strip()
        q_img = cv2.imread(q_img_path)
        q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
        q_img = cv2.resize(q_img, (128, 256))

        # --------------------------------- #
        plt.subplot(1, n_match + 1, 1)

        plt.imshow(q_img)
        # plt.text(30, -8, 'cam %d' % query_cam_ids[i], size=45)
        plt.xticks([])
        plt.yticks([])

        # # --------------------------------- #
        # plt.subplot(3, n_match + 1, 1 + n_match + 1)

        # plt.imshow(q_img)
        # plt.text(30, -8, 'cam %d' % query_cam_ids[i], size=45)
        # plt.xticks([])
        # plt.yticks([])

        # # --------------------------------- #
        # plt.subplot(3, n_match + 1, 1 + (n_match + 1) * 2)

        # plt.imshow(q_img)
        # plt.text(30, -8, 'cam %d' % query_cam_ids[i], size=45)
        # plt.xticks([])
        # plt.yticks([])

        for j in range(n_match):
            plt.subplot(1, n_match + 1, j + 2)

            g_img_path = gallery_img_paths[sort_index[j]].strip()
            g_img = cv2.imread(g_img_path)
            g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)

            # plt.text(30, -8, 'cam %d' % gallery_cam_ids[sort_index[j]], size=45)

            plt.imshow(cv2.resize(g_img, (128, 256)))
            plt.xticks([])
            plt.yticks([])

            if query_ids[i] != gallery_ids[sort_index[j]]:
                color = "red"
            else:
                color = "green"

            ax = plt.gca()
            ax.spines['right'].set_linewidth(8)
            ax.spines['top'].set_linewidth(8)
            ax.spines['left'].set_linewidth(8)
            ax.spines['bottom'].set_linewidth(8)
            ax.spines['right'].set_color(color)
            ax.spines['top'].set_color(color)
            ax.spines['left'].set_color(color)
            ax.spines['bottom'].set_color(color)

            # # --------------------------------- #
            # plt.subplot(3, n_match + 1, j + 2 + n_match + 1)

            # g_img_path = gallery_img_paths[intra_sort_index[j]].strip()
            # g_img = cv2.imread(g_img_path)
            # g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)

            # plt.text(30, -8, 'cam %d' % gallery_cam_ids[intra_sort_index[j]], size=45)
            # # plt.text(30, -8, '%.3f' % dist[i][intra_sort_index[j]], size=45)

            # plt.imshow(cv2.resize(g_img, (128, 256)))
            # plt.xticks([])
            # plt.yticks([])

            # if query_ids[i] != gallery_ids[intra_sort_index[j]]:
            #     color = "red"
            # else:
            #     color = "green"

            # ax = plt.gca()
            # ax.spines['right'].set_linewidth(8)
            # ax.spines['top'].set_linewidth(8)
            # ax.spines['left'].set_linewidth(8)
            # ax.spines['bottom'].set_linewidth(8)
            # ax.spines['right'].set_color(color)
            # ax.spines['top'].set_color(color)
            # ax.spines['left'].set_color(color)
            # ax.spines['bottom'].set_color(color)

            # # --------------------------------- #
            # plt.subplot(3, n_match + 1, j + 2 + (n_match + 1) * 2)

            # g_img_path = gallery_img_paths[inter_sort_index[j]].strip()
            # g_img = cv2.imread(g_img_path)
            # g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)

            # plt.text(30, -8, 'cam %d' % gallery_cam_ids[inter_sort_index[j]], size=45)
            # # plt.text(30, -8, '%.3f' % dist[i][inter_sort_index[j]], size=45)

            # plt.imshow(cv2.resize(g_img, (128, 256)))
            # plt.xticks([])
            # plt.yticks([])

            # if query_ids[i] != gallery_ids[inter_sort_index[j]]:
            #     color = "red"
            # else:
            #     color = "green"

            # ax = plt.gca()
            # ax.spines['right'].set_linewidth(8)
            # ax.spines['top'].set_linewidth(8)
            # ax.spines['left'].set_linewidth(8)
            # ax.spines['bottom'].set_linewidth(8)
            # ax.spines['right'].set_color(color)
            # ax.spines['top'].set_color(color)
            # ax.spines['left'].set_color(color)
            # ax.spines['bottom'].set_color(color)

        plt.tight_layout()
        plt.savefig("{}/{}.jpg".format(save_dir, os.path.basename(q_img_path)))
        plt.close()


if __name__ == '__main__':
    import os
    import argparse

    import scipy.io as sio

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=[
                        "market", "duke", "msmt", "cuhk"])
    parser.add_argument("prefix", type=str)
    args = parser.parse_args()

    q_mat_path = os.path.join("features", args.dataset,
                              "query-%s.mat" % args.prefix)
    g_mat_path = os.path.join("features", args.dataset,
                              "gallery-%s.mat" % args.prefix)

    g_mat = sio.loadmat(g_mat_path)
    q_mat = sio.loadmat(q_mat_path)

    save_dir = "./vis/baseline"
    os.makedirs(save_dir, exist_ok=True)

    rank_vis(q_mat, g_mat, n_match=10, save_dir=save_dir)
