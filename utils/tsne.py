import os
import random
import numpy as np
import scipy.io as sio
import matplotlib as mpl

mpl.use('agg')

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter

k = 25

random.seed(0)
plt.figure(figsize=(7.5, 3.5))

source_features_path = 'features/duke/gallery-duke2market-all-eps0.8-nomix_model_70.mat'
target_features_path = 'features/market/gallery-duke2market-all-eps0.8-nomix_model_70.mat'

print('Loading...')
source_mat = sio.loadmat(source_features_path)
target_mat = sio.loadmat(target_features_path)
print('Done!')

source_features = source_mat["feat"]
source_ids = source_mat["ids"].squeeze()
source_cam_ids = source_mat["cam_ids"].squeeze()
source_img_paths = source_mat['img_path']

target_features = target_mat["feat"]
target_ids = -target_mat["ids"].squeeze()
target_cam_ids = target_mat["cam_ids"].squeeze()
target_img_paths = target_mat['img_path']

s_counter = Counter(source_ids)
t_counter = Counter(target_ids)

s_select_ids = []
t_select_ids = []
for idx, num in s_counter.items():
    if 30 < num < 60 and idx not in [0, -1]:
        s_select_ids.append(idx)
for idx, num in t_counter.items():
    if 30 < num < 60 and idx not in [0, -1]:
        t_select_ids.append(idx)

assert len(s_select_ids) >= k
assert len(t_select_ids) >= k

s_select_ids = random.sample(s_select_ids, k)
t_select_ids = random.sample(t_select_ids, k)

s_flags = np.in1d(source_ids, s_select_ids)
t_flags = np.in1d(target_ids, t_select_ids)

s_ids = source_ids[s_flags]
t_ids = target_ids[t_flags]

ids = np.concatenate([s_ids, t_ids], axis=0).tolist()

id_map = dict(zip(s_select_ids + t_select_ids, range(2 * k)))

new_ids = []
for x in ids:
    new_ids.append(id_map[x])

s_feats = source_features[s_flags]
t_feats = target_features[t_flags]
feats = np.concatenate([s_feats, t_feats], axis=0)

tsne = TSNE(n_components=2, random_state=0)
proj = tsne.fit_transform(feats)

ax = plt.subplot(121)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

t_size = t_feats.shape[0]
s_size = s_feats.shape[0]
ax.scatter(proj[-t_size:, 0], proj[-t_size:, 1], c=['b'] * t_size, marker='.')
ax.scatter(proj[:s_size, 0], proj[:s_size, 1], c=['r'] * s_size, marker='.')

# --------------------------------------------------------------------- #
source_features_path = 'features/duke/gallery-duke2market-all-eps0.8-mix0.6_model_70.mat'
target_features_path = 'features/market/gallery-duke2market-all-eps0.8-mix0.6_model_70.mat'

print('Loading...')
source_mat = sio.loadmat(source_features_path)
target_mat = sio.loadmat(target_features_path)
print('Done!')

source_features = source_mat["feat"]
source_ids = source_mat["ids"].squeeze()
source_cam_ids = source_mat["cam_ids"].squeeze()

target_features = target_mat["feat"]
target_ids = -target_mat["ids"].squeeze()
target_cam_ids = target_mat["cam_ids"].squeeze()

s_flags = np.in1d(source_ids, s_select_ids)
t_flags = np.in1d(target_ids, t_select_ids)

s_ids = source_ids[s_flags]
t_ids = target_ids[t_flags]

ids = np.concatenate([s_ids, t_ids], axis=0).tolist()

new_ids = []
for x in ids:
    new_ids.append(id_map[x])

s_feats = source_features[s_flags]
t_feats = target_features[t_flags]
feats = np.concatenate([s_feats, t_feats], axis=0)

tsne = TSNE(n_components=2, random_state=0)
proj = tsne.fit_transform(feats)

ax = plt.subplot(122)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

ax.scatter(proj[-t_size:, 0], proj[-t_size:, 1], c=['b'] * t_size, marker='.')
ax.scatter(proj[:s_size, 0], proj[:s_size, 1], c=['r'] * s_size, marker='.')

plt.tight_layout()
plt.savefig('tsne.pdf')

s_paths = source_img_paths[s_flags]
t_paths = target_img_paths[t_flags]
for path in s_paths.tolist():
    os.system('cp %s /home/chuanchen_luo/vis' % path)
for path in t_paths.tolist():
    os.system('cp %s /home/chuanchen_luo/vis' % path)
