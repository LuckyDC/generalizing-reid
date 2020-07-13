import matplotlib.pyplot as plt
import numpy as np

x=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

d2m_map = [0.233131, 0.317582, 0.486466, 0.656447, 0.715028, 0.673071, 0.585358, 0.527190]
d2m_r1 = [0.472981, 0.570962, 0.724762, 0.843230, 0.880641, 0.880344, 0.855701, 0.834620]
m2d_map = [0.182909, 0.317147, 0.448963, 0.618837, 0.652238, 0.650040, 0.596468, 0.471318]
m2d_r1 = [0.315978, 0.494614, 0.640036, 0.770646, 0.794883, 0.798923, 0.765709, 0.717684]

d2m_map = np.array(d2m_map) * 100
d2m_r1 = np.array(d2m_r1) * 100
m2d_map = np.array(m2d_map) * 100
m2d_r1 = np.array(m2d_r1) * 100

fig = plt.figure()

ax_1 = plt.subplot(121)
fig.add_subplot(ax_1)

ax_1.set_xticks(x)
l1 = ax_1.plot(x,d2m_r1, marker="o", linewidth=2, label=r"Duke$\rightarrow$ Market")[0]
l2 = ax_1.plot(x,m2d_r1, marker="o", linewidth=2, label=r"Market$\rightarrow$ Duke")[0]
#ax_1.set_ylim(70,95)
ax_1.set_xlabel(r'$\alpha$')
ax_1.set_ylabel('Rank-1 accuracy (%)')
ax_1.grid()


ax_2 = plt.subplot(122)
fig.add_subplot(ax_2)

ax_2.set_xticks(x)
ax_2.plot(x,d2m_map, marker="o", linewidth=2,label=r"Duke$\rightarrow$ Market")
ax_2.plot(x,m2d_map, marker="o", linewidth=2,label=r"Market$\rightarrow$ Duke")
#ax_2.set_ylim(50,70)
ax_2.set_xlabel(r'$\alpha$')
ax_2.set_ylabel('mAP (%)')
ax_2.grid()

fig.legend([l1, l2], labels=[r"Duke$\rightarrow$ Market", r"Market$\rightarrow$ Duke"], ncol=2, loc=9, bbox_to_anchor=(0, 1.02, 1, 0))

plt.show()


