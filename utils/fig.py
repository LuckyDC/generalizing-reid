import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
import numpy as np


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.15, 1.03*height, '%s' % float(height),fontsize=8)


x=np.arange(6)
y1=[87.1,87.4,88.1,87.4,87.1,86.3]
y2=[60.5,63.3,64.6,63.9,63.1,62.3]

y3 = [84.6,87.8,88.1,87.1,86.7,85.8]
y4 = [57.2,61.6,64.6,64.3,64.0,63.1]


bar_width=0.32


fig = plt.figure()

ax_1 = SubplotZero(fig, 212)
fig.add_subplot(ax_1)

a = ax_1.bar(x,y1,bar_width,label='Rank-1')
b = ax_1.bar(x+bar_width+0.02,y2,bar_width,label='mAP')
autolabel(a)
autolabel(b)
ax_1.set_ylim(30,90)
ax_1.set_xticks(x+bar_width/2+0.01)
ax_1.set_xticklabels(['5','10','15','20','25','30'])
ax_1.set_xlabel(r"$\alpha$")
ax_1.axis['top'].set_visible(False)
ax_1.axis['right'].set_visible(False)
ax_1.axis['left'].set_axisline_style("->")
ax_1.axis['bottom'].set_axisline_style("->")


ax_2 = SubplotZero(fig, 211)
fig.add_subplot(ax_2)

c = ax_2.bar(x,y3,bar_width,label='Rank-1')
d = ax_2.bar(x+bar_width+0.01,y4,bar_width,label='mAP')
autolabel(c)
autolabel(d)
ax_2.set_ylim(30,90)
ax_2.set_xticks(x+bar_width/2+0.01)
ax_2.set_xlabel(r"$\alpha$")
ax_2.set_xticklabels(['1','3','5','7','9','11'])
ax_2.axis['top'].set_visible(False)
ax_2.axis['right'].set_visible(False)
ax_2.axis['left'].set_axisline_style("->")
ax_2.axis['bottom'].set_axisline_style("->")

ax_2.legend(loc=8, ncol=2,bbox_to_anchor=(0, 1.1, 1, 0))

plt.tight_layout()
plt.show()