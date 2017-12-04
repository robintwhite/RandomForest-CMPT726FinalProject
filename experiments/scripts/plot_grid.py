import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np

df = pd.read_csv('../results/grid-results.csv')

cols = list(df.columns)
bestF, bestM, bestMS, bestT = df[df.columns[:4]].iloc[df['TestAccuracy'].idxmax()] #values with best test accuracy
print('Best Test Accuracy values: {} Features, {} Max depth, {} Number of trees'.format(bestF, bestM, bestT))
data_F = df.loc[df[cols[0]] == bestF] #values with best num features
data_M = df.loc[df[cols[1]] == bestM] #values with best max depth
data_T = df.loc[df[cols[3]] == bestT] #values with best num trees

focus = cols[0]
x = data_F[cols[1]].tolist()
y = data_F[cols[3]].tolist()
z = data_F['TestAccuracy'].tolist()
cmap = plt.cm.get_cmap("winter")

xi, yi = np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100)
#xi, yi = np.meshgrid(xi, yi)
# grid the data.
zi = griddata(x, y, z, xi, yi, interp='linear')
# contour the gridded data, plotting dots at the nonuniform data points.
f, (ax1, ax2, ax3) = plt.subplots(1, 3)

CS = ax1.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = ax1.contourf(xi, yi, zi, 15,
                  vmax=abs(zi).max(), vmin= abs(zi).min(), cmap=cmap)
# plot data points.
ax1.scatter(x, y, marker='o', s=5, zorder=10)
#ax1.xlim(min(x), max(x))
#ax1.ylim(min(y), max(y))
ax1.set_xlabel('Maximum Tree Depth')
ax1.set_ylabel('Number of Trees')
ax1.set_title('{} = {}'.format(focus, bestF))
#-------------------------------------------------#
focus = cols[1]
x = data_M[cols[0]].tolist()
y = data_M[cols[3]].tolist()
z = data_M['TestAccuracy'].tolist()
xi, yi = np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100)
zi = griddata(x, y, z, xi, yi, interp='linear')

CS = ax2.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = ax2.contourf(xi, yi, zi, 15,
                  vmax=abs(zi).max(), vmin= abs(zi).min(), cmap=cmap)

# plot data points.
ax2.scatter(x, y, marker='o', s=5, zorder=10)
#ax2.xlim(min(x), max(x))
#ax2.ylim(min(y), max(y))
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Number of Trees')
#ax2.yaxis.set_label_position("right")
ax2.set_title('{} = {}'.format(focus, bestM))
#-------------------------------------------------#
focus = cols[3]
x = data_T[cols[0]].tolist()
y = data_T[cols[1]].tolist()
z = data_T['TestAccuracy'].tolist()
xi, yi = np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100)
zi = griddata(x, y, z, xi, yi, interp='linear')

CS = ax3.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = ax3.contourf(xi, yi, zi, 15,
                  vmax=abs(zi).max(), vmin= abs(zi).min(), cmap=cmap)

# plot data points.
ax3.scatter(x, y, marker='o', s=5, zorder=10)
ax3.set_xlabel('Number of Features')
ax3.set_ylabel('Maximum Tree Depth')
#ax3.yaxis.set_label_position("right")
ax3.set_title('{} = {}'.format(focus, bestT))

#f.colorbar()  # draw colorbar
f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.82, 0.15, 0.01, 0.7])
f.colorbar(CS, cax=cbar_ax)
plt.show()
