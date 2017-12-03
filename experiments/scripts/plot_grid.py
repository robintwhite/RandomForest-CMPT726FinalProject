import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np

df = pd.read_csv('../results/grid-results.csv')

cols = list(df.columns)
bestF, bestM, bestMS, bestT = df[df.columns[:4]].iloc[df['TestAccuracy'].idxmax()] #values with best test accuracy

data_F = df.loc[df[cols[0]] == bestF] #values with best num features
data_M = df.loc[df[cols[1]] == bestM] #values with best max depth
data_T = df.loc[df[cols[3]] == bestT] #values with best num trees

#TODO for best num features, plot max depth vs trees vs Test accuracy and train accuracy
focus = cols[0]
x = data_F[cols[1]].tolist()
y = data_F[cols[3]].tolist()
z = data_F['TestAccuracy'].tolist()

xi, yi = np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100)
#xi, yi = np.meshgrid(xi, yi)
# grid the data.
zi = griddata(x, y, z, xi, yi, interp='linear')
# contour the gridded data, plotting dots at the nonuniform data points.
CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = plt.contourf(xi, yi, zi, 15,
                  vmax=abs(zi).max(), vmin=-abs(zi).max())
plt.colorbar()  # draw colorbar
# plot data points.
plt.scatter(x, y, marker='o', s=5, zorder=10)
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.title(focus)
plt.show()
