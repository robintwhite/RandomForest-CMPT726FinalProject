import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

df = pd.read_csv('../results/features-results.csv')
df = df.sort_values(['Features'])
#plot
x = df['Features']
y1 = df['TrainAccuracy']
y2 = df['TestAccuracy']
#ax = df.plot(x='Trees', y=['TrainAccuracy', 'TestAccuracy'])
fig, ax = plt.subplots()
ax.plot(x,y1,'ro',label='Train')
ax.plot(x,y2,'bo',label='Test')
ax.set_ylabel('Accuracy [%]')
ax.set_xlabel('Number of features')
ax.legend()
plt.show()
