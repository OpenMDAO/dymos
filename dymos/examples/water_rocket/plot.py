import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results.csv', index_col=0)
df_pivot = df.pivot('V_w','m_empty')

x = 1e3* df_pivot.columns.levels[1]  # m_empty
y = 1e3* df_pivot.index  # V_w
X,Y = np.meshgrid(x,y)
Z = df_pivot.values

fig, ax = plt.subplots(1,1)
CS = ax.contour(X,Y,Z, levels=np.arange(0,100))
ax.clabel(CS, fontsize=9, inline=1, fmt='%.0f m', manual=True)
ax.set_xlabel('Empty mass (g)')
ax.set_ylabel('Water volume (L)')
plt.show()
