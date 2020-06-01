import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y, 'r--')
plt.grid(True)
plt.legend()
plt.show()
