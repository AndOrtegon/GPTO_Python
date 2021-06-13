import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections

verts = np.array( ( ((0,1),(0.8,0.6),(2,2),(0.9,0)),) )/2
print( verts.shape )
plt.ion()
fig = plt.figure(2)
ax = plt.gca()

pc  = matplotlib.collections.PolyCollection(verts,  cmap='gray' )
pc.set_array(np.array((0.3,0.5)))
ax.add_collection(pc)
plt.gca().set_aspect('equal', adjustable='box')
plt.pause(10)
plt.draw()
fig.canvas.manager.window.wm_geometry("+0+0")
