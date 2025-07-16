import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np 

fig, ax = plt.subplots(figsize=(4, 4))
ax.axis('off')  # Removes axis
plt.tight_layout()

imgs = []
directory = os.fsencode("logs/observations")  

for file in sorted(os.listdir(directory)):  # Sort to maintain order
    filename = os.fsdecode(file)
    if filename.endswith(".npy") and not "raw" in filename: 
        data = np.load(os.path.join("logs/observations", filename)).squeeze()
        img = ax.imshow(data, animated=True,cmap= "gray")
        imgs.append([img])  # wrap in list

ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)

ani.save("output.gif", writer='pillow', fps=20)  # No bbox_inches/pad_inches here
