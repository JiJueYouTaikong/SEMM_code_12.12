import numpy as np
data = np.load("taxi_pick/train.npz")
print(data['x'].shape)
print(data['x_offsets'].shape)
print(data['x_offsets'])
print(data['y'].shape)
print(data['y_offsets'].shape)
print(data['y_offsets'])