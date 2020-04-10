import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D

# Load the trajectory of the camera
camera_traj = np.load('traj.npy')

# Create a plot for representing the trajectory
position_figure = plt.figure()
ax = plt.axes(projection='3d')
r, c = camera_traj.shape

# Normalization
xp = (camera_traj[0][:]/max(abs(camera_traj[0][:]))+1)/2
yp = (camera_traj[1][:]/max(abs(camera_traj[1][:]))+1)/2
zp = (camera_traj[2][:]/max(abs(camera_traj[2][:]))+1)/2
trajN = [xp, yp, zp]

# Plot the points
# for i in range(c):
#     ax.scatter3D(trajN[0][i], trajN[1][i], trajN[2][i])

# Plot the lines
for i in range(c-1):
    xs = [trajN[0][i], trajN[0][i+1]]
    ys = [trajN[1][i], trajN[1][i+1]]
    zs = [trajN[2][i], trajN[2][i+1]]
    line = Axes3D.art3d.Line3D(xs, ys, zs)
    ax.add_line(line)

# Name the axis
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
# cv2.waitKey()
