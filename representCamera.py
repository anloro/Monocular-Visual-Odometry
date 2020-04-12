import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
from matplotlib.lines import Line2D

# Load the trajectory of the camera
camera_traj = np.load('output/traj.npy')
env = np.load('output/environm.npy')

# Create a plot for representing the trajectory
traj_figure = plt.figure()
ax3D = plt.axes(projection='3d')
# xy Plot
figxy = plt.figure()
axy = plt.axes()
# xz
fig = plt.figure()
axz = plt.axes()
# yz
fig = plt.figure()
ayz = plt.axes()


r, c = camera_traj.shape

# # Compute maximum for normalization
# maxx = max(abs(camera_traj[0][:]))
# print(maxx)
# maxx = max(abs(env[0][:]))
# print(maxx)
# print(camera_traj[0][:].shape)
# print(env[0][:].shape)
# maxx = max(abs(np.append(camera_traj[0][:], env[0][:])))
# print(maxx)

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
    line3d = Axes3D.art3d.Line3D(xs, ys, zs)
    ax3D.add_line(line3d)
    linexy = Line2D(xs, ys)
    axy.add_line(linexy)
    linexz = Line2D(xs, zs)
    axz.add_line(linexz)
    lineyz = Line2D(ys, zs)
    ayz.add_line(lineyz)


# env2 = np.zeros((3, 1))
# for i in range(c):
#     if env[0][i] < 1:
#         if env[1][i] < 1:
#             if env[2][i] < 1:
#                 el = np.array([[env[0][i]], [env[0][i]], [env[0][i]]])
#                 # print(el.shape)
#                 # print(env2.shape)
#                 env2 = np.concatenate((env2, el), axis=1)

# print(env2.shape)
#

# # Plot the environment
# r, c = env.shape
# # Normalize environment
# xp = (env[0][:]/max(abs(camera_traj[0][:]))+1)/2
# yp = (env[1][:]/max(abs(camera_traj[1][:]))+1)/2
# zp = (env[2][:]/max(abs(camera_traj[2][:]))+1)/2
# envN = [xp, yp, zp]
#
# d = []
# for i in range(1, c // 2 + 1):
#     if c % i == 0:
#         d.append(i)
#
# # print(d)
# for i in range(int(c/d[2])):
#     ax.scatter3D(envN[0][i*d[2]], envN[1][i*d[2]], envN[2][i*d[2]])

# Name the axis
ax3D.set_xlabel('X Label')
ax3D.set_ylabel('Y Label')
ax3D.set_zlabel('Z Label')

axy.set_xlabel('X Label')
axy.set_ylabel('Y Label')
axz.set_xlabel('X Label')
axz.set_ylabel('Z Label')
ayz.set_xlabel('Y Label')
ayz.set_ylabel('Z Label')
plt.show()
# cv2.waitKey()
