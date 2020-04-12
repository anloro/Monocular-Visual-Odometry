import cv2
import os
from os.path import isfile, join

pathIn = 'output/videoImDet/'
pathOut = 'output/videoDet.avi'
fps = 30
frame_array = []

# for sorting the file names properly
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort(key=lambda x: x[5:-4])
files.sort()

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

# for sorting the file names properly
files.sort(key=lambda x: x[5:-4])
files.sort()

for i in range(len(files)):
    filename = pathIn + files[i]
    # reading each files
    filename = 'output/videoImDet/objectDetect{}.png'.format(i)
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    # inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
