import cv2
import numpy as np
import os
import csv
import time
# import visdom
from visdom import Visdom

topic = "TestUDP"
folder = f"./log/{topic}"
points = list()
linep = list()
img = np.zeros([1000, 1000])
file_save = "3-1.npy"
video_file = "./video/23.4.12/3.mov"
def mouse_callback(event, x, y, flags, param):
    # Check if the left mouse button was clicked

    if event == cv2.EVENT_MOUSEMOVE:
        # Copy the original image to draw the cross lines on
        # print("MOVE",x,y)
        img_draw = img.copy()
        # Draw a cross at the current mouse position
        cv2.line(img_draw, (x-55, y), (x+55, y), (0, 0, 255), 1)
        cv2.line(img_draw, (x, y-55), (x, y+55), (0, 0, 255), 1)
        cv2.imshow('frame', img_draw)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinates of the mouse click
        if points.__len__() < 2:
            points.append([x, y])
            print(f'Mouse clicked at (x={x}, y={y})')

    if event == cv2.EVENT_MBUTTONDOWN:
        pass
        # Print the coordinates of the mouse click
        # if points.__len__() == 2:
        #     GenericBresenhamLine(img,points[0],points[1],(255,0,0))
        # cv2.imshow('frame', img)
        print(linep)


def d_image_xy(II, flag = 'x'):
    g = np.zeros(II.shape,dtype=np.float64)
    # I.dtype= np.dtype("float64")
    I = II.astype('float64')
    if flag=='x':
        g[:,:-1] = I[:,1:]-I[:,:-1]
    elif flag=='y':
        g[:-1,:] = I[1:, :] - I[:-1,:]
    else :
        g=I
    return g


def cvtRGBT(I, w0, w1, w2):
    I = I.astype('float')
    return (w0*I[:, :, 0]+w1*I[:, :, 1]+w2*I[:, :, 2])/3


if __name__ == '__main__':

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', mouse_callback)
    video_name = 3
    vid = cv2.VideoCapture(video_file)


    frame_num = 0
    frame_origin = None
    diff = None
    Plot_frame = None
    diff_temp = None
    diff_temp_gray = None
    laplace_x = None
    laplace_my = None
    frame_gray = None
    mask = None
    diff_mask = None
    diff_mask_gray = None
    diff_mask_gray_cv = None
    last_frame = []
    kpc = None
    is_need_setting = 1
    # 到193行都是预处理，这个需要调，
    while True:
        ret, frame = vid.read()
        if ret==False:
            break
        # frame = cv2.resize(frame,(640,480) )
        # sz_blur = 13  # 原来是33
        # frame = cv2.blur(frame,(sz_blur, sz_blur))
        img = frame.copy()

        img =cv2.transpose(img)
        img = cv2.flip(img, 1)
        img = cv2.medianBlur(img, 3)  # 奇数

        cv2.imshow("frame", img)
        frame_num = frame_num + 1

        if frame_num < 10000000:
            frame_origin = frame.copy()
            if is_need_setting:
                key = cv2.waitKey(0)
                if key == ord('s'):
                    np.save(file_save, points + linep)
                    lomy_list = np.load(file_save)
                    points = [lomy_list[0]]
                    linep = lomy_list[1:]
                    print(lomy_list)
            else:
                lomy_list = np.load(file_save)
                points = [lomy_list[0]]
                linep = lomy_list[1:]
                print(lomy_list)
        else:
            diff = cv2.absdiff(frame, frame_origin)  # 帧差法，作差的，选择要还是不要，一个效果是需要，另外一个是不用
        key = cv2.waitKey(1)
