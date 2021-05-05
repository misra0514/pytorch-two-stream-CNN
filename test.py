import cv2
from preprocess import Preprocess as pre
import visualize
import numpy as np
import time
from visdom import Visdom


if __name__ == '__main__':

    #vis = Visdom(port=8097)
    plot_loss = visualize.line('Loss', port=8097)
    plot_loss.register_line('Loss', 'iter', 'loss')





    # video = "hmdb51_org/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi"
    # frame_save_path = "data"
    # folder_name, video_name= video.split('/')[-2], video.split('/')[-1]
    # capture = cv2.VideoCapture(video)
    # ret, image = capture.read()
    # fname = '/{0}_{1:05d}.jpg'.format(video_name, 1)
    # cv2.imwrite('{}/{}/{}'.format(frame_save_path, folder_name, fname), image)

    # _, frame = capture.read()
    #pre.video2frame("hmdb51_org/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi","data","data")
