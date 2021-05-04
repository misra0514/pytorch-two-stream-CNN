import os
import glob
import cv2
import random
import numpy as np


class Preprocess():
    def __init__(self, ratio=.7):
        self.ratio = ratio
        
        # 这一步获得一个标签labels，标签就是文件夹的名字，并且根据字典序排序
        # lables 共51个分类。为一个主键-string的字典
        labels = sorted(os.listdir(os.path.join(os.getcwd(), 'data/hmdb51_org')))

        # vidio_list 是一个长对应lables的二维list，其实就是按照数据集中的原始分类进行读取的数据。
        video_list = []
        for label in labels:
            video_list.append([avi for avi in glob.iglob('data/hmdb51_org/{}/*.avi'.format(label), recursive=True)])

        # 레이블 인덱싱--标签索引
        label_index = {label : np.array(i) for i, label in enumerate(labels)}
            
        # 데이터 전처리--数据预处理 (video -> image)
        if not os.path.exists('data/train'):
            for label in labels:
                os.makedirs(os.path.join(os.getcwd(), 'data/train/image', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/test/image', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/train/optical', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/test/optical', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/val/image', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/val/optical', label), exist_ok=True)
            
            # 下面开始，按照一定的比例构建测试集和训练集。ratio由开始传入
            # 这步还时间挺久的。每一个大约半s。每一个视频都要抽取。下次使用可以调整一下print进度，或者干脆跳过这步。最后使用cv2写入了新文件夹
            # 话说好像如果已经存在的话就不再写入了。删除试试？或者更改一下逻辑
            counter = 0
            for videos in video_list:
                for i, video in enumerate(videos):
                    print("当前进度"+str(i) + "/" + str(len(videos)) + ";总进度" + str(counter) + "/" + str(len(video_list)))
                    # train
                    if i < round(len(videos)*self.ratio):
                        self.video2frame(video, 'data/train/image', 'data/train/optical')

                    # validation
                    elif i > round(len(videos)*0.9):
                        self.video2frame(video, 'data/val/image/', 'data/val/optical/')

                    # test
                    else:
                        self.video2frame(video, 'data/test/image', 'data/test/optical')
                counter+=1


    # 函数作用： 抽取视频中部分帧中的图像保存起来
    # 参数列表： 视频路径；帧存放路径，optical存放路径【暂未知用处】
    def video2frame(self, video, frame_save_path,optical_save_path, count=0):
        '''
            1개의 동영상 파일에서 약 16 프레임씩 이미지(.jpg)로 저장
            1从一个视频中保存约16帧作为图像
    
            args
                video : 비디오 파일 이름
                vidio : 视频文件名
                save_path : 저장 경로
                save_path : 存放路径
    
        '''
        folder_name, video_name= video.split('/')[-2], video.split('/')[-1]

        # capture生成了一个video路径下的视频对象
        capture = cv2.VideoCapture(video) 
        if(not capture.isOpened()):
            print("capture video failed")
        get_frame_rate = round(capture.get(cv2.CAP_PROP_FRAME_COUNT) / 16) # 帧数/16（获取视频率）

        _, frame = capture.read()
        prvs = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        hsv = np.zeros_like(frame) # Farneback 알고리즘 이용하기 위한 초기화--初始化使用算法
        hsv[..., 1] = 255 # 초록색 바탕 설정--绿色背景设置

        while True:
            # image 是读取到的一帧图像。
            # fixed : 这里不会出现read问题。img可以正常获取
            ret, image = capture.read()
            if not ret:
                print("read IMG failed of End of File")
                break

            # get(0) : 基于以0开始的被捕获或解码的帧索引 ; 能否整除这个是用来采样的
            if(int(capture.get(1)) % get_frame_rate == 0):
                count += 1
                fname = '/{0}_{1:05d}.jpg'.format(video_name, count) # 输出文件名字
                # FIXED : 这里不知道为什么原作者要写一个floder_name在这里。导致了路径不存在。接下来要改一下路径编码方式
                # if(cv2.imwrite('{}/{}/{}'.format(frame_save_path, folder_name, fname), image) == False):
                #     print("imwrite error")
                if(cv2.imwrite('{}/{}'.format(frame_save_path, fname), image) == False):
                    print("imwrite error")

                # 下面开始计算光流
                next_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # 可以看出光流的计算是使用 using the @cite Farneback2003 algorithm 实现的（cv2的内置函数
                # An example using the optical flow algorithm described by Gunnar Farneback can be found at opencv_source_code/samples/python/opt_flow.py
                flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                fname = '/{0}_{1:05d}_flow.jpg'.format(video_name, count)
                # 同理，这边也应该做一个修改。把已经写好的图像写入硬盘
                #cv2.imwrite('{}/{}/{}'.format(optical_save_path, folder_name, fname), rgb)
                if(cv2.imwrite('{}/{}'.format(optical_save_path, fname), rgb) == False):
                    print("optical imwrite error")

            prvs = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)


        print("{} images are extracted in {}.". format(count, frame_save_path))
