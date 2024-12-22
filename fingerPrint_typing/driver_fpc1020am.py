# !/user/bin/env python3
# -*- coding: utf-8 -*-
import mmap
import time
import struct
import numpy as np
import subprocess
from pathlib import Path
import cv2
import os
# from .image_stream_viewer import ImageStreamViewer
from find_best_match import judge

class DriverFPC1020AM:
    DRIVER_EXE_NAME = "DriverFPC1020AM.exe"
    SHARED_MEM_NAME = "FPC1020AM_SHARED_MEMORY_0"
    RUN_SERVER_COMMAND = [str(Path(__file__).parents[0] / 'sensor' / DRIVER_EXE_NAME)] + ['0']    ###################
    print(RUN_SERVER_COMMAND)
    KILL_SERVER_COMMAND = ["taskkill", "/im", DRIVER_EXE_NAME, "/f"]

    HEAD_OFFSET = 12
    RAW_WIDTH = 242
    RAW_HEIGHT = 266
    RAW_SIZE = RAW_WIDTH * RAW_HEIGHT
    IMG_WIDTH = 192
    IMG_HEIGHT = 192

    def __init__(
        self,
        live_preview: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the driver.

        Parameters:
            live_preview: Whether to show the live preview.
            verbose: Whether to print verbose information.
        """
        #print(self.RUN_SERVER_COMMAND)
        self.verbose = verbose

        self.start_fpc1020am_server()
        self.img_receiver = self._img_receiver_from_shared_memory()
        self.viewer = None
        #while 1:
            #img = self.get_image()
            #if img is not None:
                #img = img.astype(np.uint8)
                #print(img)
                #cv2.imshow("Image",img)
            #time.sleep(0.1)
        #self.viewer = ImageStreamViewer() if live_preview else None

    def start_fpc1020am_server(self):
        """
        Start the FPC1020AM device server.
        """

        # Kill the server if it is already running.
        self.stop_fpc1020am_server()

        # Start the server.
        attempt = 0
        while attempt < 3:
            proc = subprocess.Popen(
                self.RUN_SERVER_COMMAND,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )

            time.sleep(3)
            ret_code = proc.poll()
            if ret_code is None:
                print("FPC1020AM server started successfully.")
                break

            print(f"Failed to start FPC1020AM server. Retrying...")
            attempt += 1

    def stop_fpc1020am_server(self):
        """
        Stop the FPC1020AM device server.
        """

        subprocess.run(self.KILL_SERVER_COMMAND, stderr=subprocess.DEVNULL)

    def _img_receiver_from_shared_memory(self):
        """
        A generator that receives image from shared memory.

        Yields:
            np.ndarray: The image.
        """

        shared_mem = mmap.mmap(
            -1, self.HEAD_OFFSET + self.RAW_SIZE, self.SHARED_MEM_NAME
        )

        last_frame_idx = -1
        while True:
            img = np.frombuffer(shared_mem, dtype=np.uint8)
            # print(img)

            _, _, frame_idx = struct.unpack("<iii", img[: self.HEAD_OFFSET])
            if frame_idx == last_frame_idx or frame_idx < 3:  # Skip the first frames.
                yield None
                continue

            if self.verbose:
                print(f"Frame index: {frame_idx}")

            img = img[self.HEAD_OFFSET : self.HEAD_OFFSET + self.RAW_SIZE]
            img = img.reshape((self.RAW_HEIGHT, self.RAW_WIDTH))
            img = img[: self.IMG_HEIGHT, : self.IMG_WIDTH]

            last_frame_idx = frame_idx

            yield img.copy()  # Copy is essential to avoid memory bug.

    def get_image(self):
        """
        Get the image from shared memory.

        Returns:
            np.ndarray: The image.
        """

        img = next(self.img_receiver)
        if self.viewer is not None:
            self.viewer.feed(img)
        return img

    def __del__(self):
        self.stop_fpc1020am_server()

if __name__ == "__main__":
    t = DriverFPC1020AM()
    j = 0
    img_array = np.zeros((192,192))
    temp_img = None
    temp_img_judged = False
    is_typing = False
    is_typing_False_counts = 0
    finger_count = 0
    finger1 = 0
    finger2 = 0
    while True:
        img = t.get_image()
        #time.sleep(0.2)
        #print(is_typing)
        if img is not None:
            # 确保 img 转为 NumPy 数组
            img_array = np.array(img, dtype=np.uint8)
            
            temp_img = img_array
            temp_img_judged = False
            is_typing = True
            is_typing_False_counts = 0
        else:
            is_typing = False
            is_typing_False_counts += 1
    
        #print(is_typing_False_counts)
        if temp_img is not None:

            # 打印图像尺寸
            #print(f"Image shape: {img_array.shape}")
            
            # 显示图像
            cv2.imshow("Image", temp_img)
            # 检测键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):  # 按下 's' 键保存图片
                save_dir = "./images/right_4/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print(cv2.imwrite(f'./images_right_3/img_{j}.jpg', temp_img))
                    
                #cv2.imwrite("D:/往期文档/文档/大三上/图像处理/大作业/new/sensor/images/right_2/111.jpg", img_array)
                print(f"Image saved as ./images_to_test/img_{j}.jpg")
                j += 1
                #img_to_match = img
                #finger_locating(img)
            elif key == ord("q"):  # 按下 'q' 键退出循环
                print("Exiting...")
                break
            elif key == ord("j"):
                judge(temp_img)
            
            if not is_typing and is_typing_False_counts>= 50 and not temp_img_judged:
                finger = judge(temp_img)
                temp_img_judged = True
                if finger is not None:
                    finger += 1 
                    if finger1 == 0:
                        finger1 = finger
                        finger_count = 1
                        print(f'finger1 = {finger1}请输入第二个指纹')
                    elif finger2 == 0:
                        finger2 = finger
                        finger_count = 2
                        print(f'finger2 = {finger2}')
                    if finger_count == 2:
                        finger1,finger2 = typing(finger1,finger2)
                else:
                    print('识别该指纹失败，请重新输入！')
            

    # 释放资源
    cv2.destroyAllWindows()
