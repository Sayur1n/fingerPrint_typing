import tkinter
import cv2
import numpy as np
import os
from driver_fpc1020am import DriverFPC1020AM, typing
from find_best_match import judge
import time
import fingerPrint_generate_SIFT

average = 127
# 一个窗口，实时显示指纹图片，可以控制指纹录入的开启与终止，控制录制哪个指纹，动态显示录制结果
# 可以选择开始识别指纹并打字，并将打字结果实时显示


def main():
    t = DriverFPC1020AM()  # 创建传感器对象
    j = 0
    img_array = np.zeros((192, 192))
    temp_img = None  # 当前读取图像
    temp_img_judged = False  # 当前读取图像是否已经完成输入
    is_typing = False  # 是否正在输入
    is_typing_False_counts = 0  # 连续未输入次数
    finger_count = 0  # 已输入手指数
    finger1 = 0  # 输入的第一个手指识别码
    finger2 = 0  # 输入的第二个手指识别码
    patten = True  # 工作模式，True表示打字，False表示录入使用者指纹，默认为打字模式
    time1 = 0
    time2 = 0
    hand = "left"
    place = 1
    f = 1
    region = 0
    while True:
        img = t.get_image()
        # time.sleep(0.2)
        # print(is_typing)

        # 只有读取的图片非空才对它进行处理
        if img is not None:
            # 确保 img 转为 NumPy 数组
            img_array = np.array(img, dtype=np.uint8)
            # # 将图片像素平均值改到同一水平
            # mean = img_array.mean()
            # gap = average - round(mean)
            # img_array = img_array + gap
            #
            temp_img = img_array
            temp_img_judged = False
            is_typing = True
            is_typing_False_counts = 0
            time1 = time.perf_counter()  # 记录最新照片的获取时间
        else:
            is_typing = False
            is_typing_False_counts += 1

        # print(is_typing_False_counts)

        # 录入输入者指纹图像
        if temp_img is not None:
            # 显示图像
            cv2.imshow("Image", temp_img)
            # 检测键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # 按下 'q' 键退出循环
                print("Exiting...")
                break
            elif key == ord("e"):  # 按下'e'切换工作模式
                patten = not patten
                print(patten)
                # img_to_match = img
                # finger_locating(img)
            elif not patten and key == ord("s"):  # 录入模式下，按下 's' 键保存图片
                save_dir = f"./images/finger_imgs/{region}/"  # 将这里的路径改为自己的目标路径
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print(cv2.imwrite(f'./images/finger_imgs/{region}/img_{j}.jpg', temp_img))
                j += 1
            elif not patten and key == ord('p'):  # 录入模式下，按下'p'键更改录入指纹的目标手指，手指所代表的编号由使用者定义
                print("请录入目标手指区域：")
                region = int(input())
                if region < 0:
                    region = 0
                elif region > 6:
                    region = 6
                # hand = int(input("请选择左手还是右手："))
                # f = int(input("请选择手指："))
                # place = int(input("请选择手指部位："))
                # if hand != 0:
                #     hand = "right"
                # else:
                #     hand = "left"
                # if f > 5:
                #     f = 5
                # elif f < 1:
                #     f = 1
                # if place > 3:
                #     place = 3
                # elif place < 1:
                #     place = 1
                j = 0  # 重新开始对图像序列进行计数
            elif key == ord("j"):
                judge(temp_img)

            # 考虑改进————引入时间模块，用时间为停止输入的判据

            # 在输入模式下
            # if patten and is_typing_False_counts >= 50 and not temp_img_judged:
            if patten and time.perf_counter() - time1 > 0.5 and not temp_img_judged:
                finger = judge(temp_img)
                temp_img_judged = True
                if finger is not None:
                    finger += 1
                    if finger1 == 0:
                        finger1 = finger
                        finger_count = 1
                        # print(f'finger1 = {finger1}请输入第二个指纹')
                    elif finger2 == 0:
                        finger2 = finger
                        finger_count = 2
                        # print(f'finger2 = {finger2}')
                    if finger1 == 6 or finger2 == 6:  # 这里可以设置一个清空缓存区的功能
                        finger1 = 0
                        finger2 = 0
                    if finger_count == 2:
                        finger1, finger2 = typing(finger1, finger2)
                else:
                    finger1 = 0
                    finger2 = 0
                    print('字母识别失败，请重新输入下一字母编码')

    # 释放资源
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
