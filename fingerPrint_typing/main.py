import tkinter
import cv2
import numpy as np
import os
from driver_fpc1020am import DriverFPC1020AM, typing
from find_best_match import judge
import fingerPrint_generate_SIFT




# 一个窗口，实时显示指纹图片，可以控制指纹录入的开启与终止，控制录制哪个指纹，动态显示录制结果
# 可以选择开始识别指纹并打字，并将打字结果实时显示





def main():
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