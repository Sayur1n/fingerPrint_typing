import cv2
import numpy as np




def adjust_contrast(image, alpha=2.0, beta=50):
    """
    调整图像对比度和亮度
    alpha > 1: 增加对比度
    beta > 0: 增加亮度
    """
    # 调整图像对比度
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


  
'''# 显示结果
cv2.imshow('Adjusted Fingerprint', adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
if __name__  == '__main__':
    path = './images/right_1/adjusted/'
    image_files = []
    dir_path = path
    for i in range(19):
        image_name = dir_path + f'img_{i}.jpg'
        image_files.append(image_name)
    images = [cv2.imread(img, 0) for img in image_files]
    adjusted_images = [adjust_contrast(img, alpha = 1.5, beta=-200) for img in images]
    saving_path = './images/right_1/modified/'
    i = 0
    for adjusted_img in adjusted_images:
        print(cv2.imwrite(saving_path + f'img_{i}.jpg', adjusted_img))
        i += 1
 
 # 增强亮度和对比度