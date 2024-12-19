import cv2
import numpy as np
import os

def match_average_gray(images):
    grays = []

    # 1. 读取图像并计算灰度均值
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(gray)
        grays.append((img, gray, mean_gray))
    
    # 2. 计算目标平均灰度值
    target_mean = np.mean([mean for _, _, mean in grays])
    print(f"目标平均灰度值: {target_mean}")

    # 3. 调整灰度值
    adjusted_images = []
    for img, gray, mean in grays:
        adjustment = target_mean - mean
        adjusted_gray = np.clip(gray + adjustment, 0, 255).astype(np.uint8)
        
        # 若需要彩色图像，将灰度图覆盖回RGB通道
        adjusted_images.append(adjusted_gray)

    return adjusted_images

# 示例代码：文件夹中的图像处理
input_folder = "./images_right_5"
output_folder = "./images_right_5"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取文件夹中的所有图像
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
images = [cv2.imread(os.path.join(input_folder, file)) for file in image_files]

# 调整灰度值
adjusted_images = match_average_gray(images)

# 保存结果
for idx, file in enumerate(image_files):
    output_path = os.path.join(output_folder, f"adjusted_{file}")
    cv2.imwrite(output_path, adjusted_images[idx])
    print(f"保存图像到: {output_path}")
