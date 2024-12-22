import cv2
import numpy as np
import time
import os

max_valid_pixel = 255


def get_useful_mask(img):
    # max_pixel = np.max(img)
    useful_mask = img < max_valid_pixel - 3  # 去掉白色边界
    useful_mask = useful_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    useful_mask = cv2.erode(useful_mask, kernel, iterations=1)
    useful_mask = useful_mask > 0.5
    return useful_mask


def create_gaussian_weight_map(height, width, sigma=10):
    """
    这个函数生成一个以图片中心为最高点的二维高斯函数图
    """
    center_x = width // 2
    center_y = height // 2
    y, x = np.ogrid[:height, :width]
    weight_map = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
    return weight_map


def add_to_canvas(canvas, img, offset_x, offset_y, overlap_count, max_overlaps=3):
    """

    canvas:画布对象
    img:待拼接图像
    offset_x:被拼接图像左上角在画布上的横坐标
    offset_y:被拼接图像左上角在画布上的纵坐标
    overlap_count:当前画布各个位置的堆叠次数
    max_overlaps:同一位置允许的最大堆叠次数
    """
    h, w = img.shape
    offset_x, offset_y = int(offset_x), int(offset_y)

    # 判断新拼接的图像是否超出画布范围，若是，扩展画布及堆叠次数图，同时更新拼接位置
    new_h, new_w = canvas.shape
    min_x, max_x = offset_x, offset_x + w
    min_y, max_y = offset_y, offset_y + h
    extend_left = max(0, -min_x)
    extend_right = max(0, max_x - new_w)
    extend_top = max(0, -min_y)
    extend_bottom = max(0, max_y - new_h)
    if extend_left > 0 or extend_right > 0 or extend_top > 0 or extend_bottom > 0:
        # 扩展画布
        new_canvas = np.zeros((new_h + extend_top + extend_bottom,
                               new_w + extend_left + extend_right),
                              dtype=np.float32)
        new_canvas[extend_top:extend_top + new_h, extend_left:extend_left + new_w] = canvas
        canvas = new_canvas
        # 扩展画布对应的堆叠次数图
        new_overlap_count = np.zeros_like(new_canvas, dtype=np.int32)
        new_overlap_count[extend_top:extend_top + new_h, extend_left:extend_left + new_w] = overlap_count
        overlap_count = new_overlap_count
        # 更新拼接位置
        offset_x += extend_left
        offset_y += extend_top

    # 创建高斯权重图
    weight_map = create_gaussian_weight_map(h, w)

    # 创建羽化边缘
    feather_width = 3
    edge_mask = np.ones((h, w), dtype=np.float32)
    edge_mask[:feather_width, :] *= np.linspace(0, 1, feather_width)[:, np.newaxis]
    edge_mask[-feather_width:, :] *= np.linspace(1, 0, feather_width)[:, np.newaxis]
    edge_mask[:, :feather_width] *= np.linspace(0, 1, feather_width)
    edge_mask[:, -feather_width:] *= np.linspace(1, 0, feather_width)

    combined_weight = weight_map * edge_mask

    # 获取目标区域当前值
    target_region = canvas[offset_y:offset_y + h, offset_x:offset_x + w]
    overlap_region = overlap_count[offset_y:offset_y + h, offset_x:offset_x + w]

    # 仅对叠加次数小于 max_overlaps 的区域进行操作                                      ## 尝试在这里去掉白色无用边界
    #useful_mask = get_useful_mask(img)  # 去掉白色边界的mask
    valid_mask = overlap_region < max_overlaps
    overlap_mask = valid_mask & (target_region > 0) & (img > 0) & get_useful_mask(img)
    non_overlap_mask = valid_mask & (target_region == 0) & (img > 0) & get_useful_mask(img)
    #overlap_mask = valid_mask & (target_region > 0) & useful_mask
    #non_overlap_mask = valid_mask & (target_region == 0) & useful_mask

    # 在重叠区域使用加权混合
    if np.any(overlap_mask):
        target_region[overlap_mask] = (
                target_region[overlap_mask] * (1 - combined_weight[overlap_mask])
                + img[overlap_mask] * combined_weight[overlap_mask]
        )

    # 在非重叠区域直接使用新图像的值
    target_region[non_overlap_mask] = img[non_overlap_mask]

    # 更新叠加计数，这里也用上useful_mask
    #overlap_region[valid_mask & useful_mask] += 1
    overlap_region[valid_mask & (img > 0)] += 1

    # 返回更新后的画布和叠加次数图，方便迭代；同时返回此次拼接的位置
    return canvas, offset_x - extend_left, offset_y - extend_top, overlap_count

def extract_valid_region(img):
    """
    提取图像中的非零区域，返回掩码和边界
    """
    mask = (img > 0).astype(np.uint8)
    if not np.any(mask):
        return None, None, None, None

    # 找到非零区域的边界
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    return mask[y:y + h, x:x + w], x, y, (w, h)


def match_with_canvas(canvas, img2, min_matches=5):
    """
    将新图像与画布进行特征匹配，并限制旋转角度和缩放比例
    旋转限制：±30度
    缩放限制：±10%
    返回值：变换阵，画布有效区域偏离坐标
    """
    # 提取画布中的有效区域
    valid_mask, canvas_x, canvas_y, (canvas_w, canvas_h) = extract_valid_region(canvas)
    if valid_mask is None:
        return None, (0, 0)

    canvas_roi = canvas[canvas_y:canvas_y + canvas_h, canvas_x:canvas_x + canvas_w]

    # 创建SIFT特征点检测器
    sift = cv2.SIFT_create()

    # 检测特征点和描述符
    kp1, des1 = sift.detectAndCompute(canvas_roi.astype(np.uint8), None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return None, (0, 0)

    # 创建FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception:
        return None, (0, 0)

    # 应用比率测试筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < min_matches:
        return None, (0, 0)

    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算变换矩阵，使用RANSAC方法
    transform = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC,
                                            ransacReprojThreshold=5.0)[0]

    if transform is None:
        return None, (0, 0)

    # 从变换矩阵中提取旋转角度和缩放比例
    scale_x = np.sqrt(transform[0, 0] ** 2 + transform[0, 1] ** 2)
    scale_y = np.sqrt(transform[1, 0] ** 2 + transform[1, 1] ** 2)
    angle = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi

    # 检查旋转角度和缩放约束
    if (abs(angle) > 30 or  # 限制旋转角度在±30度以内
            abs(scale_x - 1.0) > 0.1 or  # 允许最多10%的缩放误差
            abs(scale_y - 1.0) > 0.1):  # 检查缩放是否接近1
        return None, (0, 0)

    # 调整变换矩阵以考虑ROI偏移
    transform[:, 2] += [canvas_x, canvas_y]

    return transform, (canvas_x, canvas_y)


def check_and_extend_canvas(canvas, overlap_count, transform, img_shape):
    """
    检查并扩展画布，确保变换后的图像完全显示
    """
    h, w = img_shape
    canvas_h, canvas_w = canvas.shape

    # 计算变换后图像的四个角点
    corners = np.array([[0, 0],
                        [w, 0],
                        [w, h],
                        [0, h]], dtype=np.float32)
    transformed_corners = cv2.transform(corners.reshape(1, -1, 2), transform).reshape(-1, 2)

    # 计算变换后图像的边界
    min_x = int(np.floor(transformed_corners[:, 0].min()))
    max_x = int(np.ceil(transformed_corners[:, 0].max()))
    min_y = int(np.floor(transformed_corners[:, 1].min()))
    max_y = int(np.ceil(transformed_corners[:, 1].max()))

    # 计算需要的扩展量
    extend_left = max(0, -min_x)
    extend_right = max(0, max_x - canvas_w)
    extend_top = max(0, -min_y)
    extend_bottom = max(0, max_y - canvas_h)

    # 如果需要扩展画布
    if extend_left > 0 or extend_right > 0 or extend_top > 0 or extend_bottom > 0:
        # 扩展画布
        new_h = canvas_h + extend_top + extend_bottom
        new_w = canvas_w + extend_left + extend_right
        new_canvas = np.zeros((new_h, new_w), dtype=np.float32)
        new_overlap_mask = np.zeros((new_h, new_w), dtype=np.float32)
        new_overlap_count = np.zeros((new_h, new_w), dtype=np.int32)

        # 复制原有内容到新画布
        new_canvas[extend_top:extend_top + canvas_h,
        extend_left:extend_left + canvas_w] = canvas

        new_overlap_count[extend_top:extend_top + canvas_h,
        extend_left:extend_left + canvas_w] = overlap_count
        # 调整变换矩阵以考虑画布扩展
        transform[0, 2] += extend_left
        transform[1, 2] += extend_top

        return new_canvas, new_overlap_count, transform, True

    return canvas, overlap_count, transform, False

# 裁剪掉所有灰度值为0的区域
def crop_non_zero_area(image):
    # 读取图片并转换为灰度图
    img = image

    # 找到灰度值大于0的区域
    coords = cv2.findNonZero(img)  # 返回灰度值大于0的所有坐标点
    if coords is None:
        print("图片中不存在灰度值大于0的区域")
        return

    # 获取包含非零像素的边界框
    x, y, w, h = cv2.boundingRect(coords)

    # 裁剪图片
    cropped_img = img[y:y + h, x:x + w]

    return cropped_img

def single_match(img, img_count, canvas, overlap_count):
    is_matched = False
    if img is not None:
        h, w = img.shape
        if img_count == 0:

            # 将第一张图像放在画布中央
            canvas_center_x = canvas.shape[1] // 2
            canvas_center_y = canvas.shape[0] // 2
            img_x = canvas_center_x - (w // 2)
            img_y = canvas_center_y - (h // 2)
            
            # 添加第一张图像
            canvas, _, _, overlap_count = add_to_canvas(canvas, img, img_x, img_y, overlap_count, 4)
            img_count += 1
            is_matched = True
            return is_matched, img_count, canvas,  overlap_count

        else:
            # 尝试与画布匹配
            transform, (canvas_x, canvas_y) = match_with_canvas(canvas, img)
            
            if transform is not None:
                # 检查并在必要时扩展画布
                canvas, overlap_count, transform, was_extended = check_and_extend_canvas(
                    canvas, overlap_count, transform, img.shape)
                
                # 执行仿射变换
                img_warped = cv2.warpAffine(img, transform, 
                                        (canvas.shape[1], canvas.shape[0]),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
                
                # 添加到画布
                canvas, _, _, overlap_count = add_to_canvas(canvas, img_warped, 0, 0, overlap_count, 4)
                img_count += 1
                is_matched = True

                return is_matched, img_count, canvas, overlap_count
            else:
                return is_matched, img_count, canvas, overlap_count
    else:
        return is_matched, img_count, canvas, overlap_count

def main(read_path, finger):
    # 读取图像
    images = []
    names = os.listdir(read_path)
    for name in names:
        image_name = os.path.join(read_path, name)
        img = cv2.imread(image_name, 0)
        if img is not None:
            images.append(img)

    if images is None:
        print("未能读取任何图像")
        return

    # 创建足够大的初始画布（比第一张图像大三倍）
    h, w = images[0].shape
    canvas = np.zeros((h * 3, w * 3), dtype=np.float32)
    overlap_count = np.zeros_like(canvas, dtype=np.int32)

    # 将第一张图像放在画布中央
    canvas_center_x = canvas.shape[1] // 2
    canvas_center_y = canvas.shape[0] // 2
    img_x = canvas_center_x - (w // 2)
    img_y = canvas_center_y - (h // 2)

    # 添加第一张图像
    canvas, _, _, overlap_count = add_to_canvas(canvas, images[0], img_x, img_y, overlap_count, 4)

    # 记录已成功拼接的图像
    stitched_images = {0}

    # 循环直到所有图像都被尝试拼接
    while len(stitched_images) < len(images):
        progress_made = False

        # 尝试拼接每张未处理的图像
        for idx in range(1, len(images)):
            if idx in stitched_images:
                continue

            # 尝试与画布匹配
            transform, (canvas_x, canvas_y) = match_with_canvas(canvas, images[idx])

            if transform is not None:
                # 检查并在必要时扩展画布
                canvas,  overlap_count, transform, was_extended = check_and_extend_canvas(
                    canvas, overlap_count, transform, images[idx].shape)

                # 执行仿射变换
                img_warped = cv2.warpAffine(images[idx], transform,
                                            (canvas.shape[1], canvas.shape[0]),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)

                # 添加到画布
                canvas, _, _, overlap_count = add_to_canvas(canvas, img_warped, 0, 0, overlap_count, 4)

                stitched_images.add(idx)
                progress_made = True

                # 显示拼接进度
                temp_canvas = canvas.copy()
                #mask = (overlap_mask > 0)
                #temp_canvas[mask] /= overlap_mask[mask]
                temp_display = np.clip(temp_canvas, 0, 255).astype(np.uint8)
                cv2.imshow("Stitching Progress", temp_display)
                cv2.waitKey(1)
                # time.sleep(1)

        if not progress_made:
            break

    # 处理最终结果
    #mask = (overlap_mask > 0)
    #canvas[mask] /= overlap_mask[mask]
    result = np.clip(canvas, 0, 255).astype(np.uint8)

    # 裁剪空白边界，保留一定边距
    gray = result.copy()
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        # 增加边距以确保不会裁剪到图像内容
        margin = 20  # 增加边距到50像素
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(result.shape[1] - x, w + 2 * margin)
        h = min(result.shape[0] - y, h + 2 * margin)
        result = result[y:y + h, x:x + w]

    # 显示并保存结果
    cv2.imshow("Final Stitched Image", result)
    cv2.waitKey(1000)  # 留出10秒确认拼接好的图片
    cv2.destroyAllWindows()
    # 保存拼接结果
    cv2.imwrite(f'./fingerPrint_images/images_to_generate/{finger}.jpg', result)
    print(f"成功拼接了 {len(stitched_images)} 张图像")


if __name__ == '__main__':
    print(os.getcwd())
    path = "./fingerPrint_images/images_to_generate/"
    fingers = os.listdir(path)
    for finger in fingers:
        main(path + finger, finger)
