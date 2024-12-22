import cv2
import numpy as np
import os

min_matches = 10


def locate_patch_in_panorama(patch_img, panorama_img):
    global min_matches
    """
    Locate the position of the patch image in the panorama image.
    
    Args:
        patch_img: The image to be located (target image).
        panorama_img: The panoramic image.
        min_matches: Minimum number of good matches required.
    
    Returns:
        corners: Coordinates of the four corners if a match is found, else None.
        num_matches: Number of good matches found.
    """
    sift = cv2.SIFT_create(contrastThreshold=0.02)  # 默认值为 0.04
    kp1, des1 = sift.detectAndCompute(patch_img, None)
    kp2, des2 = sift.detectAndCompute(panorama_img, None)

    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        print("Not enough keypoints detected.")
        return None, 0

    # FLANN Matcher setup
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=150)  # 可改
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception as e:
        print(f"Error during matching: {e}")
        return None, 0

    # Lowe's ratio test to filter good matches
    good_matches = [m for m, n in matches if m.distance < 0.85 * n.distance]  #可改

    if len(good_matches) < min_matches:
        return None, len(good_matches)

    # Find transformation matrix
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    transform = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=15.0)[0]  #可改
    if transform is None:
        return None, 0
        # 从变换矩阵中提取旋转角度和缩放比例
    scale_x = np.sqrt(transform[0, 0] ** 2 + transform[0, 1] ** 2)
    scale_y = np.sqrt(transform[1, 0] ** 2 + transform[1, 1] ** 2)
    angle = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi

    # 检查旋转角度和缩放约束
    if (abs(angle) > 180 or  # 放宽旋转角度限制到±60度以内
            abs(scale_x - 1.0) > 0.3 or  # 缩放限制在0.7-1.3内
            abs(scale_y - 1.0) > 0.3):
        # print(f"变换超出约束范围: 角度={angle:.1f}°, 缩放={scale_x:.2f}/{scale_y:.2f}")
        return None, 0

    # Compute the position of the corners
    h, w = patch_img.shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    transformed_corners = cv2.transform(corners.reshape(1, -1, 2), transform).reshape(-1, 2)

    return transformed_corners.astype(np.int32), len(good_matches)


def draw_best_match(target_img, panorama_imgs, panorama_paths, save_path=None):
    """
    Find the best-matched panorama and mark the location of the target image.
    
    Args:
        target_img: The target image to be matched.
        panorama_imgs: List of panorama images.
        panorama_paths: List of file paths for panorama images.
        save_path: Path to save the final result (optional).
    """
    best_match = None
    best_corners = None
    best_num_matches = 0
    best_image_index = -1

    # Match target image with each panorama image
    for i, panorama_img in enumerate(panorama_imgs):
        corners, num_matches = locate_patch_in_panorama(target_img, panorama_img)
        if num_matches > best_num_matches:
            best_match = panorama_img
            best_corners = corners
            best_num_matches = num_matches
            best_image_index = i

    if best_corners is None:
        print("No sufficient matches found in any panorama.")
        return None

    print(f"Best match found in: {panorama_paths[best_image_index]} with {best_num_matches} good matches.")

    # Draw the matched location on the best panorama
    result_img = cv2.cvtColor(best_match, cv2.COLOR_GRAY2BGR)
    cv2.polylines(result_img, [best_corners], True, (0, 255, 0), 2)
    for corner in best_corners:
        cv2.circle(result_img, tuple(corner), 5, (0, 0, 255), -1)

    # Display and save result
    cv2.imshow("Best Matched Location", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, result_img)
        print(f"Result saved at: {save_path}")
    return best_image_index


def find_best_match(target_img, panorama_imgs):
    """
    Find the best-matched panorama and mark the location of the target image.
    
    Args:
        target_img: The target image to be matched.
        panorama_imgs: List of panorama images.
    """
    best_corners = None
    best_num_matches = 0
    best_image_index = -1

    # Match target image with each panorama image
    for i, panorama_img in enumerate(panorama_imgs):
        corners, num_matches = locate_patch_in_panorama(target_img, panorama_img)
        if num_matches > best_num_matches:
            best_corners = corners
            best_num_matches = num_matches
            best_image_index = i

    if best_corners is None:
        # print("No sufficient matches found in any panorama.")
        return None

    # print(f"Best match found in: {panorama_paths[best_image_index]} with {best_num_matches} good matches.")
    return best_image_index


def judge(img, file_name):
    """
    此函数将目标图像与已有的库图像进行匹配
    若有匹配对象，返回最佳匹配对象的序列索引
    若无匹配对象，返回none
    """
    # Load target and panoramic images
    #img_path = './fingerPrint_images/registered_fingers/srz/'
    img_path = './fingerPrint_images/GUI_registered_fingers/' + file_name + '/'
    
    #panorama_paths = ['right_1.jpg', 'right_2.jpg', 'right_3.jpg', 'right_4.jpg', 'right_5.jpg', 'left_2.jpg', 'left_3.jpg','left_5.jpg','right_index_2.jpg','right_ring_2.jpg']
    panorama_paths = ['Right_Index_1.jpg', 'Right_Middle_1.jpg', 'Right_Ring_1.jpg', 'Right_Pinky_1.jpg', 'Right_Index_2.jpg',  'Right_Middle_2.jpg',  'Right_Ring_2.jpg', 'Right_Thumb_1.jpg']
    
    target_img = img
    panorama_imgs = [cv2.imread(img_path + path, 0) for path in panorama_paths]

    if target_img is None or any(img is None for img in panorama_imgs):
        print("Error reading images. Please check the file paths.")
        return

    return find_best_match(target_img, panorama_imgs)


if __name__ == '__main__':
    min_matches = 10
    judge()
