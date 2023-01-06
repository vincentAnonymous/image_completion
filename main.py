import os
import argparse
import numpy as np
import cv2
from utils import imshow, conv_l2_distance, graph_cut, poisson_blending, save_seg_map
import jittor as jt
from tqdm import tqdm
jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default='input1')
args = parser.parse_args()


# 文件位置
image_path = os.path.join("data", args.image_name + ".jpg")
mask_path = os.path.join("data", args.image_name + "_mask.jpg")
candidates_dir = os.path.join("data", args.image_name)
candidates_path = os.listdir(candidates_dir)
result_dir = os.path.join("result", args.image_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


# 参数
K = 20


def main():
    print("Image name: ", args.image_name)
    # 第一步：读取图像和mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask < 127] = 0
    mask[mask >= 127] = 255
    # 第二步：精细匹配
    print("Step 1: Fine matching")
    # 利用opencv的dilate提取待补全图像距离缺失区域在𝐾个像素内的区域B
    reverse_mask = 255 - mask
    region_b = cv2.dilate(reverse_mask, np.ones((3, 3), np.uint8), iterations=K) - reverse_mask
    cv2.imwrite(os.path.join(result_dir, "region_b.jpg"), region_b)

    # 获取区域B的最小包裹矩形，并将其保存
    x, y, w, h = cv2.boundingRect(cv2.dilate(reverse_mask, np.ones((3, 3), np.uint8), iterations=K))
    cv2.imwrite(os.path.join(result_dir, "region_b_rect.jpg"), region_b[y:y + h, x:x + w])
    # 获取用于graph cut的mask，将原图边缘的像素点标记为1， candidate的像素点标记为2，内部像素点标记为3，其余像素点标记为0
    image_2_boundary = cv2.dilate(reverse_mask, np.ones((3, 3), np.uint8), iterations=1) - reverse_mask
    image_2_boundary = image_2_boundary[y:y+h, x:x+w]
    image_1_boundary = cv2.dilate(reverse_mask, np.ones((3, 3), np.uint8), iterations=K) - cv2.dilate(reverse_mask, np.ones((3, 3), np.uint8), iterations=K - 1)
    image_1_boundary = image_1_boundary[y:y+h, x:x+w]
    graph_cut_mask = region_b[y:y+h, x:x+w].copy()
    graph_cut_mask[graph_cut_mask > 0] = 3
    graph_cut_mask[image_1_boundary > 0] = 1
    graph_cut_mask[image_2_boundary > 0] = 2

    # 读取候选图片, 对于每张候选图像，将B部分在候选图像上平移，计算每一个位置的𝐿^2误差‖A_𝑠𝑢𝑏−𝐵‖
    match_mask = jt.array(region_b[y:y+h, x:x+w], dtype=jt.float32) / 255.0
    match_kernel = jt.array(image[y:y+h, x:x+w], dtype=jt.float32) / 255.0
    candidates = []
    match_indexes = []
    seg_maps = []
    # 读取候选图片
    new_candidates_path = []
    candidates_dir = os.path.join("data", args.image_name)
    candidates_path = os.listdir(candidates_dir)
    for candidate_path in candidates_path:
        candidate = cv2.imread(os.path.join(candidates_dir, candidate_path))
        # 如果候选图片过小，将其从candidates_path中删除
        if candidate.shape[0] >= h and candidate.shape[1] >= w:
            candidates.append(candidate)
            new_candidates_path.append(candidate_path)
    candidates_path = new_candidates_path
    
    distance_losses = np.zeros(len(candidates_path), dtype=np.float32)
    flow_losses = np.zeros(len(candidates_path), dtype=np.float32)
    # 对于每张候选图片，计算卷积版的L2距离
    print("Step 2: Calculate L2 distance")
    for idx, candidate in enumerate(candidates):
        candidate_jt = jt.array(candidate, dtype=jt.float32) / 255.0
        l2_distance_result = conv_l2_distance(candidate_jt, match_kernel, match_mask)
        # 选取distance最小的位置
        distance_losses[idx] = np.min(l2_distance_result)
        match_indexes.append(np.unravel_index(np.argmin(l2_distance_result), l2_distance_result.shape))
    # 对于每张候选图片，利用graph-cut算法计算融合边界
    print("Step 3: Calculate graph cut")
    for idx, candidate in enumerate(candidates):
        match_index = match_indexes[idx]
        seg_map, flow = graph_cut(image[y:y+h, x:x+w], candidate[match_index[0]:match_index[0]+h, match_index[1]:match_index[1]+w], graph_cut_mask)
        seg_map[mask[y:y+h, x:x+w] == 0] = 2
        seg_map[seg_map == 0] = 1
        save_seg_map(seg_map, os.path.join(result_dir, "seg_map_{}.jpg".format(os.path.basename(candidates_path[idx]))))
        seg_maps.append(seg_map)
        flow_losses[idx] = np.sum(flow)
    # 对于每张候选图片，计算最终的loss
    # 先进行归一化
    # distance_losses = (distance_losses - np.min(distance_losses)) / (np.max(distance_losses) - np.min(distance_losses))
    # flow_losses = (flow_losses - np.min(flow_losses)) / (np.max(flow_losses) - np.min(flow_losses))
    # 计算最终的loss，并排序
    # losses = distance_losses + flow_losses
    losses = flow_losses
    ranks = np.argsort(losses)
    # 对于每张候选图片，利用泊松融合进行图片融合
    print("Step 4: Poisson blending")
    for idx, candidate in tqdm(enumerate(candidates), total=len(candidates)):
        match_index = match_indexes[idx]
        res = poisson_blending(image[y:y+h, x:x+w], candidate[match_index[0]:match_index[0]+h, match_index[1]:match_index[1]+w], seg_maps[idx])
        full_result = image.copy()
        full_result[y:y+h, x:x+w] = res
        cv2.imwrite(os.path.join(result_dir, "mixed_{}.png".format(os.path.basename(candidates_path[idx]))), full_result)

    with open(os.path.join(result_dir, "result.txt"), "w") as f:
        for rank in ranks:
            # f.write("{}:\n\ttotal_loss: {} distance_loss: {} flow_loss: {}\n".format(candidates_path[rank], losses[rank], distance_losses[rank], flow_losses[rank]))
            f.write("{}: loss: {}\n".format(candidates_path[rank], losses[rank]))


if __name__ == '__main__':
    main()