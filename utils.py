import cv2
import jittor as jt
import jsparse.nn.functional as F
import maxflow
import numpy as np
jt.flags.use_cuda = 1


def imshow(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_seg_map(seg_map, path):
    seg_map_three_channel = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_map_three_channel[seg_map == 1] = [255, 0, 0]
    seg_map_three_channel[seg_map == 2] = [0, 0, 255]
    cv2.imwrite(path, seg_map_three_channel)


def conv_l2_distance(image, kernel, mask):
    '''
        Inputs:
            image: numpy.narray, (h, w, 3)
            kernel: numpy.narray, (h, w, 3)
            mask: numpy.narray, (h, w)
        Outputs:
            result: numpy.narray, (H-h+1, W-w+1)
    '''
    # 使用了jittor的元算子，的确比pytorch快很多
    H, W, _ = image.shape
    h, w, _ = kernel.shape
    result = jt.zeros((H-h+1, W-w+1))
    image_image = image.reindex([H-h+1, W-w+1, h, w, 3], ['i0+i2', 'i1+i3', 'i4'])
    kernel_kernel = jt.broadcast_var(kernel, image_image, dims=[0, 1])
    mask_mask = jt.broadcast_var(mask, image_image, dims=[0, 1, 4])
    result = jt.sum((image_image - kernel_kernel) ** 2 * mask_mask, dims=[2, 3, 4])
    return result.fetch_sync()


def graph_cut(image_1, image_2, mask):
    '''
    Inputs:
        image_1: numpy.narray, (h, w, 3)
        image_2: numpy.narray, (h, w, 3)
        mask: numpy.narray, (h, w), 0 for background, 1 for image_1 boundary, 2 for image_2 boundary, 3 for inside
    Outputs:
        segmap: numpy.narray, (h, w), 0 if not in masked area. 1 if be with img1 and 2 if img2.
    '''
    assert len(image_1.shape) == 3 and len(image_2.shape) == 3 and len(mask.shape) == 2
    assert image_1.shape == image_2.shape and image_1.shape[:2] == mask.shape

    h, w, _ = image_1.shape

    estimate_node_num = (mask > 0).sum()
    estimate_edge_num = estimate_node_num * 4

    g = maxflow.Graph[float](estimate_node_num, estimate_edge_num)
    nodes = g.add_nodes(estimate_node_num)
    node_map = {}
    node_map_inv = {}

    for i0 in range(h):
        for i1 in range(w):
            if mask[i0, i1] > 0:
                node_map[len(node_map)] = (i0, i1)
                node_map_inv[(i0, i1)] = len(node_map_inv)

    inf = 1e9
    for i0 in range(h):
        for i1 in range(w):
            if mask[i0, i1] > 0:
                if i0 + 1 < h and mask[i0 + 1, i1] > 0:
                    weight = np.linalg.norm(image_1[i0, i1] - image_2[i0, i1]) + np.linalg.norm(image_1[i0 + 1, i1] - image_2[i0 + 1, i1])
                    g.add_edge(nodes[node_map_inv[(i0, i1)]], nodes[node_map_inv[(i0 + 1, i1)]], weight, weight)
                if i1 + 1 < w and mask[i0, i1 + 1] > 0:
                    weight = np.linalg.norm(image_1[i0, i1] - image_2[i0, i1]) + np.linalg.norm(image_1[i0, i1 + 1] - image_2[i0, i1 + 1])
                    g.add_edge(nodes[node_map_inv[(i0, i1)]], nodes[node_map_inv[(i0, i1 + 1)]], weight, weight)
                if mask[i0, i1] == 1:
                    g.add_tedge(nodes[node_map_inv[(i0, i1)]], inf, 0)
                if mask[i0, i1] == 2:
                    g.add_tedge(nodes[node_map_inv[(i0, i1)]], 0, inf)
    
    flow = g.maxflow()
    segmentation_result = 1 + np.array(list(map(g.get_segment, nodes)))
    segmap = np.zeros_like(mask)
    for i, result in enumerate(segmentation_result):
        segmap[node_map[i][0], node_map[i][1]] = result
    return segmap, flow


def poisson_matrix(mask):
    '''
    Inputs:
        mask: numpy.narray, (h, w), 1 for image_1 area, 2 for image_2 area,
    Outputs:
        pixel_map: dict, {pixel_id: (i0, i1)}
        rows: jt.Var, (n, )
        cols: jt.Var, (n, )
        vals: jt.Var, (n, )
    '''
    # 这个函数的实现比较naive，但我懒得优化了，运行时间1s左右
    pixel_num = 0
    pixel_map = {}
    pixel_map_inv = {}
    rows = []
    cols = []
    vals = []
    for i0 in range(mask.shape[0]):
        for i1 in range(mask.shape[1]):
            if mask[i0, i1] == 2:
                pixel_map[pixel_num] = (i0, i1)
                pixel_map_inv[(i0, i1)] = pixel_num
                pixel_num += 1
    for i0 in range(mask.shape[0]):
        for i1 in range(mask.shape[1]):
            if mask[i0, i1] == 2:
                rows.append(pixel_map_inv[(i0, i1)])
                cols.append(pixel_map_inv[(i0, i1)])
                val = 0
                if i0 - 1 >= 0:
                    val -= 1
                if i0 + 1 < mask.shape[0]:
                    val -= 1
                if i1 - 1 >= 0:
                    val -= 1
                if i1 + 1 < mask.shape[1]:
                    val -= 1
                vals.append(val)
                if i0 - 1 >= 0 and mask[i0 - 1, i1] == 2:
                    rows.append(pixel_map_inv[(i0, i1)])
                    cols.append(pixel_map_inv[(i0 - 1, i1)])
                    vals.append(1)
                if i0 + 1 < mask.shape[0] and mask[i0 + 1, i1] == 2:
                    rows.append(pixel_map_inv[(i0, i1)])
                    cols.append(pixel_map_inv[(i0 + 1, i1)])
                    vals.append(1)
                if i1 - 1 >= 0 and mask[i0, i1 - 1] == 2:
                    rows.append(pixel_map_inv[(i0, i1)])
                    cols.append(pixel_map_inv[(i0, i1 - 1)])
                    vals.append(1)
                if i1 + 1 < mask.shape[1] and mask[i0, i1 + 1] == 2:
                    rows.append(pixel_map_inv[(i0, i1)])
                    cols.append(pixel_map_inv[(i0, i1 + 1)])
                    vals.append(1)
    rows = jt.array(np.array(rows, dtype=np.int32))
    cols = jt.array(np.array(cols, dtype=np.int32))
    vals = jt.array(np.array(vals, dtype=np.float32))
    size = (pixel_num, pixel_num)
    return pixel_map, rows, cols, vals, size


def b_matrix(image_1, image_2, mask, pixel_map):
    # 这个函数的实现比较naive，但我懒得优化了，运行时间1s左右
    b = np.zeros((len(pixel_map), 3), dtype=np.float32)
    direction = [[-1,0],[1,0],[0,-1],[0,1]]
    for i in range(len(pixel_map)):
        i0, i1 = pixel_map[i]
        val = 0
        for d in direction:
            if 0 <= i0 + d[0] < mask.shape[0] and 0 <= i1 + d[1] < mask.shape[1]:
                val -= 1
                b[i] += image_2[i0 + d[0], i1 + d[1]]
                if mask[i0 + d[0], i1 + d[1]] == 1:
                    b[i] -= image_1[i0 + d[0], i1 + d[1]]
        b[i] += val * image_2[i0, i1]
    return jt.array(b)


def spmv(rows, cols, vals, size, x):
    return jt.squeeze(F.spmm(rows=rows, cols=cols, vals=vals, size=size, mat=jt.unsqueeze(x, 1), cuda_spmm_alg=1), dim=1)


def solve_poisson_equation(rows, cols, vals, b, size):
    '''
    Inputs:
        rows: jt.Var, (n, )
        cols: jt.Var, (n, )
        vals: jt.Var, (n, )
        b: jt.Var, (pixel_num, 3)
        size: tuple, (pixel_num, pixel_num)
    Outputs:
        x: jt.Var, (pixel_num, 3)
    '''
    # 共轭梯度法求解Ax=b
    with jt.no_grad():
        b = b.transpose(0, 1)
        # 尝试过取image_2的值作为x的初始值，但是迭代次数没有明显变化
        x = jt.ones_like(b) * 127.5
        for i in range(3):
            r = b[i] - spmv(rows, cols, vals, size, x[i])
            p = r.clone()
            while True:
                A_mul_p = spmv(rows, cols, vals, size, p)
                r_T_r = jt.matmul(r.transpose(), r)
                alpha = r_T_r / jt.matmul(p.transpose(), A_mul_p)
                # 这里必须进行clip操作，否则结果会出现少量异常颜色点
                x[i] = jt.safe_clip(x[i] + alpha * p, 0.0, 255.0)   
                r_new = r - alpha * A_mul_p
                r_T_r_new = jt.matmul(r_new.transpose(), r_new)
                if r_T_r_new < 1e-3:
                    break
                beta = r_T_r_new / r_T_r
                p = r_new + beta * p
                r = r_new
    return x.transpose(0, 1).fetch_sync()


def poisson_blending(image_1, image_2, mask):
    '''
    Inputs:
        image_1: numpy.narray, (h, w, 3)
        image_2: numpy.narray, (h, w, 3)
        mask: numpy.narray, (h, w), 1 for image_1 area, 2 for image_2 area,
    Outputs:
        new_image: numpy.narray, (h, w, 3)
    '''
    assert len(image_1.shape) == 3 and len(image_2.shape) == 3 and len(mask.shape) == 2
    assert image_1.shape == image_2.shape and image_1.shape[:2] == mask.shape

    image_1 = image_1.astype(np.float32)
    image_2 = image_2.astype(np.float32)
    pixel_map, rows, cols, vals, size = poisson_matrix(mask)
    # 构建b
    b = b_matrix(image_1, image_2, mask, pixel_map)
    # 利用共轭对称法求解Ax=b
    x = solve_poisson_equation(rows, cols, vals, b, size)
    # 利用pixel_map将x填回图像
    new_image = image_1.copy()
    for i in range(len(pixel_map)):
        new_image[pixel_map[i][0], pixel_map[i][1]] = x[i]
    return np.clip(new_image.astype(np.uint8), 0, 255)
