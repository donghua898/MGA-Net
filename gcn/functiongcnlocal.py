import numpy as np
import torch
import scipy.io as sio
import torch.backends.cudnn as cudnn
import random
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F



def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(input):
    input_normalize = torch.zeros(input.shape)
    for i in range(input.shape[0]):
        input_max = torch.max(input[i, :])
        input_min = torch.min(input[i, :])
        input_normalize[i, :] = (input[i, :] - input_min) / (input_max - input_min)
    return input_normalize


def get_dataset(dataset_name):
    if dataset_name == 'Samson':
        data = sio.loadmat("data0/samson_dataset(1).mat")
        data2 = sio.loadmat("/data1/DH/sup326/samsonsup1_segnum50_49.mat")
        col = 95
        colpad = 95
        pad_flag = False
        if pad_flag:
            colpad = 112
        order_abund, order_endmem = (0, 1, 2), (0, 1, 2)
        alpha = 1
        beta = 10
        cc = 500
        dd = 0.1
        ee = 0.05
        ff = 0.001
        n_segments = 50
    elif dataset_name == 'Apex':
        data = sio.loadmat("data0/apex_dataset.mat")
        col = 110
        colpad = 110
        pad_flag = False
        if pad_flag:
            colpad = 112
        order_abund, order_endmem = (0,1,2,3), (0,1,2,3)
        alpha = 1
        beta = 10
        cc = 0
        dd = 1
        ee = 1
        ff = 0.01
        n_segments = 100
    elif dataset_name == 'Jasper':
        data = sio.loadmat("data0/jasper1.mat")
        data2 = sio.loadmat("/data1/DH/sup326/jaspersup1_segnum50_49.mat")
        col = 100
        colpad = 100
        pad_flag = False
        if pad_flag:
            colpad = 112
        order_abund, order_endmem = (2,0,1,3), (2,0,1,3)
        alpha = 10
        beta = 1
        cc = 400
        dd = 0.1
        ee = 0.5
        ff = 0.001
        n_segments = 50
    elif dataset_name == 'Urban':
        data = sio.loadmat("data0/urban4.mat")
        col = 307
        colpad = 307
        pad_flag = False
        if pad_flag:
            colpad = 310
        order_abund, order_endmem = (0,4,2,1,3), (0,4,2,1,3)
        alpha = 10
        beta = 1
        cc = 1000
        dd = 0.001
        ee = 0.5
        ff = 0.1
        n_segments = 150
    elif dataset_name == 'Sy10':
        # data = sio.loadmat("data0/UST_NET_sy30.mat")
        data = sio.loadmat("data0/sy1/sy_10db.mat")
        data2 = sio.loadmat("/data1/DH/sup326/sy10sup1_segnum20_16.mat")
        col = 60
        colpad = 60
        pad_flag = False
        if pad_flag:
            colpad = 60
        order_abund, order_endmem = (4,2,0,3,1), (4,2,0,3,1)
        alpha = 10
        beta = 50
        cc = 500
        dd = 0.5
        ee = 0.1
        ff = 0.01
        n_segments = 100
    elif dataset_name == 'Sy20':
        # data = sio.loadmat("data0/UST_NET_sy30.mat")
        data = sio.loadmat("data0/sy1/sy_20db.mat")
        data2 = sio.loadmat("/data1/DH/sup326/sy20sup1_segnum20_16.mat")
        col = 60
        colpad = 60
        pad_flag = False
        if pad_flag:
            colpad = 60
        order_abund, order_endmem = (4,2,0,3,1), (4,2,0,3,1)
        alpha = 10
        beta = 50
        cc = 500
        dd = 0.1
        ee = 0.1
        ff = 0.01
        n_segments = 100
    elif dataset_name == 'Sy30':
        # data = sio.loadmat("data0/UST_NET_sy30.mat")
        data = sio.loadmat("data0/sy1/sy_30db.mat")
        data2 = sio.loadmat("/data1/DH/sup326/sy30sup1_segnum20_16.mat")
        col = 60
        colpad = 60
        pad_flag = False
        if pad_flag:
            colpad = 60
        order_abund, order_endmem = (4,2,0,3,1), (4,2,0,3,1)
        alpha = 10
        beta = 50
        cc = 50
        dd = 0.1
        ee = 0.5
        ff = 0.02
        n_segments = 20
    elif dataset_name == 'Sy40':
        # data = sio.loadmat("data0/UST_NET_sy30.mat")
        data = sio.loadmat("data0/sy1/sy_40db.mat")
        data2 = sio.loadmat("/data1/DH/sup326/sy40sup1_segnum20_16.mat")
        col = 60
        colpad = 60
        pad_flag = False
        if pad_flag:
            colpad = 60
        order_abund, order_endmem = (4,2,0,3,1), (4,2,0,3,1)
        alpha = 10
        beta = 50
        cc = 500
        dd = 0.1
        ee = 0.5
        ff = 0.01
        n_segments = 100

    return data, col, colpad, pad_flag, \
        order_abund, order_endmem, alpha, beta, cc, dd, ee, ff, n_segments






def l12_norm(inputs):
    # out = torch.mean(torch.sum(torch.sqrt(torch.abs(inputs)), dim=1))
    out = torch.mean(torch.sum(torch.sqrt(torch.abs(inputs)), dim=1))
    return out

def nuclear_norm(inputs):
    U, S, V  = torch.svd(inputs)
    nuclear_norm = torch.sum(S)
    return nuclear_norm


class MinVolumn(nn.Module):
    def __init__(self, band, num_classes):
        super(MinVolumn, self).__init__()
        self.band = band
        self.num_classes = num_classes

    def __call__(self, edm):
        edm_result = torch.reshape(edm, (self.band,self.num_classes))
        edm_mean = edm_result.mean(dim=1, keepdim=True)
        loss = ((edm_result - edm_mean) ** 2).sum() / self.band / self.num_classes
        return loss


# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
    return rmse


# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim
# SAD loss of reconstruction
def reconstruction_SADloss(output, target):


    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=1))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss

def find_unique_array(arr1, arr2):
    set1 = set(arr1.flatten())
    set2 = set(arr2.flatten())
    if len(set1) == len(arr1.flatten()) and len(set2) == len(arr2.flatten()):
        print('两个数组都没有重复元素')
        return None
    elif len(set1) == len(arr1.flatten()):
        print('RMSE没有重复元素')
        return arr1
    elif len(set2) == len(arr2.flatten()):
        print('SAD没有重复元素')
        return arr2
    else:
        print('两个数组都有重复元素')
        return None
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT, P):
    RMSE_matrix = np.zeros((P, P))
    SAD_matrix = np.zeros((P, P))
    RMSE_index = np.zeros(P).astype(int)
    SAD_index = np.zeros(P).astype(int)
    RMSE_abundance = np.zeros(P)
    SAD_endmember = np.zeros(P)

    for i in range(0, P):
        abundance_GT_input0 = np.stack([abundance_GT_input[:, :, i]]*P, axis=-1)
        endmember_GT0 = np.stack([endmember_GT[:, i]]*P, axis=-1)
        RMSE_matrix[i, :], a = compute_rmse(abundance_GT_input0, abundance_input)
        SAD_matrix[i, :], b = compute_sad(endmember_GT0, endmember_input)
        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])

    print(RMSE_index, SAD_index)
    result = find_unique_array(RMSE_index, SAD_index)
    if result is not None:
        RMSE_index = result
        SAD_index = result
        print("纠正后的数组:")
        print(RMSE_index, SAD_index)
    abundance_input1 = np.zeros_like(abundance_input)
    abundance_input2 = np.zeros_like(abundance_input)

    abundance_input1[:, :, np.arange(P)] = abundance_input[
        :, :, RMSE_index]
    abundance_input2[:, :, np.arange(P)] = abundance_input[
                                                      :, :, SAD_index]
    endmember_input[:, np.arange(P)] = endmember_input[:, SAD_index]
    return abundance_input1, endmember_input, RMSE_abundance, SAD_endmember

def arange_A(abundance_input, abundance_GT_input, P):


    abundance_input = abundance_input / (torch.sum(abundance_input, dim=0))

    abundance_input = abundance_input.transpose(1,0).cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.transpose(1,0).cpu().detach().numpy()

    RMSE_matrix = np.zeros((P, P))
    RMSE_index = np.zeros(P).astype(int)
    RMSE_abundance = np.zeros(P)

    for i in range(0, P):
        abundance_GT_input0 = np.stack([abundance_GT_input[:, i]] * P, axis=-1)

        RMSE_matrix[i, :], a = compute_rmsesup(abundance_GT_input0, abundance_input)

        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])

    # print(RMSE_index)
    abundance_input1 = np.zeros_like(abundance_input)
    abundance_input1[:, np.arange(P)] = abundance_input[:, RMSE_index]
    abundance_input1 = torch.from_numpy(abundance_input1)
    return abundance_input1, RMSE_abundance

def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse, mean_rmse
def compute_rmsesup(x_true, x_pre):
    h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:,i] - x_pre[:,i]) ** 2).sum() / (h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (h * c))
    return class_rmse, mean_rmse

def compute_re(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt(((x_true - x_pred) ** 2).sum() / (img_w * img_h * img_c))


def compute_sad(inp, target):
    p = inp.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        inp_norm = np.linalg.norm(inp[:, i], 2)
        tar_norm = np.linalg.norm(target[:, i], 2)
        summation = np.matmul(inp[:, i].T, target[:, i])
        sad_err[i] = np.arccos(summation / (inp_norm * tar_norm))
    mean_sad = np.mean(sad_err)

    return sad_err, mean_sad


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()


    def forward(self, x):

        h_diff = torch.abs(x[:-1, :] - x[1:, :])
        # 计算总变差
        tv = torch.mean(h_diff)
        return tv



def plot_abundance(ground_truth, estimated, em):
    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        plt.subplot(2, em, i + 1)
        plt.imshow(ground_truth[:, :, i], cmap='jet')
    for i in range(em):
        plt.subplot(2, em, em + i + 1)
        plt.imshow(estimated[:, :, i], cmap='jet')
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_dir + "abundance.png")



def plot_endmembers(target, pred, em):
    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
        plt.plot(pred[:, i], label="Extracted")
        plt.plot(target[:, i], label="GT")
        plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


    # plt.savefig(save_dir + "end_members.png")








# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=0))

    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input


# endmember normalization
def norm_endmember(endmember_input, endmember_GT, endmember_number):

    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT



def visualize_superpixel_segmentation(hyperspectral_image, segments):

    # 绘制超像素分割结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(hyperspectral_image[:,:,(10,90,180)])
    plt.title('Original Image')


    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(hyperspectral_image[:,:,(10,90,180)].numpy(), segments))
    plt.title('Superpixel Segmentation')
    plt.show()


# 超像素质心提取函数
def extract_superpixel_centroids(image, segments):
    print(segments)
    num_segments = np.max(segments)
    centroids = np.zeros((num_segments, image.shape[-1]))
    print(centroids.shape,'374')
    for i in range(num_segments):
        mask = (segments-1 == i)
        centroids[i] = torch.mean(image[mask], dim=0)
    return centroids


# 构建关联矩阵 Q
def construct_association_matrix(hyperspectral_image, segments):
    num_pixels = hyperspectral_image.shape[0] * hyperspectral_image.shape[1]
    num_segments = np.max(segments)
    association_matrix = torch.zeros(num_pixels, num_segments)
    segments = segments.reshape(num_pixels)
    for i in range(num_segments):
        mask = (segments-1 == i)
        association_matrix[mask, i] = 1
    return association_matrix


# 构建邻接矩阵 A
def construct_adjacency_matrix(segments):
    num_segments = np.max(segments)
    adjacency_matrix = torch.zeros(num_segments, num_segments)
    for i in range(num_segments):
        for j in range(num_segments):
            if i == j:
                adjacency_matrix[i, j] = 1
            else:
                if are_adjacent(segments, i, j):
                    adjacency_matrix[i, j] = 1
    return adjacency_matrix


# 定义函数判断相邻超像素
def are_adjacent(segments, segment_i, segment_j):
    boundary_i = np.where(segments == segment_i)
    boundary_j = np.where(segments == segment_j)

    if len(np.intersect1d(boundary_i[0], boundary_j[0])) > 0:
        return True
    else:
        return False


def graph2image(segments, proton_matrices):
    # 获取图像的高度和宽度
    height, width = segments.shape
    num_segments = np.max(segments) + 1
    # 创建一个与原始图像相同大小的零矩阵，用于存储重建后的高光谱图像
    hyperspectral_image = np.zeros((height, width, proton_matrices.shape[1]))
    # 遍历每个超像素分割区域
    for segment_id in range(num_segments):
        # 找到当前超像素分割区域在图像中对应的像素索引
        indices = np.where(segments == segment_id)
        # 获取当前超像素分割区域对应的质子矩阵
        proton_matrix = proton_matrices[segment_id]
        # 将当前超像素分割区域内的所有像素替换为质子矩阵中的值
        for i, j in zip(indices[0], indices[1]):
            hyperspectral_image[i, j, :] = proton_matrix

    return hyperspectral_image
