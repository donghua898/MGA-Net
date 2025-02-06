import math

from skimage.segmentation import slic
import numpy as np
import matplotlib.pyplot as plt
from functiongcnlocal import *
from modelgcnlocal import *


seed = 1234
seed_torch(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCH = 600
drop_out = 0.2
learning_rate = 0.03
dataname = 'Jasper'

data, col, colpad, pad_flag, order_abund, order_endmem,\
alpha, beta, cc, dd, ee, ff, n_segments = get_dataset(dataname)

abundance_GT = torch.from_numpy(data["A"])
endmember_number, pixel_number = abundance_GT.shape
abundance_GT = torch.reshape(abundance_GT, (endmember_number, col, col))

original_HSI0 = torch.from_numpy(data['Y'].astype(float)).to(torch.float32)
band_Number = original_HSI0.shape[0]
original_HSI = torch.reshape(original_HSI0, (band_Number, col, col))
original_HSI = original_HSI.permute(1,2,0)

VCA_endmember = data["M1"]
GT_endmember = data["M"]
endmember_init = torch.from_numpy(VCA_endmember).float()
GT_init = torch.from_numpy(GT_endmember).unsqueeze(2).unsqueeze(3).float()



segments = slic(original_HSI, n_segments=n_segments, compactness=3, sigma=1)
#(100, 100)
labels = segments.ravel()


# 提取超像素质心
centroids = extract_superpixel_centroids(original_HSI, segments)
inputsup = torch.from_numpy(centroids.astype(float)).to(torch.float32)
#(100, 198)


# 构建关联矩阵 Q 和邻接矩阵 A
Q = construct_association_matrix(original_HSI, segments)
# torch.Size([10000, 100])
A = construct_adjacency_matrix(segments)
# torch.Size([100, 100])

# 绘制超像素分割结果
# visualize_superpixel_segmentation(original_HSI, segments)


sortabund = torch.zeros((endmember_number, colpad * colpad))
deweightsup = endmember_init
for i in range(1, np.max(labels)+1):
    net = multiStageUnmixing(col, endmember_number, band_Number).to(device)
    model_dict = net.state_dict()
    model_dict["decoder.0.weight"] = endmember_init
    net.load_state_dict(model_dict)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)
    MSE = torch.nn.MSELoss(size_average=True)


    b = math.ceil(colpad * colpad / float(np.max(labels)) / float(endmember_number)) * endmember_number
    # 160

    print('i=',i)


    idx = np.where(labels == i)[0]

    inputlocal = original_HSI0[:, idx].transpose(1, 0)
    for epoch in range(EPOCH):


        scheduler.step()
        original_HSI = original_HSI.to(device)
        inputsup = inputsup.to(device)
        A = A.to(device)
        inputlocal = inputlocal.to(device)

        net.train()

        input = original_HSI.permute(2,0,1).unsqueeze(0)
        abund, output = net(input, inputsup, A, inputlocal, i)
        abundanceLoss = reconstruction_SADloss(inputlocal, output)
        MSELoss = MSE(inputlocal, output)

        ALoss = alpha * abundanceLoss
        BLoss = beta * MSELoss

        total_loss = ALoss + BLoss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for para in net.decoder.parameters():
            para.data.clamp_(0, 1)

        if epoch == EPOCH - 1:

            abund_GTsup = abundance_GT.reshape(endmember_number, -1)
            abund_GTsup = abund_GTsup[:, idx]
            en_abundance, RMSE_abundance = arange_A(abund.transpose(1, 0), abund_GTsup, endmember_number)
            sortabund[:, idx] = en_abundance.transpose(1, 0)
            deweightsup = net.state_dict()["decoder.0.weight"]





net.eval()
abund = sortabund.reshape(endmember_number, colpad, colpad)


decoder_para = net.state_dict()["decoder.0.weight"].cpu().numpy()


en_abundance, abundance_GT = norm_abundance_GT(abund, abundance_GT)
decoder_para, GT_endmember = norm_endmember(decoder_para, GT_endmember, endmember_number)
en_abundance = np.transpose(en_abundance, (1,2,0))
en_abundance, decoder_para, RMSE_abundance, SAD_endmember= arange_A_E(
    en_abundance, np.transpose(abundance_GT, (1,2,0)), decoder_para, GT_endmember, endmember_number)

rmse_cls, mean_rmse = compute_rmse(np.transpose(abundance_GT, (1,2,0)), en_abundance)
print("Class-wise RMSE value:")
for i in range(endmember_number):
    print("Class", i + 1, ":", rmse_cls[i])
print("Mean RMSE:", mean_rmse)


sad_cls, mean_sad = compute_sad(GT_endmember, decoder_para)
print("Class-wise SAD value:")
for i in range(endmember_number):
    print("Class", i + 1, ":", sad_cls[i])
print("Mean SAD:", mean_sad)

