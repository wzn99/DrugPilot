import torch

# 假设 gt_keep_mask_adj 已经存在
gt_keep_mask_adj = torch.tensor([ 0, 1, 1, 0, 0], device='cuda:0', dtype=torch.float32)

# 假设 gt_adj 是原始的邻接矩阵，形状为 (29, 29)
gt_adj = torch.randn(5, 5, device='cuda:0')

# 创建一个全为1的掩码矩阵
mask_matrix = torch.ones(5, 5, device='cuda:0')

# 将需要隐藏的节点及其所有边关系置为0
for i in range(len(gt_keep_mask_adj)):
    if gt_keep_mask_adj[i] == 1:
        mask_matrix[i, :] = 0
        mask_matrix[:, i] = 0

# 生成新的邻接矩阵
masked_adj = gt_adj * mask_matrix

print(masked_adj)
print(masked_adj.shape)  # 应该是 torch.Size([29, 29])

# 生成新的邻接矩阵
masked_adj = gt_adj * (1-mask_matrix)

print(masked_adj)
print(masked_adj.shape)  # 应该是 torch.Size([29, 29])
