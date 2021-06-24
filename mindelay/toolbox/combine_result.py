import numpy as np
# from grid.mmdet.ops.nms.nms_wrapper import nms
from mmcv.ops import nms
import torch

def combine_result(result, result_det, thre_nms):
    n = result.__len__()
    output = np.empty((n,), dtype = np.object)
    for i in range(n):
        if result_det[i].shape[0] > 0:
            temp_lr = np.log(result_det[i][:,4]+0.25) - np.log(1 - result_det[i][:,4]+0.25)
            temp_lr[temp_lr < 0] = 0
            result_det[i][:, 4] = temp_lr
        # print(result_det)
        # print(np.vstack((result[i], result_det[i])).shape)
        # print(np.vstack((result[i], result_det[i]))[:, :4].shape)
        # print(np.vstack((result[i], result_det[i]))[:, 4].shape)
        box = torch.from_numpy(np.vstack((result[i], result_det[i]))[:, :4]).contiguous().float()
        score = torch.from_numpy(np.vstack((result[i], result_det[i]))[:, 4]).contiguous().float()
        output[i] = nms(box, score, iou_threshold = thre_nms)[0].numpy()
        # output = nms(np.vstack((result[i], result_det[i])), thre_nms, device_id=None)[0]
        output[i] = output[i][output[i][:,4]>0,:]
    return output
