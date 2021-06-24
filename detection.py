import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis.inference import init_detector
from mmdet.models import build_detector
from mmdet.apis import inference_detector
import numpy as np
import os
import mindelay.toolbox as toolbox
import mindelay.association as association  
import copy
import time
import scipy.io

cfg_file = './faster_rcnn_r101_fpn_1x.py'
checkpoint_file = './faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
data_dir = '../Dataset/data_tracking_image_2/training/image_02/'
result_dir = './test/'
os.makedirs(result_dir, exist_ok=True)

# cfg = mmcv.Config.fromfile(cfg_file)
# cfg.model.pretrained = None

# model = build_detector(cfg.model, test_cfg=cfg.model.test_cfg)
# load_checkpoint(model, checkpoint_file)
device = 'cuda:0'
model = init_detector(cfg_file, checkpoint=checkpoint_file, device=device)

num_frames = [154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294, 373, 78, 340, 106, 376, 209, 145, 339, 1059, 837]
num_cat = 80

for dataset in range(21):
    output = [[]]*num_frames[dataset]
    output_raw = [[]] * num_frames[dataset]
    result = toolbox.initialize_result(num_cat)
    # result = [np.zeros([0, 5])] * num_cat
    for i in range(num_frames[dataset]):
        print(i)
        img = mmcv.imread(data_dir + '%.4i' % dataset + '/' + '%.6i' % i + '.png')
        # mmcv.imshow(img)
        result_det = inference_detector(model, img)
        # model.show_result(img, result_det, out_file='result.jpg')  
        t = time.time()
        # print(result.__len__())
        # print(len(result_det), len(result))
        # print(result_det)
        # print(result[0].shape, result_det[0].shape)
        result = association.update(result, result_det)
        # print(result)
        # print(result_det)
        # result = likelihoo024d.update(result, result_det)
        result = toolbox.combine_result(result, result_det, 0.5)
        # print(time.time()-t)
        # print(result[0].shape, result_det[0].shape)
        # print(result[0])
        output[i] = copy.deepcopy(result)
        # print(output[i][0])
        output_raw[i] = copy.deepcopy(result_det)
    print(result)

    scipy.io.savemat(result_dir + '%.4i' % dataset + '.mat', {"result": output})
    scipy.io.savemat(result_dir + '%.4i' % dataset + '_raw.mat', {"result": output_raw})
    print(output[0][0])

    m = scipy.io.loadmat(result_dir + '%.4i' % dataset + '.mat')
    print(m['result'].shape, m['result'][0][0])