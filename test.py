import json
from mmcv import Config
conf_folder = '/home/maksim/mmdetection/configs/'
path_conf = 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(conf_folder+path_conf)
model = cfg.model
print(cfg.total_epochs)
#cfg.model.roi_head.bbox_head.num_classes = 12
#print(cfg.model.roi_head.bbox_head.num_classes)