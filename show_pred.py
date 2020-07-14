from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np
model = init_detector('/mmdetection_aircraft/work_dirs/cascade_mask_101/cascade_mask_101.py', 
'/mmdetection_aircraft/epoch_100_cascade_focal_loss.pth')
img = '/mmdetection_aircraft/data/coco/images/photo_2020-05-20_21-16-53.jpg'
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.8, fig_size=(15, 10))