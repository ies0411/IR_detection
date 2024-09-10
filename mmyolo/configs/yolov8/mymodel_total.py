_base_ = "./yolov8_l_syncbn_fast_8xb16-500e_coco.py"
# load_from = "work_dir/y8/scale_aug/epoch_10.pth"
# load_from = "work_dir/y8/scale_aug/epoch_270.pth"
load_from = "work_dir/y8/base/epoch_260.pth"

resume = True

deepen_factor = 1.00
widen_factor = 1.25
# base_lr = 0.01
base_lr = 0.005
# max_epochs = 200  # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
# close_mosaic_epochs = 10

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)),
)


# CUDA_VISIBLE_DEVICES=2,3 tools/dist_train.sh configs/yolov8/mymodel.py 4 --work-dir work_dir/y8/refine/
