python3 ../tools/inference_mmdet_models.py --config_path ../mmdetection/configs/yolox/mymodel.py --checkpoint ../weights/yolox/final.pth --scale 640 640 --output ../output/1_yolox.txt;
python3 ../tools/inference_mmdet_models.py --config_path ../mmdetection/configs/yolox/mymodel.py --checkpoint ../weights/yolox/final.pth --tta --output ../output/1_yolox_tta.txt