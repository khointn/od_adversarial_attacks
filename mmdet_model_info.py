# detection models pre-trained on coco dataset
# https://mmdetection.readthedocs.io/en/stable/model_zoo.html

import argparse

model_info = {
    'Faster R-CNN': {
        'config_file': './mmdet_configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    },
    'RetinaNet': {
        'config_file': './mmdet_configs/retinanet/retinanet_r50_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
    },

    'YOLOv3': {
        'config_file': './mmdet_configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'
    },
    'Libra R-CNN': {
        'config_file': './mmdet_configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth'
    },
    'FCOS': {
        'config_file': './mmdet_configs/fcos/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
    },
    'Deformable DETR': {
        'config_file': './mmdet_configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
    },
}

extra_model_info = {

    # Extra
    # Faster RCNN
    'FasterX101': {
        'config_file': './mmdet_configs/faster_rcnn/faster-rcnn_x101-32x4d_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth'
    },
    'FasterRN101': {
        'config_file': './mmdet_configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
    },
    
    #YOLOv3
    'YOLOv3MN': {
        'config_file': './mmdet_configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'
    },

    # RetinaNet
    'RetinaNetRN101':{
        'config_file': './mmdet_configs/retinanet/retinanet_r101_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/retinanet/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth'
    },
    'RetinaNetX101': {
        'config_file': './mmdet_configs/retinanet/retinanet_x101-32x4d_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/retinanet/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth'
    },

    'SSD512': {
        'config_file': './mmdet_configs/ssd/ssd512_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/ssd512_coco_20210803_022849-0a47a1ca.pth',
    },

    #Librarcnn
    'Libra': {
        'config_file': './mmdet_configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth'
    },
    'LibraRN101':{
        'config_file': './mmdet_configs/libra_rcnn/libra-faster-rcnn_r101_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco_20200203-8dba6a5a.pth'
    },
    'LibraX101':{
        'config_file': './mmdet_configs/libra_rcnn/libra-faster-rcnn_x101-64x4d_fpn_1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315-3a7d0488.pth'
    },

    #FCOS
    'FCOSRN101': {
        'config_file': './mmdet_configs/fcos/fcos_r101-caffe_fpn_gn-head-1x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco-0e37b982.pth'
    },
    'FCOSX101': {
        'config_file': './mmdet_configs/fcos/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth',
    },

    # RTMDet
    'RTMDetT': {
        'config_file': "./mmdet_configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
        'checkpoint_file': './mmdetection/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    },
    'RTMDetM': {
        "config_file": "./mmdet_configs/rtmdet/rtmdet_m_8xb32-300e_coco.py",
        "checkpoint_file": "./mmdetection/checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
    },
    'RTMDetL': {
        "config_file": "./mmdet_configs/rtmdet/rtmdet_l_8xb32-300e_coco.py",
        "checkpoint_file": "./mmdetection/checkpoints/rtmdet/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
    },
    'RTMDetX': {
        "config_file": "./mmdet_configs/rtmdet/rtmdet_x_8xb32-300e_coco.py",
        "checkpoint_file": "./mmdetection/checkpoints/rtmdet/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
    },

    # GLIP
    'GLIPTA': {
        'config_file': "./mmdet_configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py",
        'checkpoint_file': "./mmdetection/checkpoints/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_180419-e6addd96.pth"
    },
    'GLIPTB': {
        'config_file': "./mmdet_configs/glip/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco.py",
        'checkpoint_file': "./mmdetection/checkpoints/glip/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230916_163538-650323ba.pth"
    },
    'GLIPTC': {
        'config_file': "/mmdet_configs/glip/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco.py",
        'checkpoint_file': "./mmdetection/checkpoints/glip/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_182935-4ba3fc3b.pth"
    },
    'GLIPL': {
        "config_file": "./mmdet_configs/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco.py",
        "checkpoint_file": "/mmdetection/checkpoints/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800-e9be4274.pth"
    },
    'GLIP': {
        'config_file': './mmdet_configs/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco.py',
        'checkpoint_file': './mmdetection/checkpoints/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_224410-ba97be24.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230914_224410-ba97be24.pth'
    },

    #DINO:
    'DINO': {
        'config_file': '/mmdet_configs/mm_grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py',
        'checkpoint_file': '/mmdetection/checkpoints/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201-f219e0c0.pth',
        'download_link': ''
    },
    'DINOT': {
        'config_file': "./mmdet_configs/mm_grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py",
        'checkpoint_file': "/mmdetection/checkpoints/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth"
    }
}


def main(extra=False):
    import urllib.request
    from pathlib import Path

    checkpoints_root = Path('mmdetection/checkpoints')
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    existing_files = list(checkpoints_root.glob('*.pth'))
    existing_files = [file.name for file in existing_files]

    for idx,model_name in enumerate(model_info):
        url = model_info[model_name]['download_link']
        file_name = url.split('/')[-1]
        if file_name in existing_files:
            print(f"{model_name} already exists, {idx+1}/{len(model_info)}")
            continue
        print(f'downloading {model_name} {idx+1}/{len(model_info)}')
        file_data = urllib.request.urlopen(url).read()
        with open(checkpoints_root / file_name, 'wb') as f:
            f.write(file_data)

    if extra:
            print("downloading extra models")
            for idx,model_name in enumerate(extra_model_info):
                url = extra_model_info[model_name]['download_link']
                file_name = url.split('/')[-1]
                if file_name in existing_files:
                    print(f"{model_name} already exists, {idx+1}/{len(extra_model_info)}")
                    continue
                print(f'downloading {model_name} {idx+1}/{len(extra_model_info)}')
                file_data = urllib.request.urlopen(url).read()
                with open(checkpoints_root / file_name, 'wb') as f:
                    f.write(file_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate perturbations")
    parser.add_argument("--extra", action='store_true')
    extra = parser.parse_args().extra
    main(extra)