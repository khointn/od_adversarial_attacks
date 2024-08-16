# this module is related to mmdetection models
from collections import defaultdict
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
import mmengine

from mmdet.models.detectors import *
from mmdet.apis import DetInferencer as inference_detector

from mmdet_model_info import model_info, extra_model_info
model_info.update(extra_model_info)
print(model_info['GLIP'])

def get_bb_loss(detections, target_clean, LOSS):
    """define the blackbox attack loss
        if the original object is detected, the loss is the conf score of the victim object
        otherwise, the original object disappears, the conf is below the threshold, the loss is the wb ensemble loss
    args:
        detections ():
        target_clean ():
        LOSS ():
    return:
        bb_loss (): the blackbox loss
    """
    max_iou = 0
    for items in detections:
        iou = get_iou(items, target_clean[0])
        if iou > max(max_iou, 0.3) and items[4] == target_clean[0][4]:
            max_iou = iou
            bb_loss = 1e3 + items[5] # add a large const to make sure it is larger than conf ens loss

    # if it disappears
    if max_iou < 0.3:
        bb_loss = LOSS['ens'][-1]

    return bb_loss

def get_loss_and_success_list(im_np, adv_np, LOSS, target_clean, all_models):    
    """get the loss bb, success_list on all surrogate models, and save detections to fig
    
    args:

    returns:
        loss_bb (float): loss on the victim model
        success_list (list of 0/1s): successful for all models
    """

    n_all = len(all_models)
    # 1st row, clean image, detection on surrogate models, detection on victim model
    # 2nd row, perturbed image, detection on surrogate models, detection on victim model

    for model_idx, model in enumerate(all_models):
        det_adv = model.det(im_np)
        #bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]

    success_list = [] # 1 for success, 0 for fail for all models
    for model_idx, model in enumerate(all_models):
        det_adv = model.det(adv_np)
        #bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]

        # check for success and get bb loss
        if model_idx == n_all-1:
            loss_bb = get_bb_loss(det_adv, target_clean, LOSS)

        # victim model is at the last index
        success_list.append(is_success(det_adv, target_clean))

    return loss_bb, success_list

def PM_tensor_weight_balancing(im_path, im, adv, target, w, ensemble, eps, n_iters, alpha, dataset='voc', weight_balancing=False):
    """perturbation machine, balance the weights of different surrogate models
    args:
        im (tensor): original image, shape [1,3,h,w].cuda()
        adv (tensor): adversarial image
        target (numpy.ndarray): label for object detection, (xyxy, cls, conf)
        w (numpy.ndarray): ensemble weights
        ensemble (): surrogate ensemble
        eps (int): linf norm bound (0-255)
        n_iters (int): number of iterations
        alpha (flaot): step size

    returns:
        adv_list (list of Tensors): list of adversarial images for all iterations
        LOSS (dict of lists): 'ens' is the ensemble loss, and other individual surrogate losses
    """
    # prepare target label input: voc -> coco, since models are trained on coco
    bboxes_tgt = target[:,:4].astype(np.float32)
    labels_tgt = target[:,4].astype(int).copy()
    if dataset == 'voc':
        for i in range(len(labels_tgt)): 
            labels_tgt[i] = voc2coco[labels_tgt[i]]

    im_np = im.squeeze().cpu().numpy().transpose(1, 2, 0)
    adv_list = []
    pert = adv - im
    LOSS = defaultdict(list) # loss lists for different models
    for i in range(n_iters):
        pert.requires_grad = True
        loss_list = []
        loss_list_np = []
        for model in ensemble:
            loss = model.loss(im_path, im_np, pert, bboxes_tgt, labels_tgt)
            loss_list.append(loss)
            loss_list_np.append(loss.item())
            LOSS[model.model_name].append(loss.item())
        
        # if balance the weights at every iteration
        if weight_balancing:
            w_inv = 1/np.array(loss_list_np)
            w = w_inv / w_inv.sum()

        # print(f"w: {w}")
        loss_ens = sum(w[i]*loss_list[i] for i in range(len(ensemble)))
        loss_ens.backward()
        with torch.no_grad():
            pert = pert - alpha*torch.sign(pert.grad)
            pert = pert.clamp(min=-eps, max=eps)
            LOSS['ens'].append(loss_ens.item())
            adv = (im + pert).clip(0, 255)
            adv_list.append(adv)
    return adv_list, LOSS


def PM_tensor_weight_balancing_np(im_path, im_np, target, w_np, ensemble, eps, n_iters, alpha, dataset='voc', weight_balancing=False, adv_init=None):
    """perturbation machine, numpy input version
    
    """
    #device = next(ensemble[0].parameters()).device
    device = "cuda:0"
    im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to(device)
    if adv_init is None:
        adv = torch.clone(im) # adversarial image
    else:
        adv = torch.from_numpy(adv_init).permute(2,0,1).unsqueeze(0).float().to(device)

    # w = torch.from_numpy(w_np).float().to(device)
    adv_list, LOSS = PM_tensor_weight_balancing(im_path, im, adv, target, w_np, ensemble, eps, n_iters, alpha, dataset, weight_balancing)
    adv_np = adv_list[-1].squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return adv_np, LOSS


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

COCO_BBOX_LABEL_NAMES = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush')

# voc index to coco index
voc2coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]

def is_to_rgb(model):
    """check if a model takes rgb images or not
    Args: 
        model (~ mmdet.models.detectors): a mmdet model
    """
    to_rgb = True
    for item in model.cfg.model.data_preprocessor:
        if item == 'bgr_to_rgb':
            to_rgb = model.cfg.model.data_preprocessor['bgr_to_rgb']
    return to_rgb


def get_conf_thres(model_name):
    """assign a different confidence threshold for every model
    Args: 
        model_name (str): the name of model
    Returns:
        conf_thres (~ float): the confidence threshold
        conf_thres is selected to reduce false positive rate
    """
    if model_name in ['Grid R-CNN']:
        conf_thres = 0.7
    elif model_name in ['Faster R-CNN', 'FreeAnchor', 'SSD', 'FasterRN101']:
        conf_thres = 0.6    
    elif model_name in ['YOLOv3', 'RetinaNet', 'Libra R-CNN', 'GN+WS', 'YOLOv3MN', 'RetinaNetRN101', 'RetinaNetX101']:
        conf_thres = 0.5
    elif model_name in ['FoveaBox', 'RepPoints', 'DETR']:
        conf_thres = 0.4
    elif model_name in ['FCOS', 'CenterNet', 'FCOSRN101', 'FCOSX101']:
        conf_thres = 0.3
    elif model_name in ['Deformable DETR', 'DINO']: # was 0.3
        conf_thres = 0.1 # in context paper it was 0.1
    elif model_name in ['ATSS']:
        conf_thres = 0.2
    else:
        conf_thres = 0.2
        # tested for YOLOX, 
    return conf_thres


def output2det(outputs, im, conf_thres=0.5, dataset='voc'):
    """Convert the model outputs to targeted format
    Args: 
        outputs (lists): 80 lists, each has a numpy array of Nx5, (bbox and conf)
        conf_thres (float): confidence threshold
        im (np.ndarray): input image for get the size and clip at the boundary
    Returns:
        det (numpy.ndarray): _bboxes(xyxy) - 4, _cls - 1, _prob - 1
        dataset (str): if use 'voc', only the labels within the voc dataset will be returned
    """
    det = []
    #print(outputs)
    for idx, items in enumerate(outputs["predictions"]):
        for item in range(len(items['labels'])):
            #det.append(item[:4].tolist() + [idx] + item[4:].tolist())
            #print(items['bboxes'][item],[idx],items['scores'][item])
            det.append(items['bboxes'][item] + [items['labels'][item]] + [items['scores'][item]])
            #print("TESTING:", items['bboxes'][item] + [items['labels'][item]] + [items['scores'][item]])
    det = np.array(det)
    
    # if det is empty
    if len(det) == 0: 
        return np.zeros([0,6])

    # thresholding the confidence score
    det = det[det[:,-1] >= conf_thres]
    
    if dataset == 'voc':
        # map the labels from coco to voc
        voc2coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]
        for idx, item in enumerate(det):
            if item[4] not in voc2coco:
                item[4] = -1
            else:
                det[idx,4] = voc2coco.index(item[4])
        det = det[det[:,4] != -1]

    # make the value in range
    m, n, _ = im.shape
    for item in det:
        item[0] = min(max(item[0],0),n)
        item[2] = min(max(item[2],0),n)
        item[1] = min(max(item[1],0),m)
        item[3] = min(max(item[3],0),m)
    return det

def get_test_data(model, im_path, im, text="$: coco"):
    """get data format for training
    Args:
        model (~ mmdet.models.detectors): a mmdet model
        im (np.ndarray): input numpy image (in bgr format)
        bboxes (np.ndarray): desired bboxes
        labels (np.ndarray): desired labels
    Returns:
        data_train (): train data format
    """
    from mmcv.transforms import Compose
    import math

    if not is_to_rgb(model): im = im[:,:,::-1]
    cfg = model.cfg
    cfg = cfg.copy()

    test_pipeline = Compose(cfg.test_pipeline)
    data = dict(img=im, img_path = im_path)
    data = test_pipeline(data)

    # Calculate pad shape (%32):
    w_pad = math.ceil(data['data_samples'].img_shape[0]/32) * 32
    h_pad = math.ceil(data['data_samples'].img_shape[1]/32) * 32

    # Set metainfo
    pad_shape = (w_pad, h_pad, 3)
    mean = cfg.model.data_preprocessor.mean
    std = cfg.model.data_preprocessor.std
    metainfo = data['data_samples'].metainfo
    metainfo['mean'] = mean
    metainfo['std'] = std
    metainfo['pad_shape'] = pad_shape
    metainfo['batch_input_shape'] = data['data_samples'].img_shape
    if text=="$: coco":
        from mmdet.evaluation import get_classes
        metainfo['text'] = tuple(get_classes('coco'))
    data['data_samples'].set_metainfo(metainfo)
    return data

    
def get_train_data(model, im, pert, device, data, bboxes, labels):
    """get data format for training
    Args:
        model (~ mmdet.models.detectors): a mmdet model
        im (np.ndarray): input numpy image (in bgr format) / with grad
        bboxes (np.ndarray): desired bboxes
        labels (np.ndarray): desired labels
    Returns:
        data_train (): train data format
    """

    from mmengine.structures import InstanceData

    # get model device

    # BELOW IS TRAIN
    data_train = data.copy()
    if not is_to_rgb(model): im = im[:,:,::-1]
    img = torch.from_numpy(im.copy().transpose((2, 0, 1)))[None].float().to(device).contiguous()
    img = (img + pert).clamp(0,255)

    # 'type': 'Resize', 'keep_ratio': True, (1333, 800)

    ''' from file: datasets/pipelines/transforms.py '''
    image_sizes = data_train['data_samples'].img_shape
    w_scale = data_train['data_samples'].scale_factor[0]
    h_scale = data_train['data_samples'].scale_factor[1]
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

    gt_bboxes = bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, image_sizes[1])
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, image_sizes[0])

    gt_instances_data = dict(
                             bboxes=torch.from_numpy(gt_bboxes).to(device),
                             labels=torch.from_numpy(labels).to(device))

    gt_instances = InstanceData(**gt_instances_data)
    data_train['data_samples'].gt_instances = gt_instances
    img = F.interpolate(img, size=image_sizes, mode='bilinear', align_corners=True)

    # 'type': 'Normalize', 'mean': [103.53, 116.28, 123.675], 'std': [1.0, 1.0, 1.0], 'to_rgb': False
    mean = data_train['data_samples'].mean
    std = data_train['data_samples'].std
    transform = transforms.Normalize(mean=mean, std=std)
    img = transform(img)

    # 'type': 'Pad', 'size_divisor': 32
    pad_sizes = data_train['data_samples'].pad_shape[:2]
    left = top = 0
    bottom = pad_sizes[0] - image_sizes[0]
    right = pad_sizes[1] - image_sizes[1]
    img = F.pad(img, (left, right, top, bottom), "constant", 0)

    data_train['inputs'] = img
    return data_train


def get_loss_from_dict(model_name, loss_dict):
    """Return the correct loss based on the model type
    Args:
        model_name (~ str): the mmdet model name, eg: 'Faster R-CNN', 'YOLOv3', 'RetinaNet', 'FreeAnchor' ...
        loss_dict (~ dict): the loss of the model, stored in a dictionary
    Returns:
        losses (~ torch.Tensor): the summation of the loss
    """
    if model_name in ['Faster R-CNN', 'Libra R-CNN', 'GN+WS', 'FasterRN101']:
        losses = loss_dict['loss_cls'] + loss_dict['loss_bbox'] + sum(loss_dict['loss_rpn_cls']) + sum(loss_dict['loss_rpn_bbox'])
        # losses = sum(loss_dict.values())
    elif model_name in ['Grid R-CNN']:
        losses = loss_dict['loss_cls'] + sum(loss_dict['loss_rpn_cls']) + sum(loss_dict['loss_rpn_bbox'])
    elif model_name in ['YOLOv3', 'RetinaNet', 'RepPoints', 'SSD', 'YOLOv3MN', 'RetinaNetRN101', 'RetinaNetX101']:
        losses = sum(sum(loss_dict[key]) for key in loss_dict)
    else: # ['FreeAnchor', 'DETR', 'CenterNet', 'YOLOX', 'FoveaBox']
        losses = sum(loss_dict.values())
    return losses


class model_train(torch.nn.Module):
    """return a model in train mode, such that we can get the loss
    Args:
        input the same config_file, checkpoint_file as test models
        device (~ str): indicates which gpu to allocate
    """

    def __init__(self, model_name, device='cuda:0', dataset='voc') -> None:
        super().__init__()
        config_file = model_info[model_name]['config_file']
        checkpoint_file = model_info[model_name]['checkpoint_file']
        config = mmengine.Config.fromfile(config_file)
        model_train = inference_detector(config, device = device, weights=checkpoint_file, show_progress=False)

        self.model = model_train
        self.model_name = model_name
        self.device = device
        self.conf_thres = get_conf_thres(model_name)
        self.dataset = dataset

    def forward(self, x):
        """inference model using image x
        Args:
            x (numpy.ndarray): input image
            result (list): a list of output from mmdet model
        """
        result = self.model(x)
        return result

    def loss(self, im_path, x, pert, bboxes_tgt, labels_tgt):
        """get the loss

        args:
            x (numpy.ndarray):
            pert (tensor):             

        """
        #result = self.model(x.astype(np.uint8), texts="$: coco")
        #result = output2det(result, x, conf_thres=self.conf_thres, dataset=self.dataset)

        data = get_test_data(self.model, im_path, x)
        data_train = get_train_data(self.model, x, pert.to(self.device), self.device, data, bboxes_tgt, labels_tgt)

        batch_data_samples = self.model.model.predict(batch_inputs =data_train['inputs'], batch_data_samples = [data_train['data_samples']])
        #batch_data_samples = [data_train['data_samples']]
        self.model.model.training = True
        loss_dict = self.model.model.loss(batch_inputs = data_train['inputs'], batch_data_samples = batch_data_samples)
        self.model.model.training = False
        loss = get_loss_from_dict(self.model_name, loss_dict)

        # print(f"loss_dict: {loss_dict}")
        #loss = get_loss_from_dict(self.model_name, loss_dict)

        return loss

    def rgb(self):
        to_rgb = False # false by default
        for item in self.model.cfg.data.test.pipeline[1]['transforms']:
            if 'to_rgb' in item:
                to_rgb = item['to_rgb']
                return to_rgb

    def det(self, x):
        """inference model using image x, get the processed output as detection
        
        args:
            x (numpy.ndarray): input image
        """
        result = self.model(x, texts="$: coco")
        det = output2det(result, x, conf_thres=self.conf_thres, dataset=self.dataset)
        return det


def get_train_model(config_file, checkpoint_file, device='cuda:0'):
    """return a model in train mode, such that we can get the loss
    Args:
        input the same config_file, checkpoint_file as test models
        device (~ str): indicates which gpu to allocate
    """
    import mmcv
    from mmdet.models import build_detector
    from mmcv.runner import load_checkpoint
    config = mmcv.Config.fromfile(config_file)
    model_train = build_detector(config.model, test_cfg=config.get('test_cfg'))
    map_loc = 'cpu' if device == 'cpu' else None
    checkpoint = load_checkpoint(model_train, checkpoint_file, map_location=map_loc)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model_train.CLASSES = checkpoint['meta']['CLASSES']
    #model_train.cfg = config  # save the config in the model for convenience
    model_train.to(device)
    # model_train.train()
    model_train.eval()
    return model_train


def get_iou(bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        bbox1 (numpy.ndarray): x1,y1,x2,y2
        bbox2 (numpy.ndarray): x1,y1,x2,y2
    Returns:
        iou (float): iou in [0, 1]
    """
    w1,h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    w2,h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    try:
        assert all([w1,h1,w2,h2])
    except:
        return 0

    # determine the coordinates of the intersection rectangle
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = w1*h1
    area2 = w2*h2
    iou = area_inter / float(area1 + area2 - area_inter)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def is_success(detections, target_clean, iou_threshhold=0.9):
    """ see if the detection has target label at the corresponding location with IOU > 0.3
    Args:
        detections (np.ndarray): a list of detected objects. Shape (n,6)
        target_clean (np.ndarray): a single object, our desired output. Shape (1,6) - [xyxy,cls,score]
    Returens:
        (bool): whether the detection is a success or not
    """
    for items in detections:
        iou = get_iou(items, target_clean[0])
        if iou > iou_threshhold and items[4] == target_clean[0][4]:
            return True
    return False


def is_success_hiding(detections):
    """ if nothing is detected it is a success
    Args:
        detections (np.ndarray): a list of detected objects. Shape (n,6)
    Returens:
        (bool): whether the detection is a success or not
    """
    if len(detections) == 0:
        return True
    else:
        return False