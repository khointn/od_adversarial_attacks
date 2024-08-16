"""
    Attack object detectors in a blackbox setting
    design blackbox loss
    clean code for running ebad
    TODO: Log runtime, memory usage.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from pathlib import Path
import random
import time
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

from utils.utils_ebad import VOC_BBOX_LABEL_NAMES, COCO_BBOX_LABEL_NAMES
from utils.utils_ebad import model_train, get_loss_and_success_list
from utils.utils_ebad import PM_tensor_weight_balancing_np

import yaml
general_config = yaml.safe_load(Path("general_config.yml").read_text())

def parse_arg():

    parser = argparse.ArgumentParser(description="generate perturbations")
    parser.add_argument("--eps", type=int, default=10, help="perturbation level: 10,20,30,40,50")
    parser.add_argument("--iters", type=int, default=10, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--root", type=str, default='result', help="the folder name of result")
    parser.add_argument("--victim", type=str, default='RetinaNet', help="victim model") #RetinaNet
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--n_wb", type=int, default=2, help="number of models in the ensemble")
    parser.add_argument("--surrogate", type=str, default='Faster R-CNN', help="surrogate model when n_wb=1")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of w")
    parser.add_argument("--iterw", type=int, default=5, help="iterations of updating w")
    parser.add_argument("--dataset", type=str, default='coco', help="model dataset 'voc' or 'coco'. This will change the output range of detectors.")
    parser.add_argument("-single", action='store_true', help="only care about one obj")
    parser.add_argument("-no_balancing", action='store_true', help="do not balance weights at beginning")
    args = parser.parse_args()

    return args

def pre_setup():
    args = parse_arg()
    print(f"args.single: {args.single}")
    eps = args.eps
    n_iters = args.iters
    x_alpha = args.x
    alpha = eps / n_iters * x_alpha
    iterw = args.iterw
    n_wb = args.n_wb
    lr_w = args.lr
    dataset = args.dataset
    victim_name = args.victim
    single = args.single
    no_balancing = args.no_balancing
    surrogate = (args.surrogate if n_wb==1 else None)

    # name experiment
    exp_name = f'{dataset}_wb_{n_wb}_linf_{eps}_iters{n_iters}_alphax{x_alpha}_victim_{victim_name}_lr{lr_w}_iterw{iterw}'
    if surrogate != None:
        exp_name += f'_{surrogate}'
    if single:
        exp_name += '_single'
    if no_balancing:
        exp_name += '_nobalancing'
    print(f"\nExperiment: {exp_name} \n")
    
    # create folders for saving adversarial results
    result_root = Path(general_config['results_root']) / "ebad/"
    result_root.mkdir(parents=True, exist_ok=True)
    exp_root = result_root / exp_name
    adv_fail_root = exp_root / 'advs_fail'
    adv_fail_root.mkdir(parents=True, exist_ok=True)
    adv_success_root = exp_root / 'advs_success'
    adv_success_root.mkdir(parents=True, exist_ok=True)
    adv_nodet_root = exp_root / 'advs_notdet'
    adv_nodet_root.mkdir(parents=True, exist_ok=True)

    # read original images
    data_root = Path(general_config["data_root"])
    match dataset:
        case 'voc':
            dataset_num_labels = 20
            dataset_labels = VOC_BBOX_LABEL_NAMES
            im_root = data_root / "VOCdevkit/VOC2007/JPEGImages/"
            test_image_ids_path = data_root / "VOCdevkit/VOC2007/ImageSets/Main/test.txt"
            with open(test_image_ids_path) as f:
                test_image_ids = f.read().splitlines()
        case 'coco':
            dataset_num_labels = 80
            dataset_labels = COCO_BBOX_LABEL_NAMES
            im_root = data_root / "coco/val2017/"
            for (root, dirs, files) in os.walk(im_root):
                if files:
                    test_image_ids = files

    # check already run images
    already_run = []
    for (_, dirs, files) in os.walk(adv_success_root):
        if files:
            already_run = files
    for (_, dirs, files) in os.walk(adv_fail_root):
        if files:
            already_run.extend(files)
    for (_, dirs, files) in os.walk(adv_nodet_root):
        if files:
            already_run.extend(files)

    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(f'{exp_root / exp_name}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    dirs = {
        "experiment": exp_root,
        "original": im_root,
        "already_run": already_run,
        "save_success": adv_success_root,
        "save_failed": adv_fail_root,
        "save_nodet": adv_nodet_root
    }

    adv_params = {
        "eps": eps,
        "n_iters": n_iters,
        "x_alpha": x_alpha,
        "alpha": eps / n_iters * x_alpha,
        "iterw": iterw,
        "n_wb": n_wb,
        "lr_w": lr_w,
        "single": single,
        "no_balancing": no_balancing,
        "victim_name": victim_name,
        "dataset_name": dataset,
        "dataset_labels": dataset_labels,
        "dataset_num_labels": dataset_num_labels,
        "surrogate": surrogate,
    }

    return test_image_ids, exp_name, dirs, adv_params, logger

def main():
    test_image_ids, exp_name, dirs, adv_params, logger = pre_setup()

    # load surrogate models
    ensemble = []
    models_all = ['YOLOv3', 'Faster R-CNN', 'RetinaNet', 'FCOS', 'SSD', 'Grid R-CNN']
    model_list = models_all[:adv_params['n_wb']]
    if adv_params['n_wb'] == 1:
        model_list = [adv_params['surrogate']]
    for model_name in model_list:
        ensemble.append(model_train(model_name=model_name, dataset=adv_params['dataset_name']))

    # load victim model
    # ['RetinaNet', 'Libra', 'FoveaBox', 'FreeAnchor', 'DETR', 'Deformable']
    if adv_params['victim_name'] == 'Libra':
        adv_params['victim_name'] = 'Libra R-CNN'
    elif adv_params['victim_name'] == 'Deformable':
        adv_params['victim_name'] = 'Deformable DETR'
    elif adv_params['victim_name'] == 'Faster':
        adv_params['victim_name'] = 'Faster R-CNN'

    model_victim = model_train(model_name=adv_params['victim_name'], dataset=adv_params['dataset_name'])
    all_model_names = model_list + [adv_params['victim_name']]
    all_models = ensemble + [model_victim]

    total_runtime = []
    dict_k_sucess_id_v_query = {} # query counts of successful im_ids
    dict_k_valid_id_v_success_list = {} # lists of success for all mdoels for valid im_ids

    for im_idx, im_id in enumerate(tqdm(test_image_ids)):
        if '.jpg' in im_id:
            im_id = im_id[:-4]
        
        if f'{im_id}.jpg' not in dirs['already_run']:
            im_path = dirs['original'] / f"{im_id}.jpg"
            im_np = np.array(Image.open(im_path).convert('RGB'))
            start_time = time.time()
            
            # get detection on clean images and determine target class
            det = model_victim.det(im_np)
            bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
            print(f"n_objects: {len(det)}")
            if len(det) == 0: # if nothing is detected, skip this image
                adv_nodet_path = dirs['save_nodet'] / f"{im_id}.jpg"
                adv_png = Image.fromarray(im_np.astype(np.uint8))
                adv_png.save(adv_nodet_path)
                continue
            else:
                dict_k_valid_id_v_success_list[im_id] = []

            all_categories = set(labels.astype(int))  # all apperaing objects in the scene
            # randomly select a victim
            victim_idx = random.randint(0,len(det)-1)
            victim_class = int(det[victim_idx,4])

            # randomly select a target
            select_n = 1 # for each victim object, randomly select 5 target objects
            target_pool = list(set(range(adv_params['dataset_num_labels'])) - all_categories)
            target_pool = np.random.permutation(target_pool)[:select_n]

            # for target_class in target_pool:
            target_class = int(target_pool[0])

            # basic information of attack
            info = f"im_idx: {im_idx}, im_id: {im_id}, victim_class: {adv_params['dataset_labels'][victim_class]}, target_class: {adv_params['dataset_labels'][target_class]}\n"
            logger.info(info)

            target = det.copy()
            # only change one label
            target[victim_idx, 4] = target_class
            # only keep one label
            target_clean = target[victim_idx,:][None]

            if adv_params['single']: # only care about the target object
                target = target_clean    
            

            if adv_params['no_balancing']:
                print(f"no_balancing, using equal weights")
                w_inv = np.ones(adv_params['n_wb']) 
                w_np = np.ones(adv_params['n_wb']) / adv_params['n_wb']
            else:
                # determine the initial w, via weight balancing
                dummy_w = np.ones(adv_params['n_wb'])
                _, LOSS = PM_tensor_weight_balancing_np(im_path, im_np, target, dummy_w, ensemble, 
                                                        adv_params['eps'], n_iters=1, alpha=adv_params['alpha'], dataset=adv_params['dataset_name'])
                loss_list_np = [LOSS[name][0] for name in model_list]
                w_inv = 1 / np.array(loss_list_np)
                w_np = w_inv / w_inv.sum()
                print(f"w_np: {w_np}")


            adv_np, LOSS = PM_tensor_weight_balancing_np(im_path, im_np, target, w_np, ensemble, 
                                                         adv_params['eps'], n_iters=adv_params['n_iters'], alpha=adv_params['alpha'], dataset=adv_params['dataset_name'])
            n_query = 0
            loss_bb, success_list = get_loss_and_success_list(im_np, adv_np, LOSS, target_clean, all_models)
            dict_k_valid_id_v_success_list[im_id].append(success_list)

            adv_success_path = dirs['save_success'] / f"{im_id}.jpg"
            adv_fail_path = dirs['save_failed'] / f"{im_id}.jpg"

            # stop whenever successful
            if success_list[-1]:
                dict_k_sucess_id_v_query[im_id] = n_query
                print(f"success! image im idx: {im_idx}")
                
                w_list = []
                loss_bb_list = [loss_bb]
                loss_ens_list = LOSS['ens'] # ensemble losses during training
                adv_png = Image.fromarray(adv_np.astype(np.uint8))
                adv_png.save(adv_success_path)
            else:

                n_query += 1

                w_list = []        
                loss_bb_list = [loss_bb]
                loss_ens_list = LOSS['ens'] # ensemble losses during training

                idx_w = 0 # idx of wb in W, rotate
                while n_query < adv_params['iterw']:

                    ##################################### query plus #####################################
                    w_np_temp_plus = w_np.copy()
                    w_np_temp_plus[idx_w] += adv_params['lr_w'] * w_inv[idx_w]
                    adv_np_plus, LOSS_plus = PM_tensor_weight_balancing_np(im_path, im_np, target, w_np_temp_plus, ensemble, 
                                                                           adv_params['eps'], adv_params['n_iters'], alpha=adv_params['alpha'], dataset=adv_params['dataset_name'], adv_init=adv_np)
                    loss_bb_plus, success_list = get_loss_and_success_list(im_np, adv_np_plus, LOSS_plus, target_clean, all_models)
                    dict_k_valid_id_v_success_list[im_id].append(success_list)

                    n_query += 1
                    print(f"iter: {n_query}, {idx_w} +, loss_bb: {loss_bb_plus}")

                    # stop whenever successful
                    if success_list[-1]:
                        dict_k_sucess_id_v_query[im_id] = n_query
                        print(f"success! image im idx: {im_idx}")
                        loss_bb = loss_bb_plus
                        loss_ens = LOSS_plus["ens"]
                        w_np = w_np_temp_plus
                        adv_np = adv_np_plus
                        adv_png = Image.fromarray(adv_np.astype(np.uint8))
                        adv_png.save(adv_success_path)

                        break

                    #######################################################################################
                    
                    ##################################### query minus #####################################
                    w_np_temp_minus = w_np.copy()
                    w_np_temp_minus[idx_w] -= adv_params['lr_w'] * w_inv[idx_w]
                    adv_np_minus, LOSS_minus = PM_tensor_weight_balancing_np(im_path, im_np, target, w_np_temp_minus, ensemble, 
                                                                             adv_params['eps'], adv_params['n_iters'], alpha=adv_params['alpha'], dataset=adv_params['dataset_name'], adv_init=adv_np)
                    loss_bb_minus, success_list = get_loss_and_success_list(im_np, adv_np_minus, LOSS_minus, target_clean, all_models)
                    dict_k_valid_id_v_success_list[im_id].append(success_list)

                    n_query += 1
                    print(f"iter: {n_query}, {idx_w} -, loss_bb: {loss_bb_minus}")

                    # stop whenever successful
                    if success_list[-1]:
                        dict_k_sucess_id_v_query[im_id] = n_query
                        print(f"success! image im idx: {im_idx}")
                        loss_bb = loss_bb_minus
                        loss_ens = LOSS_minus["ens"]
                        w_np = w_np_temp_minus
                        adv_np = adv_np_minus
                        adv_png = Image.fromarray(adv_np.astype(np.uint8))
                        adv_png.save(adv_success_path)
                        break

                    #######################################################################################

                    ##################################### update w, adv #####################################
                    if loss_bb_plus < loss_bb_minus:
                        loss_bb = loss_bb_plus
                        loss_ens = LOSS_plus["ens"]
                        w_np = w_np_temp_plus
                        adv_np = adv_np_plus
                    else:
                        loss_bb = loss_bb_minus
                        loss_ens = LOSS_minus["ens"]
                        w_np = w_np_temp_minus
                        adv_np = adv_np_minus

                    # relu and normalize
                    w_np = np.maximum(0, w_np)
                    w_np = w_np + 0.005 # minimum set to 0.005
                    w_np = w_np / w_np.sum()
                    #######################################################################################

                    idx_w = (idx_w+1)%adv_params['n_wb']
                    w_list.append(w_np.tolist())
                    loss_bb_list.append(loss_bb)
                    loss_ens_list += loss_ens
                
                if not success_list[-1]:
                    print("Failed:", im_id)
                    adv_png = Image.fromarray(adv_np.astype(np.uint8))
                    adv_png.save(adv_fail_path)

            end_time = time.time() - start_time
            print(f"Iter {n_query} running time: {round(end_time, 5)}s")
            total_runtime.append(end_time)


            if im_id in dict_k_sucess_id_v_query:
                # save to txt
                info = f"im_idx: {im_idx}, id: {im_id}, query: {n_query}, loss_bb: {loss_bb:.4f}, w: {w_np}\n"
                logger.info(info)
            print(f"im_idx: {im_idx}; total_success: {len(dict_k_sucess_id_v_query)}")

            if len(dict_k_sucess_id_v_query) > 0:
                query_list = [dict_k_sucess_id_v_query[key] for key in dict_k_sucess_id_v_query]
                #print(f"query_list: {query_list}")
                print(f"avg queries: {np.mean(query_list)}")
                print(f"success rate (victim): {len(dict_k_sucess_id_v_query) / len(dict_k_valid_id_v_success_list)}")

            # print surrogate success rates
            success_list_stack = []
            for valid_id in dict_k_valid_id_v_success_list:
                success_list = np.array(dict_k_valid_id_v_success_list[valid_id])
                success_list = success_list.sum(axis=0).astype(bool).astype(int).tolist()
                success_list_stack.append(success_list)

            success_list_stack = np.array(success_list_stack).sum(axis=0)

            for idx, success_cnt in enumerate(success_list_stack):
                print(f"success rate of {all_model_names[idx]}: {success_cnt / len(dict_k_valid_id_v_success_list)}")


            if len(total_runtime) >0:
                info = f"Average running time: {round(sum(total_runtime)/len(total_runtime), 3)}s"
                logger.info(info)
                print(info)
            else:
                print("Runtime list not updated")
    
    info = f"Average running time: {round(sum(total_runtime)/len(total_runtime), 3)}s\n\
            Running times for every images:\n{total_runtime}"
    logger.info(info)
    print(info)


if __name__ == '__main__':
    main()