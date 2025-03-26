import sys
import argparse
import os
import os.path as osp
import time
import warnings
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.ppal.datasets import *
from mmdet.ppal.models import *
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
from torchvision.transforms.functional import gaussian_blur
import copy
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
import re
from torchvision.ops import roi_align
import torchvision.ops as ops
global_avg_pool = nn.AdaptiveAvgPool2d(1)

def add_perturbation(image, method='random_noise', noise_std=0.2):
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch.Tensor")
    if method == 'random_noise':
        noise = torch.randn_like(image) * noise_std
        perturbed = image + noise
        perturbed = torch.clamp(perturbed, min1, max1)
    elif method == 'gaussian_blur':
        perturbed = gaussian_blur(image, kernel_size=5)
    else:
        raise ValueError(f"Unknown perturbation method: {method}")
    return perturbed

def simple_test1(model, feat, img_metas, rescale=False):
    results_list = model.bbox_head.simple_test(feat, img_metas, rescale=rescale)
    return results_list

def extract_roi_features1(feat, bboxes, img_shape,json_bboxes_label):
    roi_features = {}
    feature_map = feat[-1]
    stride_h = img_shape[0] / feature_map.shape[2]
    stride_w = img_shape[1] / feature_map.shape[3]
    feature_map_bg = feature_map.clone()
    for i in range(len(bboxes)):
        bboxes_info = bboxes[i]
        bboxes_label = json_bboxes_label[i]
        scaled_bboxes = bboxes_info.clone().float()
        scaled_bboxes[0] /= stride_w
        scaled_bboxes[2] /= stride_w
        scaled_bboxes[1] /= stride_h
        scaled_bboxes[3] /= stride_h
        x1, y1, x2, y2 = scaled_bboxes
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, feature_map.shape[3]), min(y2, feature_map.shape[2])
        roi_feature = feature_map[:, :, y1:y2, x1:x2]
        roi_feature = global_avg_pool(roi_feature)
        if bboxes_label not in roi_features:
            roi_features[bboxes_label] = []
        roi_features[bboxes_label].append(roi_feature)
        feature_map_bg[:, :, y1:y2, x1:x2]  = 0
    feature_map_bg = global_avg_pool(feature_map_bg)
    return roi_features, feature_map_bg

def extract_roi_features(feat, bboxes, img_shape):
    feature_map = feat[-1]
    batch_size, num_channels, fm_h, fm_w = feature_map.shape
    stride_h = img_shape[0] / fm_h
    stride_w = img_shape[1] / fm_w
    all_boxes = []
    categories = []
    for bbox_info in bboxes:
        cat = bbox_info['category']
        boxes = torch.from_numpy(bbox_info['bboxes']).float()
        num_boxes = boxes.shape[0]
        scaled_boxes = boxes.clone()
        scaled_boxes[:, [0, 2]] /= stride_w
        scaled_boxes[:, [1, 3]] /= stride_h
        scaled_boxes[:, [0, 2]] = torch.clamp(scaled_boxes[:, [0, 2]].round(), 0, fm_w)
        scaled_boxes[:, [1, 3]] = torch.clamp(scaled_boxes[:, [1, 3]].round(), 0, fm_h)
        scaled_boxes[:, 2] = torch.max(scaled_boxes[:, 2], scaled_boxes[:, 0] + 1)
        scaled_boxes[:, 3] = torch.max(scaled_boxes[:, 3], scaled_boxes[:, 1] + 1)
        all_boxes.append(scaled_boxes)
        categories.extend([cat]*num_boxes)
    
    if not all_boxes:
        return []
    all_boxes = torch.cat(all_boxes, dim=0)
    rois = torch.cat([
        torch.zeros(len(all_boxes), 1),
        all_boxes
    ], dim=1).to(feature_map.device)
    pooled_features = ops.roi_align(
        input=feature_map,
        boxes=rois,
        output_size=(1, 1),
        spatial_scale=1.0,
        aligned=True
    ).squeeze(-1).squeeze(-1)
    roi_features = [
        {"category": cat, "feature": feat} 
        for cat, feat in zip(categories, pooled_features)
    ]
    return roi_features

def extract_bboxes(pred_results, score_threshold=0.0):
    all_bboxes = []
    for category_idx, bboxes in enumerate(pred_results[0]):
        filtered_bboxes = bboxes[bboxes[:, 4] >= score_threshold]
        if filtered_bboxes.shape[0] > 0:
            coords = filtered_bboxes[:, :4]
            all_bboxes.append({
                "category": category_idx,
                "bboxes": coords
            })
    return all_bboxes

def calculate_bbox_diff_tensor(result, perturbed_results):
    assert all(len(result) == len(perturbed_result) for perturbed_result in perturbed_results), \
        "All perturbed results should have the same number of classes as result."
    all_differences = {
        f'perturbed_result_{i+1}': {} for i in range(len(perturbed_results))
    }
    for i, class_result in enumerate(result[0]):
        num_boxes_result = len(class_result)
        class_result = torch.from_numpy(class_result)
        for perturbed_idx, perturbed_result in enumerate(perturbed_results):
            class_perturbed_result = torch.from_numpy(perturbed_result[0][i])
            num_boxes_perturbed = len(class_perturbed_result)
            class_result1 = class_result
            if num_boxes_perturbed < num_boxes_result:
                padding = torch.zeros(num_boxes_result - num_boxes_perturbed, 7)
                class_perturbed_result = torch.cat([class_perturbed_result, padding], dim=0)
            elif num_boxes_perturbed > num_boxes_result:
                class_perturbed_result = class_perturbed_result[:num_boxes_result]  
            diff = torch.norm(class_result1[:, :4] - class_perturbed_result[:, :4], dim=1)
            if i not in all_differences[f'perturbed_result_{perturbed_idx+1}']:
                all_differences[f'perturbed_result_{perturbed_idx+1}'][i] = diff.tolist()
            else:
                all_differences[f'perturbed_result_{perturbed_idx+1}'][i].extend(diff.tolist())
    return all_differences


def calculate_angle_and_cosine(a, b):
    dot_product = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    cosine = abs(dot_product / (norm_a * norm_b))
    return cosine

def extract_cls_uncertainty(pred_results, score_threshold):
    all_bboxes = []
    for category_idx, bboxes in enumerate(pred_results[0]):
        filtered_bboxes = bboxes[bboxes[:, 4] >= score_threshold]
        if filtered_bboxes.shape[0] > 0:
            coords = filtered_bboxes[:, 5]
            all_bboxes.append({
                "category": category_idx,
                "cls_uncertainty": coords
            })
    return all_bboxes

def single_gpu_test1(model,
                    data_loader,
                    prototype_features_gt,
                    prototype_feature_bg,
                    regression_vetore_mean,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    dx, dy, dw, dh = regression_vetore_mean
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            result_1 = []
            img_tensor = data['img'][0].data.cuda()
            perturbed_data1 = add_perturbation(data['img'][0], method='random_noise', noise_std=0.2)
            perturbed_data2 = add_perturbation(data['img'][0], method='random_noise', noise_std=0.4)
            perturbed_data3 = add_perturbation(data['img'][0], method='random_noise', noise_std=0.6)
            perturbed_data4 = add_perturbation(data['img'][0], method='random_noise', noise_std=0.8)
            perturbed_data5 = add_perturbation(data['img'][0], method='random_noise', noise_std=1.0)
            perturbed_data_1 = copy.deepcopy(data)
            perturbed_data_2 = copy.deepcopy(data)
            perturbed_data_3 = copy.deepcopy(data)
            perturbed_data_4 = copy.deepcopy(data)
            perturbed_data_5 = copy.deepcopy(data)
            perturbed_data_1['img'][0] = perturbed_data1
            perturbed_data_2['img'][0] = perturbed_data2
            perturbed_data_3['img'][0] = perturbed_data3
            perturbed_data_4['img'][0] = perturbed_data4
            perturbed_data_5['img'][0] = perturbed_data5
            perturbed_data_1['img'][0] = perturbed_data_1['img'][0].cuda()
            perturbed_data_2['img'][0] = perturbed_data_2['img'][0].cuda()
            perturbed_data_3['img'][0] = perturbed_data_3['img'][0].cuda()
            perturbed_data_4['img'][0] = perturbed_data_4['img'][0].cuda()
            perturbed_data_5['img'][0] = perturbed_data_5['img'][0].cuda()
            img_bacnbone = model.extract_feat(img_tensor)
            img_shape = data['img_metas'][0].data[0]['img_shape']
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
                perturbed_result1 = model(return_loss=False, rescale=True, **perturbed_data_1)
                perturbed_result2 = model(return_loss=False, rescale=True, **perturbed_data_2)
                perturbed_result3 = model(return_loss=False, rescale=True, **perturbed_data_3)
                perturbed_result4 = model(return_loss=False, rescale=True, **perturbed_data_4)
                perturbed_result5 = model(return_loss=False, rescale=True, **perturbed_data_5)
                perturbed_result = [perturbed_result1, perturbed_result2, perturbed_result3,perturbed_result4, perturbed_result5]
            all_differences = calculate_bbox_diff_tensor(result, perturbed_result)
            all_differences1 = [value for values in all_differences['perturbed_result_1'].values() if values for value in values]
            all_differences2 = [value for values in all_differences['perturbed_result_2'].values() if values for value in values] 
            all_differences3 = [value for values in all_differences['perturbed_result_3'].values() if values for value in values]
            all_differences4 = [value for values in all_differences['perturbed_result_4'].values() if values for value in values]
            all_differences5 = [value for values in all_differences['perturbed_result_5'].values() if values for value in values]
            bboxes = extract_bboxes(result, score_threshold=0.00)
            predict_box_feature  = extract_roi_features(img_bacnbone, bboxes, img_shape)
            predict_box_feature_only = [item['feature'] for item in predict_box_feature if 'feature' in item]
            cls_uncertainty = extract_cls_uncertainty(result, score_threshold=0.00)
            cls_uncertainty_list = [value for entry in cls_uncertainty for value in entry['cls_uncertainty']]
            all_differences1 = torch.tensor(all_differences1).cuda(0)
            all_differences2 = torch.tensor(all_differences2).cuda(0)
            all_differences3 = torch.tensor(all_differences3).cuda(0)
            all_differences4 = torch.tensor(all_differences4).cuda(0)
            all_differences5 = torch.tensor(all_differences5).cuda(0)
            directions  = torch.stack([dx, dy, dw, dh])
            prototypes = torch.stack(list(prototype_features_gt.values())).squeeze().cuda(0)
            prototype_feature_bg = prototype_feature_bg.cuda(0)
            n = len(predict_box_feature_only)
            regression_information = [None] * n
            classification_information = [None] * n
            regression_weights = [None] * n
            classification_weights = [None] * n
            anchore_weights = [None] * n
            for z in range(n):
                feature = predict_box_feature_only[z].squeeze().cuda(0)
                cos_sims = torch.stack([
                    calculate_angle_and_cosine(feature, directions[i]) 
                    for i in range(4)
                ])
                weight = cos_sims.mean()
                junzhi = ( all_differences1[z] + all_differences2[z] + all_differences3[z] +all_differences4[z] + all_differences5[z] )/5
                offset = torch.tensor([all_differences1[z], all_differences2[z],all_differences3[z],all_differences4[z], all_differences5[z]])
                offset_softmax = F.softmax(offset, dim=0)
                information_anchor = -torch.sum(offset_softmax * torch.log(offset_softmax+ 1e-9))
                classification_weight = torch.norm(feature - prototype_feature_bg, p=2)
                distances = torch.norm(feature - prototypes, p=2, dim=1)
                probabilities = F.softmax(-distances, dim=0)
                dis_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9))
                regression_information[z] = junzhi / (information_anchor+1e-9)
                classification_information[z] = cls_uncertainty_list[z]
                regression_weights[z] = weight
                classification_weights[z] = classification_weight
                anchore_weights[z] = dis_entropy
            anchore_weights = torch.stack(anchore_weights).cuda()
            regression_weights = torch.stack(regression_weights).cuda()
            regression_information = torch.stack(regression_information).cuda() 
            classification_weights = torch.stack(classification_weights).cuda()
            classification_information = [torch.as_tensor(x) for x in classification_information]
            classification_information = torch.stack(classification_information).cuda() 
            array = torch.stack([anchore_weights, regression_weights, regression_information, classification_weights, classification_information], dim=0)
            result_0 = array.T 
            result_0 = result_0.cpu().numpy()
            result_1.append(result_0)
            batch_size = 1
            results.append(result_1)
            for _ in range(batch_size):
                prog_bar.update()
            del perturbed_data1, perturbed_data2, img_bacnbone, result, perturbed_result1, perturbed_result2, all_differences
            torch.cuda.empty_cache()
    return results


def get_bounding_boxes(image_name, json_data):
    image_id = None
    for image in json_data['images']:
        if image['file_name'] == image_name:
            image_id = image['id']
            break
    
    if image_id is None:
        return f"Image {image_name} not found in the dataset."
    bounding_boxes = []
    bounding_boxes_label = []
    for annotation in json_data['annotations']:
        if annotation['image_id'] == image_id:
            bounding_boxes.append(annotation['bbox'])
            bounding_boxes_label.append(annotation['category_id'])
    return bounding_boxes, bounding_boxes_label

def extract_file_name(file_path):
    return file_path.split('/')[-1]

def single_gpu_test1_prototype(model,
                    data_loader,
                    round_name,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    with open(f'/work_dirs/retinanet_voc_7rounds_5percent_to_20percent/{round_name}/annotations/labeled.json', 'r') as f:
        json_data = json.load(f)
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    prototype_feature_bg = None
    category_prototypes = {}
    prototype_features_gt = {}
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img_tensor = data['img'][0].data.cuda()
            img_metas = data['img_metas'][0].data[0]
            image_name1 = img_metas[0]['filename']
            image_name1 = extract_file_name(image_name1)
            json_bboxes, json_bboxes_label = get_bounding_boxes(image_name1, json_data)
            GT_feature, feature_map_bg = extract_roi_features1(model.extract_feat(img_tensor), torch.tensor(json_bboxes).cuda(), data['img_metas'][0].data[0][0]['img_shape'],json_bboxes_label)
            if prototype_feature_bg is None:
                prototype_feature_bg = feature_map_bg
            else:
                prototype_feature_bg = (prototype_feature_bg + feature_map_bg) / 2
            for category_id, features in GT_feature.items():
                if category_id not in category_prototypes:
                    category_prototypes[category_id] = {
                        "sum": sum(features),
                        "count": len(features)
                }
                else:
                    category_prototypes[category_id]["sum"] += sum(features)
                    category_prototypes[category_id]["count"] += len(features)
            prog_bar.update(1)
            del img_tensor
            torch.cuda.empty_cache()
        for category_id, data in category_prototypes.items():
            prototype_features_gt[category_id] = data["sum"] / data["count"]
    return prototype_features_gt, prototype_feature_bg
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    args = parse_args()
    workdirs = args.checkpoint
    round_name = workdirs.split('/')[-1]
    
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    last_number = re.search(r'\d+$', round_name).group()
    last_number = int(last_number)
    test_cycle = f'test{last_number}'
    dataset1 = build_dataset(cfg['data'][test_cycle])
    data_loader1 = build_dataloader(
        dataset1,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    regression_vetore = model.bbox_head.retina_reg.weight
    regression_vetore = global_avg_pool(regression_vetore)
    regression_vetore = regression_vetore.view(9, 4, 1028)
    dx = regression_vetore[:, 0, :]
    dy = regression_vetore[:, 1, :]
    dw = regression_vetore[:, 2, :]
    dh = regression_vetore[:, 3, :]
    dx = torch.mean(dx, dim=0, keepdim=True).squeeze()
    dy = torch.mean(dy, dim=0, keepdim=True).squeeze()
    dw = torch.mean(dw, dim=0, keepdim=True).squeeze()
    dh = torch.mean(dh, dim=0, keepdim=True).squeeze()

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
        prototype_features_gt,prototype_feature_bg  = single_gpu_test1_prototype(model, data_loader1, round_name, args.show, args.show_dir,
                                  args.show_score_thr)
        prototype_feature_bg = prototype_feature_bg.squeeze()
        regression_vetore_mean = [dx.cuda(),dy.cuda(),dw.cuda(),dh.cuda()]
        outputs1 = single_gpu_test1(model, data_loader,prototype_features_gt,prototype_feature_bg, regression_vetore_mean, args.show, args.show_dir,
                                    args.show_score_thr)
        outputs_with_info = []
        for sample_output, sample_output1 in zip(outputs, outputs1):
            merged_sample = []
            start = 0
            for frame_output in sample_output:
                length = len(frame_output)
                if len(frame_output) == 0:
                    merged_sample.append([])
                    continue
                merged_frame = np.concatenate((frame_output, sample_output1[0][start:start + length]), axis=1)
                merged_sample.append(merged_frame)
                start = start + length
            outputs_with_info.append(merged_sample)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    rank, _ = get_dist_info()  
    if rank == 0: 
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs_with_info, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)
if __name__ == '__main__':
    main()
