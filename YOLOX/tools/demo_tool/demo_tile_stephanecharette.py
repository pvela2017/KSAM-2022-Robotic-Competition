#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
from doctest import OutputChecker
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess, vis

from random import *

import copy

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        # cls_names=COCO_CLASSES,
        cls_names = ["apis", "black", "jangsu", "ggoma", "simil", "crabro"],
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        # self.cls_names = cls_names
        # self.cls_names = ["apis", "black", "crabro", "simil", "jangsu", "ggoma"]
        self.cls_names = ["apis", "black", "jangsu", "ggoma", "simil", "crabro"]
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


    def inference(self, img):
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs



def merge_bbox(inference_data):
    result = copy.deepcopy(inference_data)

    vertical_tile_size = 3
    horizontal_tile_size = 3
    vertival_threshold = 0.10
    # vertival_threshold = 0.15
    horizontal_threshold = 0.10
    # horizontal_threshold = 0.15
    
    merged_result = []

    # Vertical case
    for i in range(vertical_tile_size * horizontal_tile_size):
        if i % 3 == 2:
            continue
        for j in range(len(inference_data[i])):
            tile = inference_data[i][j]
            for idx, target_tile in enumerate(inference_data[i+1]):
                if len(tile) == 0 or len(target_tile) == 0:
                    continue
                # tile_calss == target_tile_class && x1 == x'2 && y1 == y'1 && y2 == y'2 
                # tile_calss == target_tile_class && x2 ==x'1  && y1 == y'1 && y2 == y'2
                # x1, x'2 --> min(x1, x'2)*threshold > max(x1, x'2)
                # add result [x1, min(y1, y'1), max(x2, x'2), max(y2, y'2), class_id]
                if tile[-1] == target_tile[-1] and min(tile[2],target_tile[0])*(1+vertival_threshold)>max(tile[0],target_tile[2]) and min(tile[1],target_tile[1])*(1+vertival_threshold)>max(tile[1], target_tile[1]) and min(tile[3], target_tile[3])*(1+vertival_threshold) > max(tile[3], target_tile[3]):
                    merged_bbox = [tile[0], min(tile[1],target_tile[1]), max(tile[2],target_tile[2]), max(tile[3],target_tile[3]), max(tile[-3], target_tile[-3]) , max(tile[-2], target_tile[-2]), tile[-1]]
                    merged_result.append(merged_bbox)
                    result[i][j] = None
                    result[i+1][idx] = None
                    #print(f'{i}, {j}, {idx}')
                    #print(inference_data[i][j])
                    continue


    # Horizontal case
    for i in range(vertical_tile_size * horizontal_tile_size - vertical_tile_size):
        for j in range(len(inference_data[i])):
            tile = inference_data[i][j]
            for idx, target_tile in enumerate(inference_data[i+horizontal_tile_size]):
                if len(tile) == 0 or len(target_tile) == 0:
                        continue
                # tile_calss == target_tile_class && y'1 == y2 && x1 == x'1 && x2 == x'2
                # x1, x'2 --> min(x1, x'2)*threshold > max(x1, x'2)
                # add result [min(x1, x'1), y1, max(x2, x'2), max(y2, y'2), class_id]
                if tile[-1] == target_tile[-1] and min(tile[3],target_tile[1])*(1+horizontal_threshold)>max(tile[1],target_tile[3]) and min(tile[0],target_tile[0])*(1+horizontal_threshold)>max(tile[0], target_tile[0]) and min(tile[2], target_tile[2])*(1+horizontal_threshold) > max(tile[2], target_tile[2]):
                    merged_bbox = [min(tile[0], target_tile[0]), tile[1], max(tile[2], target_tile[2]), max(tile[3], target_tile[3]), max(tile[-3], target_tile[-3]) , max(tile[-2], target_tile[-2]), tile[-1]]
                    merged_result.append(merged_bbox)
                    result[i][j] = None
                    result[i+horizontal_tile_size][idx] = None
                    #print(f'{i+1} tile case')
                    #print(f'{i}, {j}, {idx}')
    result = sum(result,[])
    result = [item for item in result if item != None]
    return result + merged_result


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        img = cv2.imread(image_name)
        outputs_group = []
        for x in range(3):
            for y in range(3):
                height, width = img.shape[:2]
                img_tile = img[int(height*x/3):int(height*(x+1)/3), int(width*y/3):int(width*(y+1)/3)].copy()
                outputs = predictor.inference(img_tile)
                for output in outputs:
                    if output == None:
                        outputs_group.append([[]])
                    else:
                        output = output.tolist()
                        outputs_group.append(output)      
        img_info = {"id": 0}
        img_info["file_name"] = os.path.basename(image_name)
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        ratio = min(exp.test_size[0] / img_tile.shape[0], exp.test_size[1] / img_tile.shape[1])
        img_info["ratio"] = ratio
        temp = [[0.0, 0.0], [0.0, 1/3], [0.0, 2/3], [1/3, 0.0], [1/3, 1/3], [1/3, 2/3], [2/3, 0.0], [2/3, 1/3], [2/3, 2/3]]



        for tile in range(len(outputs_group)):
            for box in range(len(outputs_group[tile])):
                y, x = temp[tile]
                if tile == 0:
                    continue
                else:
                    try:
                        width_weight = (width * ratio) * x
                        height_weight = (height * ratio) * y
                        outputs_group[tile][box][0] += width_weight
                        outputs_group[tile][box][1] += height_weight
                        outputs_group[tile][box][2] += width_weight
                        outputs_group[tile][box][3] += height_weight
                    except:
                        continue
                continue
        
        # classes = ['apis', 'black', 'crabro', 'simil', 'jangsu', 'ggoma']
        classes = ['apis', 'black', 'jangsu', 'ggoma', 'simil', 'crabro']

        outputs_group = merge_bbox(outputs_group)
        outputs_txt = [[bbox[6], bbox[4], (bbox[0]+bbox[2])/2/960, (bbox[1]+bbox[3])/2/540, (bbox[2]-bbox[0])/960, (bbox[3]-bbox[1])/540] for bbox in outputs_group if len(bbox) == 7]
        save_txt = ''
        for bbox in outputs_txt:
            # save_txt = save_txt + f'{int(bbox[0])} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}\n'
            save_txt = save_txt + f'{classes[int(bbox[0])]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}\n'
            # save_txt = save_txt + f'{int(bbox[0])} {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}\n' #yolo_mark 테스트용
        outputs_group = [bbox for bbox in outputs_group if len(bbox) == 7]
        outputs_group =  torch.tensor(outputs_group)
        result_image = predictor.visual(outputs_group, img_info, predictor.confthre)
        
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            # cv2.imwrite(f'{save_file_name}_{randint(1, 10000)}', result_image)
            cv2.imwrite(f'{save_file_name}', result_image)

            f = open(f'{save_file_name[:-4]}.txt', 'w')
            f.write(save_txt)
            f.close()


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)


