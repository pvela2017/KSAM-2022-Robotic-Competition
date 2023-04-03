#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

import numpy as np

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from camera import Start_Cameras

import socket

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
         "demo", default="capture", help="demo type, eg. image, video and webcam, capture"
     )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # parser.add_argument(
    #     "--path", default="./assets/dog.jpg", help="path to images or video"
    # )
    # parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of counting",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
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
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="gpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
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

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        
        # Coding by dohwan. start
        #
        #
        # bboxes' coordinates
        #print(bboxes[:,0]) # xmin
        #print(bboxes[:,1]) # ymin
        #print(bboxes[:,2]) # xmax
        #print(bboxes[:,3]) # ymax
        #print((bboxes[:,0]+bboxes[:,2])/2) # x center of bboxes
        #print((bboxes[:,1]+bboxes[:,3])/2) # y center of bboxes
        
        # Find biggest tree box
        #
        #
        tree_cls = 2
        tree_bboxes_index = []
        #print("The number of bboxes: ",len(output)) # Find the number of bboxes from 'output' which has information of bboxes (xmin, ymin, xmax, ymax, cls ...)
        
        for i in range(len(output)):
            if output[i, 6] == tree_cls:
                tree_bboxes_index.append(i)
        #print("The number of trees: ", len(tree_bboxes_index)) # Find the number of tree bboxes which class number is '2'(tree_cls is '2', normal=0, diseased=1) 
        #print("The tree bboxes index: ", tree_bboxes_index) # Find the tree index from bboxes
        
        tree_bboxes_area = (bboxes[tree_bboxes_index,2]-bboxes[tree_bboxes_index,0])*(bboxes[tree_bboxes_index,3]-bboxes[tree_bboxes_index,1])
        tree_bboxes_area = tree_bboxes_area.numpy()
        #print("The area of tree bboxes: ",tree_bboxes_area) # Find the index number of tree from bboxes
        
        biggest_tree_bbox_number = np.argmax(tree_bboxes_area)
        biggest_tree_bbox_index = tree_bboxes_index[biggest_tree_bbox_number]
        #print("Index number of biggest tree bbox on the tree bboxes index: ", biggest_tree_bbox_index) # Find the index number of biggest tree
        #
        #
        # Found biggest tree bbox
        
        
        # Find apples(normal and diseased) in biggest tree bbox
        #
        #
        normal_cls = 0
        diseased_cls = 1
        normal_bboxes_index = []
        diseased_bboxes_index = []
        
        xrange_min = bboxes[biggest_tree_bbox_index, 0].numpy()
        yrange_min = bboxes[biggest_tree_bbox_index, 1].numpy()
        xrange_max = bboxes[biggest_tree_bbox_index, 2].numpy()
        yrange_max = bboxes[biggest_tree_bbox_index, 3].numpy()
        #print("Biggest tree bbox's range: ", xrange_min, yrange_min, xrange_max, yrange_max)
        
        for i in range(len(output)):
            if output[i, 6] == normal_cls:
                normal_bboxes_index.append(i)
        #print("The number of normal apples: ", len(normal_bboxes_index)) # Find the number of normal apple bboxes which class number is '0'(normal=0, diseased=1, tree_cls=2) 
        #print("The normal apple bboxes index: ", normal_bboxes_index) # Find the normal apple index from bboxes
        
        normal_bboxes_xcenter_info = (bboxes[normal_bboxes_index,0]+bboxes[normal_bboxes_index,2])/2
        normal_bboxes_xcenter_info = normal_bboxes_xcenter_info.numpy()
        normal_bboxes_ycenter_info = (bboxes[normal_bboxes_index,1]+bboxes[normal_bboxes_index,3])/2
        normal_bboxes_ycenter_info = normal_bboxes_ycenter_info.numpy()
        
        normal_x = np.where((normal_bboxes_xcenter_info >= xrange_min) & (normal_bboxes_xcenter_info <= xrange_max))
        #print("The normal apples' x center pixel on the biggest tree bbox: ", normal_bboxes_xcenter_info[normal_x])
        normal_y = np.where((normal_bboxes_ycenter_info[normal_x] >= yrange_min) & (normal_bboxes_ycenter_info[normal_x] <= yrange_max))
        #print("The normal apples' y center pixel on the biggest tree bbox: ", normal_bboxes_ycenter_info[normal_y])
        
        #print("The number of normal apples on the biggest tree bbox: ", len(normal_bboxes_ycenter_info[normal_y]))

        for i in range(len(output)):
            if output[i, 6] == diseased_cls:
                diseased_bboxes_index.append(i)
        #print("The number of diseased apples: ", len(diseased_bboxes_index)) # Find the number of diseased apple bboxes which class number is '1'(normal=0, diseased=1, tree_cls=2) 
        #print("The diseased apple bboxes index: ", diseased_bboxes_index) # Find the diseased apple index from bboxes
        
        diseased_bboxes_xcenter_info = (bboxes[diseased_bboxes_index,0]+bboxes[diseased_bboxes_index,2])/2
        diseased_bboxes_xcenter_info = diseased_bboxes_xcenter_info.numpy()
        diseased_bboxes_ycenter_info = (bboxes[diseased_bboxes_index,1]+bboxes[diseased_bboxes_index,3])/2
        diseased_bboxes_ycenter_info = diseased_bboxes_ycenter_info.numpy()
                
        diseased_x = np.where((diseased_bboxes_xcenter_info >= xrange_min) & (diseased_bboxes_xcenter_info <= xrange_max))
        #print("The diseased apples' x center pixel on the biggest tree bbox: ", diseased_bboxes_xcenter_info[diseased_x])
        diseased_y = np.where((diseased_bboxes_ycenter_info[diseased_x] >= yrange_min) & (diseased_bboxes_ycenter_info[diseased_x] <= yrange_max))
        #print("The diseased apples' y center pixel on the biggest tree bbox: ", diseased_bboxes_ycenter_info[diseased_y])
        
        #print("The number of diseased apple on the biggest tree bbox: ", len(diseased_bboxes_ycenter_info[diseased_y]))
        
        number_of_normal = len(normal_bboxes_ycenter_info[normal_y])
        number_of_diseased = len(diseased_bboxes_ycenter_info[diseased_y])
        # Found apple(normal and diseased) coordinates and number in biggest tree bbox
        #
        #
        # Coding by dohwan. end
        
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return number_of_normal, number_of_diseased, vis_res

def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        number_of_normal, number_of_diseased, result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
            print("The number of normal apples on the biggest tree bbox: ", number_of_normal)
            print("The number of diseased apples on the biggest tree bbox: ", number_of_diseased)
            
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # save_result : save the counting result using txt library
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            number_of_normal, number_of_diseased, result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            print("The number of normal apples on the biggest tree bbox: ", number_of_normal)
            print("The number of diseased apples on the biggest tree bbox: ", number_of_diseased)  
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
            
def image_capture_demo(predictor, vis_folder, current_time, args):
    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()
    width = 1280  # float
    height = 720  # float
    fps = 30

    HOST = "192.168.1.3"
    PORT = 20001

    counting_array = np.zeros((16,2))
    counting_array = counting_array.astype(int)

    image_base_name = "Image"
    tree_list=['1-l', '5-r', '2-l', '6-r', '3-l', '7-r', '4-l', '8-r', '12-l', '11-l', '10-l', '9-l', '13-r', '14-r', '15-r']

    if args.save_result:
        usb_path = '/media/myusb2/'
        save_folder = os.path.join(usb_path, "Fruit_counting/")
        os.makedirs(save_folder, exist_ok=True)
        logger.info(f"Text save_path is {save_folder}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        
        while True:
            conn, addr = s.accept()
            data = conn.recv(4)
            
            if data:
                trigger = data.decode()
                print(trigger)

            left_grabbed, left_frame = left_camera.read()
            right_grabbed, right_frame = right_camera.read()

            if left_grabbed and right_grabbed:
                if trigger == 0 :
                    print("Pablo, please check your trigger error sending 0")

                elif trigger in tree_list:

                    where_is_tree = trigger.split('-')[1]

                    if where_is_tree in ['l']:
                        left_outputs, left_img_info = predictor.inference(left_frame)
                        number_of_normal_left, number_of_diseased_left, result_frame_left = predictor.visual(left_outputs[0], left_img_info, predictor.confthre)

                        left_tree_number = trigger.split('-')[0]

                        counting_array[left_tree_number-1][0] = counting_array[left_tree_number-1][0] + number_of_normal_left
                        counting_array[left_tree_number-1][1] = counting_array[left_tree_number-1][1] + number_of_diseased_left

                        save_img_folder = os.path.join(save_folder, "%d/" % left_tree_number)
                        os.makedirs(save_img_folder, exist_ok=True)

                        timestr = time.strftime("%Y%m%d_%H%M%S")
                        filename = "_".join([image_base_name, timestr])
                        filename = ".".join([filename,"jpg"])
                        cv2.imwrite(save_img_folder + filename, left_frame)
                    else:
                        right_outputs, right_img_info = predictor.inference(right_frame)
                        number_of_normal_right, number_of_diseased_right, result_frame_right = predictor.visual(right_outputs[0], right_img_info, predictor.confthre)

                        right_tree_number = trigger.split('-')[0]

                        counting_array[right_tree_number-1][0] = counting_array[right_tree_number-1][0] + number_of_normal_right
                        counting_array[right_tree_number-1][1] = counting_array[right_tree_number-1][1] + number_of_diseased_right

                        save_img_folder = os.path.join(save_folder, "%d/" % right_tree_number)
                        os.makedirs(save_img_folder, exist_ok=True)
                        timestr = time.strftime("%Y%m%d_%H%M%S")
                        filename = "_".join([image_base_name, timestr])
                        filename = ".".join([filename,"jpg"])
                        cv2.imwrite(save_img_folder + filename, right_frame)

                    count = 1
                    for normal, diseased in counting_array:
                        f=open(save_folder + "counting_left.txt", 'a')
                        f.write("// Tree number {0} // normal: {1} diseased: {2}\n".format(count, normal, diseased))
                        f.close()
                        count += 1

                elif trigger in ['16-r']:
                    right_outputs, right_img_info = predictor.inference(right_frame)
                    number_of_normal_right, number_of_diseased_right, result_frame_right = predictor.visual(right_outputs[0], right_img_info, predictor.confthre)

                    right_tree_number = trigger.split('-')[0]

                    counting_array[right_tree_number-1][0] = counting_array[right_tree_number-1][0] + number_of_normal_right
                    counting_array[right_tree_number-1][1] = counting_array[right_tree_number-1][1] + number_of_diseased_right

                    save_img_folder = os.path.joint(save_folder, "%d/" %right_tree_number)
                    os.makedirs(save_img_folder, exist_ok=True)

                    timestr = time.strftime("%Y.%m.%d-%H:%M:%S")
                    filename = "_".join([image_base_name, timestr])
                    filename = ".".join([filename,"jpg"])
                    cv2.imwrite(save_img_folder + filename, right_frame)

                    count = 1
                    for normal, diseased in counting_array:
                        f=open(save_folder + "counting_left.txt", 'a')
                        f.write("// Tree number {0} // normal: {1} diseased: {2}\n".format(count, normal, diseased))
                        f.close()
                        count += 1
                    break

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()


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
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "capture":
        image_capture_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    #video_check = os.system("ls /dev/video*")
    #remove_cache = os.system("rm ~/.cache/gstreamer-1.0/registry.aarch64.bin")
    #env_setting = os.system("export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1")
    #create_capturesession = os.system("sudo service nvargus-daemon restart")
    
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)

