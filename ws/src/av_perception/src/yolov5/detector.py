import torch
import cv2
import os
import sys
import numpy as np


from yolov5.models.experimental import attempt_load  # scoped to avoid circular import
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import Annotator, colors

from imutils.video import FileVideoStream
from imutils.video import FPS
import time

WEIGHT_PATH = '/home/thinh/Project/cmpe249-av-perception/ws/src/av_perception/src/yolov5/weights/best.pt'
IMAGE_SIZE = [1280, 1280]
CONFIDENCE_THRESH = 0.25  # confidence threshold
NMS_IOU_THRESH = 0.45
MAX_DETECTION_PER_IMAGE = 500

class Yolov5Detector(object):
    def __init__(self):
        # Load model
        torch.cuda.set_per_process_memory_fraction(0.25)
        device = select_device()
        model = attempt_load(WEIGHT_PATH, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names
        imageSize = check_img_size(IMAGE_SIZE, s=stride)  # check image size
        model.float()
        if device.type != 'cpu':
            model(torch.zeros(1, 3, *imageSize).to(device).type_as(next(model.model.parameters())))  # warmup
        self.__dict__.update(locals()) 

    # def detectImage(self, path):
    #     img = cv2.imread(path)  # BGR
    #     assert img is not None, f'Image Not Found {path}'
    #     img_out = self.detect(img)
    #     cv2.imshow('result', img_out)
    #     cv2.waitKey(0)  # 1 millisecond
    #     # Save results (image with detections)
    #     ext = os.path.splitext(path)
    #     cv2.imwrite(f'{ext[0]}_result{ext[1]}', img_out)            

    # def detectVideo(self, path):
    #     fvs = FileVideoStream(path).start()                 
    #     time.sleep(1.0)
    #     # start the FPS timer
    #     while fvs.more():	
    #         frame = fvs.read()
    #         if frame is not None:
    #             img_out = self.detect(frame)
    #             cv2.imshow("Frame", img_out)
    #             cv2.waitKey(20)        
    
    @torch.no_grad()
    def detect(self, frame):
        # Padded resize
        img = letterbox(frame, IMAGE_SIZE, stride=self.stride)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # Run inference        
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, conf_thres=CONFIDENCE_THRESH, iou_thres=NMS_IOU_THRESH, max_det=MAX_DETECTION_PER_IMAGE)
        annotator = Annotator(frame)
        pred_dict = {'bboxes': [], 'types': [], 'scores': []}
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(frame)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                # Convert to numpy
                det = det.detach().cpu().numpy()
                # Process results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    pred_dict['bboxes'].append(np.array(xyxy, dtype=np.int64)),
                    pred_dict['types'].append(self.names[c]),
                    pred_dict['scores'].append(conf),

        # Stream results
        im_out = annotator.result()
        
        return im_out, pred_dict