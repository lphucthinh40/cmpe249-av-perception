import math
import numpy as np
import torch

from pointpillar.pcdet.datasets import DatasetTemplate
from pointpillar.pcdet.models import build_network, load_data_to_gpu
from pointpillar.pcdet.config import cfg, cfg_from_yaml_file
from pointpillar.pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

LIDAR_CENTER_OFFSET = 20 
KITTI_IMAGE_SHAPE_NUMPY = np.asarray([375, 1242])

class KittiInferenceDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class PCDetDetector:
    def __init__(self, config_path, model_path, calib_file):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        self.calib_file = calib_file
        
    def initialize(self):
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = KittiInferenceDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False)        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()
        self.calib = calibration_kitti.Calibration(self.calib_file)
        self.classes = cfg.CLASS_NAMES

    def get_template_prediction(self, num_samples):
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict

    def run(self, points, frame):
        num_features = 4 # X,Y,Z,intensity       
        self.points = points.reshape([-1, num_features])

        frame = 0
        timestamps = np.empty((len(self.points),1))
        timestamps[:] = frame

        self.points = np.append(self.points, timestamps, axis=1)
        self.points[:,0] += LIDAR_CENTER_OFFSET

        input_dict = {
            'points': self.points,
            'frame_id': frame,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        
        with torch.no_grad():
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.net.forward(data_dict)

        boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
        types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()

        pred_boxes = np.copy(boxes_lidar)
        pred_dict = self.get_template_prediction(scores.shape[0])
        if scores.shape[0] == 0:
            return pred_dict

        pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, self.calib)
        pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            pred_boxes_camera, self.calib, image_shape=KITTI_IMAGE_SHAPE_NUMPY
        )
        pred_boxes_img = pred_boxes_img.astype(np.int64)

        pred_dict['name'] = np.array(cfg.CLASS_NAMES)[types - 1]
        pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
        pred_dict['bbox'] = pred_boxes_img
        pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
        pred_dict['location'] = pred_boxes_camera[:, 0:3]
        pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
        pred_dict['score'] = scores
        pred_dict['boxes_lidar'] = pred_boxes

        classes = np.array(cfg.CLASS_NAMES)[types - 1]      

        return scores, boxes_lidar, pred_boxes_img, classes, pred_dict