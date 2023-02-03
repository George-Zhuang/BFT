import torch

from exps.base_exp import BaseExp
from utils import BFTracker
from yolox.tools.demo import get_exp, Predictor

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # ----------------- det model config ----------------- #
        self.det_exp_file = './yolox/exps/default/yolox_x.py'
        self.det_name = 'yolox-x'
        self.det_model = './yolox/yolox_x.pth'
        self.trt_file = None # without trt file
        self.decoder = None 
        self.device = "gpu"
        self.fp16 = False

        self.test_size = (640, 640)
        # ---------------- track model config ---------------- #
        self.track_thresh = 0.5
        self.match_thresh = 0.8 # 0.6 0.8
        self.aspect_ratio_thresh = 1.6
        self.track_buffer = 30
        self.mot20 = None
        self.fps = 30   
        self.min_box_area = 5
        self.save_result = True
        self.sot_tracker_name = 'KeepTrack'
        self.sot_thresh = 0.55
        self.track_data_path = './pytracking/input/BFT'

    def get_det_model(self, logger):
        exp = get_exp(self.det_exp_file, self.det_name)
        det_model = exp.get_model()
        if self.device == "gpu":
            det_model.cuda()
            if self.fp16:
                det_model.half()  # to FP16
        det_model.eval()

        ckpt_file = self.det_model
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        det_model.load_state_dict(ckpt["model"])
        logger.info(' And we get detection model '.center(100, '-'))

        predictor = Predictor(det_model, exp, self.trt_file, self.decoder, device=self.device, fp16=self.fp16)

        return predictor
    
    def get_tracker(self, logger):
        logger.info(' Loading tracker: {} '.format(self.sot_tracker_name).center(100, '-'))
        return BFTracker(self.track_thresh, self.match_thresh, self.track_buffer, self.mot20, \
            self.sot_tracker_name, self.track_data_path, self.sot_thresh)