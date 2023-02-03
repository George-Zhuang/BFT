import os
import os.path as osp
from pathlib import Path

from abc import ABC, abstractmethod
from pytracking.run_tracker import make_parser_pytracking, main_pytracking
from utils.soft_link import create_softlink

class SOT_Tracker(ABC):
    @abstractmethod                  
    def track(self, new_frame_path):
        pass

class PyTracker(SOT_Tracker):
    def __init__(self, project_data_path, save_fps=False, mode='predict'):
        self.softlink_path = project_data_path
        self.args = make_parser_pytracking().parse_args()
        self.args.save_fps = save_fps
        self.args.mode = mode
        self.re_init_flag = False

    def init(self, box, init_frame_path, id=None):
        self.init_frame_path = init_frame_path
        # self.args.optional_box = box
        # self.args.optional_id = id
        self.optional_box = box
        self.optional_id = id 

    def re_init(self, box, reinit_frame_path, id=None):
        self.re_init_flag = True
        self.reinit_frame_path = reinit_frame_path
        # self.args.optional_box = box
        # self.args.optional_id = id
        self.optional_box = box
        self.optional_id = id 

    def track(self, new_frame_path):
        self.args.optional_box = self.optional_box
        self.args.optional_id = self.optional_id

        path = Path(new_frame_path)
        folder_name = path.parent.name
        file_name = path.name
        dst_dir_path = osp.join(self.softlink_path, folder_name)
        os.makedirs(dst_dir_path, exist_ok=True)
        dst_path = osp.join(dst_dir_path, file_name)
        self._create_softlink(new_frame_path, dst_path)
        if not self.re_init_flag:
            self._create_softlink(self.init_frame_path)
        else:
            self._create_softlink(self.reinit_frame_path)
            self.re_init_flag = False
        track_box, conf_score = main_pytracking(self.args)

        os.remove(dst_path)
        if not self.re_init_flag:
            self._remove_softlink(self.init_frame_path)
        else:
            self._remove_softlink(self.reinit_frame_path)
            self.re_init_flag = False

        return track_box, conf_score

    def _create_softlink(self, frame_path, dst_path=None):
        if dst_path is None:
            path = Path(frame_path)
            folder_name = path.parent.name
            file_name = path.name
            dst_dir_path = osp.join(self.softlink_path, folder_name)
            os.makedirs(dst_dir_path, exist_ok=True)
            dst_path = osp.join(dst_dir_path, file_name)
        create_softlink(frame_path, dst_path)
        # os.remove(dst_path)

    def _remove_softlink(self, frame_path, dst_path=None):
        path = Path(frame_path)
        folder_name = path.parent.name
        file_name = path.name
        dst_dir_path = osp.join(self.softlink_path, folder_name)
        dst_path = osp.join(dst_dir_path, file_name)
        if osp.exists(dst_path):
            os.remove(dst_path)


        

