import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from abc import ABC, abstractmethod
from pytracking.run_tracker import make_parser_pytracking, main_pytracking

class SOT_Tracker(ABC):
    @abstractmethod                  
    def track(self, new_frame_path):
        pass

class PyTracker(SOT_Tracker):
    def __init__(self, project_data_path, save_fps=False, mode='predict'):
        self.args = make_parser_pytracking().parse_args()
        self.args.save_fps = save_fps
        self.args.mode = mode
        self.init_frame_path = None
        self.project_data_path = project_data_path

    def init(self, box, init_frame_path, id=None):
        self.init_frame_path = init_frame_path
        self.args.optional_box = box
        self.args.optional_id = id
        self.init_folder_name, self.init_prefix, self.init_suffix = self._get_file_info(init_frame_path)

    def track(self, new_frame_path):
        folder_name, prefix, suffix = self._get_file_info(new_frame_path)
        new_temp_dir = os.path.join(self.project_data_path, folder_name)
        os.makedirs(new_temp_dir, exist_ok=True)
        with NamedTemporaryFile(dir=new_temp_dir, prefix=prefix, suffix=suffix, delete=False) as new_temp_file, \
             NamedTemporaryFile(dir=new_temp_dir, prefix=self.init_prefix, suffix=self.init_suffix, delete=False) as init_temp_file:

            shutil.copy(new_frame_path, new_temp_file.name)
            shutil.copy(self.init_frame_path, init_temp_file.name)
            
            track_box, conf_score = main_pytracking(self.args)
            
            os.remove(new_temp_file.name) # remove the temp file after use
            os.remove(init_temp_file.name) # remove the temp file after use
            
        return track_box, conf_score

    def _get_file_info(self, path):
        path = Path(path)
        folder_name = path.parent.name
        prefix, suffix = path.name.split('.')
        return folder_name, prefix, suffix
