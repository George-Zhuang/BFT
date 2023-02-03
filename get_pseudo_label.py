'''
Flock-aligned pseudo labeling.
For detailed instruction, please refer to readme.md
'''

import importlib
import argparse
import glob
import os
import os.path as osp
import cv2

from loguru import logger
from tabulate import tabulate
from utils import get_image_list, plot_mot, predict

from pytracking.evaluation import environment
from ltr.external import clone

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
environment.set_project_root(PROJECT_ROOT)
environment.create_default_local_file()
clone.clone_PreciseRoIpooling()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-exp", "--exp_name", type=str, default='sahi', help="experiment name")
    parser.add_argument(
        "-i", "--input_path", type=str, default='/media/zhuang/HKU_Data/flock/BFT/data/img_seq/', help="image sequence path")
    parser.add_argument(
        "-n", "--seq_name", type=str, default='test', help="name of sequence, 'all' for all")
    parser.add_argument(
        '-o', "--output_path", type=str, default='./output', help="path to save results")

    return parser.parse_args() 

def create_logger():
    os.makedirs('./log', exist_ok=True)
    log_path = "./log/Pseudo_label.log"
    logger.add(log_path)
    logger.info(" Log created, please refer to {} ".format(log_path).center(100, '-'))
    return logger

def make_args(logger):
    args = parse_args()
    exp_module = ".".join(['exps', args.exp_name])
    exp_object = importlib.import_module(exp_module).Exp()
    
    print_args = [['Key', 'Value']] + [[k, v] for k, v in vars(args).items()] + [[k, v] for k, v in exp_object.__dict__.items()]
    print_args = tabulate(print_args, headers='firstrow', tablefmt='grid')
    logger.info('Arg Table: \n'+print_args)
    return args, exp_object

def get_img_seq(args, logger):
    # get image sequences
    if args.seq_name == 'all':
        seq_list = sorted(glob.glob(args.input_path))
    else: # single sequence mode
        seq_list = [osp.join(args.input_path, args.seq_name)]
    logger.info(' Here we get {} image sequences '.format(len(seq_list)).center(100, '-'))
    return seq_list

if __name__ == "__main__":
    # create a logger object
    logger = create_logger()
    # get the arguments and experiment object
    args, exp_object = make_args(logger)
    # get a list of image sequences
    seq_list = get_img_seq(args, logger)
    # get the tracker from the experiment object
    tracker = exp_object.get_tracker(logger)
    # initialize the results list
    results = []

    for seq in seq_list:
        logger.info(' Now processing sequence: {}'.format(seq).center(100, '-'))
        input_path = osp.join(args.input_path, seq)
        # get image list
        if osp.isdir(input_path):
            files = get_image_list(input_path)
        else:
            files = [input_path]
        files.sort()

        for frame_id, img_path in enumerate(files, 1):
            outputs, img_info = predict(model_type=exp_object.model_type,
                                        model_path=exp_object.model_path,
                                        model_config_path=exp_object.model_config_path,
                                        model_device=exp_object.model_device,
                                        model_confidence_threshold=exp_object.model_confidence_threshold,
                                        source=img_path,
                                        slice_height=exp_object.slice_height,
                                        slice_width=exp_object.slice_width,
                                        overlap_height_ratio=exp_object.overlap_height_ratio,
                                        overlap_width_ratio=exp_object.overlap_width_ratio,
                                        project=args.output_path,
                                        novisual=True,
                                        verbose=0,
                                        no_sliced_prediction=True
                                    )
            img_info['img_path'] = img_path
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], 
                                                [img_info['height'], img_info['width'], img_info['img_path']],
                                                exp_object.test_size, 
                                                output_mode=args.exp_name)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > exp_object.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > exp_object.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                online_im = plot_mot(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=0
                )
            else:
                online_im = img_info['raw_img']

            if exp_object.save_result:
                save_folder = osp.join(args.output_path, osp.basename(seq))
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        if args.output_path is not None:
            res_file = osp.join(args.output_path, f"{osp.basename(seq)}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"Save results to {res_file}".center(100, '-'))


    
