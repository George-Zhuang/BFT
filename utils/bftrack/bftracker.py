'''
Modified from ByteTrack with single object tracking (SOT).
Initiate SOT tracker with a single detection box and predict in the next frame.

------------------------------------------------
| Detection | Tracking |        Process         |
------------------------------------------------
|    high   |   high   |       matching         |
------------------------------------------------
|    high   |   low    | re-initialize tracking |
------------------------------------------------
|    low    |   high   |   continue tracking    |
------------------------------------------------
|    low    |   low    |         lost           |
------------------------------------------------
'''


import numpy as np
# from collections import deque
# import os
# import os.path as osp
# import copy
# import torch
# import torch.nn.functional as F

from . import matching
from .basetrack import BaseTrack, TrackState
from .sot_tracker import PyTracker

class STrack(BaseTrack):
    # shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, tracker_name='KeepTrack', data_path=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        if tracker_name == 'KeepTrack':
            self.tracker = PyTracker(data_path)
        self.track_box = None
        self.sot_score = 1.
        self.init_frame_path = None
        self.previous_frame_path = None

    def activate(self, frame_id, frame_path=None):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.init_frame_path = frame_path
        self.tracker.init(self._tlwh, frame_path)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, frame_path=None):
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        # )
        self.track_box, self.sot_score = self.tracker.track(frame_path)
        self.init_frame_path = frame_path
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, frame_path=None):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        if self.previous_frame_path == None:
            self.previous_frame_path = self.init_frame_path
            
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.track_box, self.sot_score = self.tracker.track(frame_path)
        self.track_box = np.asarray(self.track_box[-1], dtype=np.float)
    
        self.tracker.init(new_tlwh, self.previous_frame_path)

        self.previous_frame_path = frame_path
        
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score



    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        # if self.mean is None:
        #     return self._tlwh.copy()
        # ret = self.mean[:4].copy()
        # ret[2] *= ret[3]
        # ret[:2] -= ret[2:] / 2
        # return ret
        if self.track_box is None:
            return self._tlwh.copy()
        # return np.asarray(self.track_box[-1], dtype=np.float)
        # elif len(self.track_box) == 4:
        #     return np.asarray(self.track_box, dtype=np.float)
        else:
            return self.track_box.copy()

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)



# class BFTracker(object):
#     def __init__(self, args, frame_rate=30):
#         self.tracked_stracks = []  # type: list[STrack]
#         self.lost_stracks = []  # type: list[STrack]
#         self.removed_stracks = []  # type: list[STrack]

#         self.frame_id = 0
#         self.args = args
#         #self.det_thresh = args.track_thresh
#         self.det_thresh = args.track_thresh + 0.1
#         self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
#         self.max_time_lost = self.buffer_size
#         # self.kalman_filter = KalmanFilter()
        
#         self.tracker_name = args.sot_tracker_name
#         self.data_path = args.track_data_path
#         self.tracker_thresh = args.sot_thresh

#     def update(self, output_results, img_info, img_size):
#         self.frame_id += 1
#         activated_starcks = []
#         refind_stracks = []
#         lost_stracks = []
#         removed_stracks = []

#         if output_results.shape[1] == 5:
#             scores = output_results[:, 4]
#             bboxes = output_results[:, :4]
#         else:
#             output_results = output_results.cpu().numpy()
#             scores = output_results[:, 4] * output_results[:, 5]
#             bboxes = output_results[:, :4]  # x1y1x2y2
#         img_h, img_w = img_info[0], img_info[1]
#         scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
#         bboxes /= scale

#         remain_inds = scores > self.args.track_thresh
#         inds_low = scores > 0.1
#         inds_high = scores < self.args.track_thresh

#         inds_second = np.logical_and(inds_low, inds_high)
#         dets_second = bboxes[inds_second]
#         dets = bboxes[remain_inds]
#         scores_keep = scores[remain_inds]
#         scores_second = scores[inds_second]

#         if len(dets) > 0:
#             '''Detections'''
#             detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.tracker_name, self.data_path) for
#                           (tlbr, s) in zip(dets, scores_keep)]
#         else:
#             detections = []

#         ''' Add newly detected tracklets to tracked_stracks'''
#         unconfirmed = []
#         tracked_stracks = []  # type: list[STrack]
#         for track in self.tracked_stracks:
#             if not track.is_activated:
#                 unconfirmed.append(track)
#             else:
#                 tracked_stracks.append(track)

#         ''' Step 2: First association, with high score detection boxes'''
#         strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
#         # Predict the current location with KF
#         # STrack.multi_predict(strack_pool)

#         dists = matching.iou_distance(strack_pool, detections)
#         if not self.args.mot20:
#             dists = matching.fuse_score(dists, detections)
#         matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

#         for itracked, idet in matches:
#             track = strack_pool[itracked]
#             det = detections[idet]
#             if track.state == TrackState.Tracked:

#                 # track.tracker.init()
                
#                 track.update(detections[idet], self.frame_id, frame_path=img_info[2])
#                 activated_starcks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False, frame_path=img_info[2])
#                 refind_stracks.append(track)

#             # |    high   |   low    | re-initialize tracking |
#             if track.sot_score[-1] < self.tracker_thresh:
#                 track.tracker.re_init(det.tlwh, img_info[2])

#         ''' Step 3: Second association, with low score detection boxes'''
#         # association the untrack to the low score detections
#         if len(dets_second) > 0:
#             '''Detections'''
#             detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.tracker_name, self.data_path) for
#                           (tlbr, s) in zip(dets_second, scores_second)]
#         else:
#             detections_second = []
#         r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
#         dists = matching.iou_distance(r_tracked_stracks, detections_second)
#         matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
#         for itracked, idet in matches:
#             track = r_tracked_stracks[itracked]
#             det = detections_second[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id, frame_path=img_info[2])
#                 activated_starcks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False, frame_path=img_info[2])
#                 refind_stracks.append(track)

#         # for it in u_track:
#         #     track = r_tracked_stracks[it]
#         #     # |    low    |   low    |         lost           |
#         #     if track.sot_score[-1] < self.tracker_thresh:
#         #         if not track.state == TrackState.Lost:
#         #             track.mark_lost()
#         #             lost_stracks.append(track)
#         #     # |    low    |   high   |   continue tracking    |
#         #     else:
#         #         # pseudo detection label
#         #         pseudo_det = track
#         #         if track.state == TrackState.Tracked:
#         #             track.update(pseudo_det, self.frame_id, frame_path=img_info[2])
#         #             activated_starcks.append(track)
#         #         else:
#         #             track.re_activate(pseudo_det, self.frame_id, new_id=False, frame_path=img_info[2])
#         #             refind_stracks.append(track)
#         for it in u_track:
#             track = r_tracked_stracks[it]
#             if not track.state == TrackState.Lost:
#                 track.mark_lost()
#                 lost_stracks.append(track)


#         '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
#         detections = [detections[i] for i in u_detection]
#         dists = matching.iou_distance(unconfirmed, detections)
#         if not self.args.mot20:
#             dists = matching.fuse_score(dists, detections)
#         matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
#         for itracked, idet in matches:
#             unconfirmed[itracked].update(detections[idet], self.frame_id, frame_path=img_info[2])
#             activated_starcks.append(unconfirmed[itracked])
#         for it in u_unconfirmed:
#             track = unconfirmed[it]
#             track.mark_removed()
#             removed_stracks.append(track)

#         """ Step 4: Init new stracks"""
#         for inew in u_detection:
#             track = detections[inew]
#             if track.score < self.det_thresh:
#                 continue
#             # track.activate(self.kalman_filter, self.frame_id)
#             # track.activate(self.tracker, self.frame_id, frame_path=img_info[2])
#             track.activate(self.frame_id, frame_path=img_info[2])
#             activated_starcks.append(track)
#         """ Step 5: Update state"""
#         for track in self.lost_stracks:
#             if self.frame_id - track.end_frame > self.max_time_lost:
#                 track.mark_removed()
#                 removed_stracks.append(track)

#         # print('Ramained match {} s'.format(t4-t3))

#         self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
#         self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
#         self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
#         self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
#         self.lost_stracks.extend(lost_stracks)
#         self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
#         self.removed_stracks.extend(removed_stracks)
#         self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
#         # get scores of lost tracks
#         output_stracks = [track for track in self.tracked_stracks if track.is_activated]

#         return output_stracks


class BFTracker(object):
    def __init__(self, track_thresh, match_thresh, track_buffer, mot20, \
        sot_tracker_name, track_data_path, sot_thresh, frame_rate=30):
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        #self.det_thresh = args.track_thresh
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        self.mot20 = mot20

        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        # self.kalman_filter = KalmanFilter()
        
        self.tracker_name = sot_tracker_name
        self.data_path = track_data_path
        self.tracker_thresh = sot_thresh

    def update(self, output_results, img_info, img_size, output_mode='yolox'):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        
        if output_mode == 'yolox':
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.tracker_name, self.data_path) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        # STrack.multi_predict(strack_pool)

        dists = matching.iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:

                # track.tracker.init()
                
                track.update(detections[idet], self.frame_id, frame_path=img_info[2])
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, frame_path=img_info[2])
                refind_stracks.append(track)

            # |    high   |   low    | re-initialize tracking |
            if track.sot_score[-1] < self.tracker_thresh:
                track.tracker.re_init(det.tlwh, img_info[2])

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.tracker_name, self.data_path) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, frame_path=img_info[2])
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, frame_path=img_info[2])
                refind_stracks.append(track)

        # for it in u_track:
        #     track = r_tracked_stracks[it]
        #     # |    low    |   low    |         lost           |
        #     if track.sot_score[-1] < self.tracker_thresh:
        #         if not track.state == TrackState.Lost:
        #             track.mark_lost()
        #             lost_stracks.append(track)
        #     # |    low    |   high   |   continue tracking    |
        #     else:
        #         # pseudo detection label
        #         pseudo_det = track
        #         if track.state == TrackState.Tracked:
        #             track.update(pseudo_det, self.frame_id, frame_path=img_info[2])
        #             activated_starcks.append(track)
        #         else:
        #             track.re_activate(pseudo_det, self.frame_id, new_id=False, frame_path=img_info[2])
        #             refind_stracks.append(track)
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, frame_path=img_info[2])
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # track.activate(self.kalman_filter, self.frame_id)
            # track.activate(self.tracker, self.frame_id, frame_path=img_info[2])
            track.activate(self.frame_id, frame_path=img_info[2])
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
