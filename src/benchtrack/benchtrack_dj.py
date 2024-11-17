"""
Inspired from elements
"""
import os
import glob
import time
import sys
import logging
from dataclasses import dataclass

import datajoint as dj
import cv2
import pandas as pd
import numpy as np
import debugpy

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

schema = dj.Schema("benchtrack")

@schema
class Project(dj.Manual):
    definition = """
    project_id: varchar(32)
    ---
    project_name: varchar(255)
    project_description=''  : varchar(1024)
    """

@schema
class Subject(dj.Manual):
    definition = """
    subject_id: varchar(32)
    ---
    subject_name: varchar(255)
    subject_description=''  : varchar(1024)
    """

@schema
class Session(dj.Manual):
    definition = """
    -> Project
    session_id: varchar(32)
    ---
    session_date=null: date
    session_description=''  : varchar(1024)
    """
    class Subject(dj.Part):
        definition = """
        -> master
        -> Subject
        """

    class Segment(dj.Part):
        definition = """
        -> master
        segment_name: varchar(64)
        ---
        segment_description=''  : varchar(1024)
        """

    class AnnotatedKeypoint(dj.Part):
        """
        Attributes:
        - frame_idx: list of frame indices
        - pose: ndarray of shape (num_frames, keypoint dimension)
        - likelihood: ndarray of shape (num_frames, num_parts)
        """
        definition = """
        -> master
        -> Session.Segment
        -> Session.Subject
        ---
        frame_idx: longblob
        pose: longblob
        likelihood=null: longblob
        """

@schema
class Video(dj.Manual):
    definition = """
    -> Session
    video_id: varchar(64)
    ---
    video: longblob
    """

@schema
class VideoInfo(dj.Manual):
    definition = """
    -> Video
    ---
    description=''  : varchar(1024)
    fps             : float
    height          : int
    width           : int
    num_frames      : int
    """


# just an example of one way to pass data to cotracker
class TrackingWorker:

    @dataclass
    class TrackingWorkerData:
        tracker: str
        video: np.ndarray
        keypoints: np.ndarray # (num_keypoint, 3) first col is frame numbe in `video` and the second and third are x, y
        keypoint_features: dict
        keypoint_order_idx: np.ndarray
        keypoint_range: tuple[int, int]
        backward_tracking: bool

    def __init__(self, device='cuda'):
        self.device = device
        self.model = None

    def track(self, cfg: TrackingWorkerData):
        import torch
        if self.model is None:
            self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.device)
        if len(cfg.video) <= self.model.step:
            self.model.step = int(len(cfg.video)/2)
        def _process_step(window_frames, is_first_step, queries):
            video_chunk = (
                torch.tensor(np.stack(window_frames[-self.model.step * 2:]), device=self.device)
                .float()
                .permute(0, 3, 1, 2)[None]
            )  # (1, T, 3, H, W)
            return self.model(video_chunk, is_first_step=is_first_step, queries=queries[None], add_support_grid=True)
        # video is originally of shape (num_frames, height, width, channels)
        video = np.array(cfg.video)
        window_frames = []

        # We need to swap x, y so that it matches what cotracker expects 
        # cfg.keypoints[:, [1, 2]] = cfg.keypoints[:, [2, 1]]

        queries = torch.from_numpy(cfg.keypoints).to(self.device).float()
        
        # Iterating over video frames, processing one window at a time:
        is_first_step = True
        for i, frame in enumerate(video):
            if i % self.model.step == 0 and i != 0:
                pred_tracks, _pred_visibility = _process_step(window_frames, is_first_step, queries=queries)
                is_first_step = False
            window_frames.append(frame)

        # Processing final frames in case video length is not a multiple of model.step
        # TODO: Use visibility
        pred_tracks, _pred_visibility = _process_step(
            window_frames[-(i % self.model.step) - self.model.step - 1:],
            is_first_step,
            queries=queries,
        )

        tracks = pred_tracks.squeeze().cpu().numpy()
        if cfg.keypoints.shape[0] > 1:
            tracks = tracks[:, :cfg.keypoints.shape[0], :] # drop the support grid (necessary only for cotracker version < 3)
        tracks = tracks.reshape(-1, 2)
        if cfg.backward_tracking:
            tracks = tracks[::-1]
        frame_ids = np.repeat(np.arange(cfg.keypoint_range[0], cfg.keypoint_range[1]), cfg.keypoints.shape[0])
        tracks = np.column_stack((frame_ids, tracks))
        cfg.keypoint_features = pd.concat([cfg.keypoint_features] * len(np.unique(tracks[:, 0])), ignore_index=True)
        cfg.keypoints = tracks
        return cfg

@schema
class TrackingBenchmark(dj.Computed):
    definition = """
    -> Session
    ---
    keypoint_order: longblob
    tracked_keypoints: longblob
    tracking_events: longblob
    """

    tracker = TrackingWorker()

    def make(self, key):
        subjects = (Session.Subject & key).fetch(as_dict=True)
        video = (Video & key).fetch1('video')
        annotated_keypoints_raw = (Session.AnnotatedKeypoint & key).fetch(as_dict=True)
        segments = (Session.Segment & key).fetch(as_dict=True)

        window_size = 20
        rmse_threshold = 3

        num_frames = len(video)
        
        # Create a dictionary for quick lookup of keypoints
        keypoint_pose = {(kp['subject_id'], kp['segment_name']): kp['pose'] for kp in annotated_keypoints_raw}
        keypoint_order = keypoint_pose.keys()

        annotated_keypoints = []
        annotated_keypoints_info = []

        tracking_events = []

        for frame_idx in range(num_frames):
            for (segment_name, subject_id), pose in keypoint_pose.items():
                annotated_keypoints.append([frame_idx, pose[frame_idx][0], pose[frame_idx][1]])
                annotated_keypoints_info.append([frame_idx, segment_name, subject_id])

        annotated_keypoints = np.array(annotated_keypoints) # (num_frames * num_keypoints, 3) where the columns are frame_idx, x, y
        annotated_keypoints_info = pd.DataFrame(annotated_keypoints_info, columns=['frame_idx', 'segment_name', 'subject_id']) # (num_frames * num_keypoints, 3) where the columns are frame_idx, segment_name, subject_id

        
        tracked_keypoints = annotated_keypoints.copy()
        tracked_keypoints[:, 1:] = np.nan
        # set the rows corresponding to the first frame

        for frame_idx in range(num_frames-8):
            # compare current_keypoints with annotated_keypoints for frame_idx and 
            # return the indices of the keypoints in the current frame (in terms of keypoint_order) that have a rmse higher than threshold or
            # are in annotated_keypoints but are not in current_keypoints
            rmse = np.sqrt(np.mean((tracked_keypoints[tracked_keypoints[:, 0] == frame_idx, 1:] - annotated_keypoints[annotated_keypoints[:, 0] == frame_idx, 1:])**2, axis=1))
            keypoints_to_track = np.full_like(tracked_keypoints[tracked_keypoints[:, 0] == frame_idx], np.nan)
            keypoints_to_track_idx = np.where((rmse > rmse_threshold) | np.isnan(rmse))[0]
            # these are keypoints that are nan in annotated_keypoints
            keypoinys_not_to_track_idx = np.where(np.any(np.isnan(annotated_keypoints[annotated_keypoints[:, 0] == frame_idx]), axis=1))[0]
            keypoints_to_track_idx = np.setdiff1d(keypoints_to_track_idx, keypoinys_not_to_track_idx)
            keypoints_to_track[keypoints_to_track_idx] = annotated_keypoints[annotated_keypoints[:, 0] == frame_idx][keypoints_to_track_idx]
            keypoints_to_track_info = annotated_keypoints_info[annotated_keypoints_info['frame_idx'] == frame_idx]

            # drop rows with nan from keypoints_to_track and keypoints_to_track_info
            keypoints_to_track_info = keypoints_to_track_info.iloc[~np.isnan(keypoints_to_track).any(axis=1)]
            keypoints_to_track = keypoints_to_track[~np.isnan(keypoints_to_track).any(axis=1)]

            if len(keypoints_to_track) == 0:
                continue

            tracking_events.append((frame_idx, keypoints_to_track_idx))
            
            # set frame column to 0
            keypoints_to_track[:, 0] = 0
            if frame_idx + window_size >= num_frames:
                window_size = num_frames - frame_idx - 1
            # track from the current frame to the next window_size frames
            # with keypoints_to_track
            print(f"Tracking {len(keypoints_to_track)} keypoints from frame {frame_idx} to frame {frame_idx + window_size}")
            tracking_data = TrackingBenchmark.tracker.track(TrackingWorker.TrackingWorkerData(
                tracker="cotracker",
                video=video[frame_idx:frame_idx + window_size],
                keypoints=keypoints_to_track,
                keypoint_features=annotated_keypoints_info[annotated_keypoints_info['frame_idx'] == frame_idx],
                keypoint_order_idx=keypoints_to_track_idx,
                keypoint_range=(frame_idx, frame_idx + window_size),
                backward_tracking=False
            ))

            # update tracked_keypoints with the tracked keypoints from tracking_data
            for f_idx in range(tracking_data.keypoint_range[0], tracking_data.keypoint_range[1]):
                # This is voodoo; but the question is, is this too much?
                tracked_keypoints[(tracked_keypoints[:, 0] == f_idx).nonzero()[0][tracking_data.keypoint_order_idx]] = tracking_data.keypoints[tracking_data.keypoints[:, 0] == f_idx]

        # create an mp4 file with the tracked keypoints which is of shape (num_frames, 3) where the columns are frame_idx, x, y
        # and the order of the keypoints in each frame is the same as in keypoint_order which corresponds to keypoint_info
        out_video_path = "tracked_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = video[0].shape[:2]
        out = cv2.VideoWriter(out_video_path, fourcc, 10.0, (width, height))

        for frame_idx, frame in enumerate(video):
            frame = frame.copy()  # Make the frame writable
            for kp in tracked_keypoints[tracked_keypoints[:, 0] == frame_idx]:
                if not np.isnan(kp).any():
                    cv2.circle(frame, (int(kp[1]), int(kp[2])), 2, (0, 255, 0), -1)
            out.write(frame)

        out.release()


        key['keypoint_order'] = keypoint_order
        key['tracked_keypoints'] = tracked_keypoints
        key['tracking_events'] = tracking_events

        # self.insert1(key)

    def render_overlay(self, key):
        pass



            

if __name__ == "__main__":
    TrackingBenchmark.populate()
