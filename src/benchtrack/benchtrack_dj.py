"""
Inspired from elements
"""
import os
import pickle
import glob
import time
import sys
import logging
from dataclasses import dataclass

import datajoint as dj
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import debugpy

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

def config_plots():
    # Global configuration for matplotlib
    plt.rcParams.update({
        'font.size': 16,  # Font size for text
        'font.family': 'sans-serif',  # Font family
        'axes.titlesize': 18,  # Font size for axes titles
        'axes.labelsize': 16,  # Font size for axes labels
        'xtick.labelsize': 14,  # Font size for x-tick labels
        'ytick.labelsize': 14,  # Font size for y-tick labels
        'legend.fontsize': 18,  # Font size for legend
        'lines.linewidth': 3,  # Line width
        'figure.dpi': 100,  # Figure DPI
        'lines.linewidth': 2,  # Line width
        'lines.markersize': 8,  # Marker size
        'axes.titleweight': 'bold'
    })

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
        segment_color: varchar(32)
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
        if cfg.backward_tracking:
            cfg.video = cfg.video[::-1]
        video = np.array(cfg.video)
        window_frames = []

        queries = torch.from_numpy(cfg.keypoints).to(self.device).float()
        
        # Iterating over video frames, processing one window at a time:
        is_first_step = True
        for i, frame in enumerate(video):
            if i % self.model.step == 0 and i != 0:
                pred_tracks, _pred_visibility = _process_step(window_frames, is_first_step, queries=queries)
                is_first_step = False
            window_frames.append(frame)

        # Processing final frames in case video length is not a multiple of model.step
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
        # The order of keypoints in each frame is now reversed. We go through each frame and reverse the order of keypoints
        if cfg.backward_tracking:
            for frame_idx in range(cfg.keypoint_range[0], cfg.keypoint_range[1]):
                tracks[(tracks[:, 0] == frame_idx).nonzero()[0]] = tracks[tracks[:, 0] == frame_idx][::-1]
        cfg.keypoint_features = pd.concat([cfg.keypoint_features] * len(np.unique(tracks[:, 0])), ignore_index=True)
        cfg.keypoints = tracks
        return cfg

@schema
class TrackingBenchmarkSettingsLookup(dj.Lookup):
    definition = """
    setting_id: int
    ---
    window_size: int
    rmse_threshold: float
    """

    contents = [
        {"setting_id": "1", "window_size": "20", "rmse_threshold": "3"},
        {"setting_id": "2", "window_size": "10", "rmse_threshold": "5"},
    ]

@schema
class TrackingBenchmark(dj.Computed):
    definition = """
    -> Session
    -> TrackingBenchmarkSettingsLookup
    """

    class TrackedKeypoint(dj.Part):
        definition = """
        -> master
        -> Session.Subject
        -> Session.Segment
        ---
        pose: longblob
        rmse: longblob
        forward_pose: longblob
        forward_rmse: longblob
        start_frame_idx: int
        bothway_pose: longblob
        bothway_rmse: longblob
        middle_frame_idx: int
        tracking_events: longblob
        """

    tracker = TrackingWorker()

    def make(self, key):
        video = (Video & key).fetch1('video')
        annotated_keypoints_raw = (Session.AnnotatedKeypoint & key).fetch(as_dict=True)

        window_size, rmse_threshold = (TrackingBenchmarkSettingsLookup & key).fetch1('window_size', 'rmse_threshold')

        num_frames = len(video)
        # fetch the color from Session.Segments using the name in AnnotatedKeypoint
        kp_colors = [(Session.Segment & kp_key).fetch1('segment_color') for kp_key in annotated_keypoints_raw]
        # Create a dictionary for quick lookup of keypoints
        keypoint_pose = {(kp['subject_id'], kp['segment_name'], kp_color): kp['pose'] for kp, kp_color in zip(annotated_keypoints_raw, kp_colors)}
        keypoint_order = keypoint_pose.keys()

        annotated_keypoints = []
        annotated_keypoints_info = []


        for frame_idx in range(num_frames):
            for (segment_name, subject_id, segment_color), pose in keypoint_pose.items():
                annotated_keypoints.append([frame_idx, pose[frame_idx][0], pose[frame_idx][1]])
                annotated_keypoints_info.append([frame_idx, segment_name, subject_id, segment_color])

        annotated_keypoints = np.array(annotated_keypoints) # (num_frames * num_keypoints, 3) where the columns are frame_idx, x, y
        annotated_keypoints_info = pd.DataFrame(annotated_keypoints_info, columns=['frame_idx', 'segment_name', 'subject_id', 'segment_color']) # (num_frames * num_keypoints, 3) where the columns are frame_idx, segment_name, subject_id

        tracking_events = np.zeros((num_frames, len(keypoint_order)), dtype=bool)
        
        tracked_keypoints = annotated_keypoints.copy()
        tracked_keypoints[:, 1:] = np.nan
        # set the rows corresponding to the first frame

        end_slack = 8

        for frame_idx in range(num_frames-end_slack):
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

            # TODO: if we are in the last iteration, retrack to end all the keypoints in the (last_frame - end_lack - 1) frame that are nan in the last end_slack frames.

            # drop rows with nan from keypoints_to_track and keypoints_to_track_info
            keypoints_to_track_info = keypoints_to_track_info.iloc[~np.isnan(keypoints_to_track).any(axis=1)]
            keypoints_to_track = keypoints_to_track[~np.isnan(keypoints_to_track).any(axis=1)]

            if len(keypoints_to_track) == 0:
                continue

            tracking_events[frame_idx, keypoints_to_track_idx] = True
            
            # set frame column to 0
            keypoints_to_track[:, 0] = 0
            if frame_idx + window_size > num_frames:
                window_size = num_frames - frame_idx
            # track from the current frame to the next window_size frames
            # with keypoints_to_track
            logging.debug(f"Tracking {len(keypoints_to_track)} keypoints from frame {frame_idx} to frame {frame_idx + window_size}")
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

        # Reshape them back to (num_frames, num_keypoints, 2)
        num_keypoints = len(keypoint_order)
        rmse_values = np.full((num_frames, num_keypoints), np.nan)
        tracked_keypoints_reshaped = np.zeros((num_frames, num_keypoints, 2))
        for frame_idx in range(num_frames):
            for kp_idx in range(num_keypoints):
                tracked_kp = tracked_keypoints[(tracked_keypoints[:, 0] == frame_idx)][kp_idx][1:]
                annotated_kp = annotated_keypoints[(annotated_keypoints[:, 0] == frame_idx)][kp_idx][1:]
                tracked_keypoints_reshaped[frame_idx, kp_idx] = tracked_kp
                rmse = np.sqrt(np.mean((tracked_kp - annotated_kp) ** 2))
                rmse_values[frame_idx, kp_idx] = rmse


        ### Track from start to end
        start_frame_idx = 0
        forward_rmse = np.full((num_frames, num_keypoints), np.nan)
        # track from the current frame to the last frames
        logging.debug(f"Tracking keypoints from frame {start_frame_idx} to frame {len(video)}.")
        # find the first frame where all keypoints are annotated
        while True:
            if np.any(np.isnan(annotated_keypoints[annotated_keypoints[:, 0] == start_frame_idx, 1:])):
                start_frame_idx += 1
            else:
                break
        keypoints_to_track = annotated_keypoints[annotated_keypoints[:, 0] == start_frame_idx]
        keypoints_to_track[:, 0] = 0
        tracking_data = TrackingBenchmark.tracker.track(TrackingWorker.TrackingWorkerData(
            tracker="cotracker",
            video=video[start_frame_idx:len(video)],
            keypoints=keypoints_to_track,
            keypoint_features=annotated_keypoints_info[annotated_keypoints_info['frame_idx'] == start_frame_idx],
            keypoint_order_idx=range(num_keypoints),
            keypoint_range=(start_frame_idx, len(video)),
            backward_tracking=False
        ))
        # calculate forward_rmse from tracking_data
        forward_tracked_keypoints_reshaped = np.zeros((num_frames, num_keypoints, 2))
        for f_idx in range(start_frame_idx, len(video)):
            for kp_idx in range(num_keypoints):
                tracked_kp = tracking_data.keypoints[tracking_data.keypoints[:, 0] == f_idx][kp_idx][1:]
                annotated_kp = annotated_keypoints[annotated_keypoints[:, 0] == f_idx][kp_idx][1:]
                forward_tracked_keypoints_reshaped[f_idx, kp_idx] = tracked_kp
                rmse = np.sqrt(np.mean((tracked_kp - annotated_kp) ** 2))
                forward_rmse[f_idx, kp_idx] = rmse

        ### Track from middle to start and then middle to end
        bothway_rmse = np.full((num_frames, num_keypoints), np.nan)
        middle_frame_idx = len(video) // 2
        # find the closest frame to the center that has all the keypoints annotated
        while True:
            if np.any(np.isnan(annotated_keypoints[annotated_keypoints[:, 0] == middle_frame_idx, 1:])):
                middle_frame_idx += 1
            else:
                break
        # track from the current frame to the last frames
        logging.debug(f"Tracking keypoints from frame {middle_frame_idx} to frame {len(video)}")
        keypoints_to_track = annotated_keypoints[annotated_keypoints[:, 0] == middle_frame_idx]
        keypoints_to_track[:, 0] = 0
        tracking_data = TrackingBenchmark.tracker.track(TrackingWorker.TrackingWorkerData(
            tracker="cotracker",
            video=video[middle_frame_idx:len(video)],
            keypoints=keypoints_to_track,
            keypoint_features=annotated_keypoints_info[annotated_keypoints_info['frame_idx'] == middle_frame_idx],
            keypoint_order_idx=range(num_keypoints),
            keypoint_range=(middle_frame_idx, len(video)),
            backward_tracking=False
        ))
        # calculate bothway_rmse from tracking_data
        bothway_tracked_keypoints_reshaped = np.zeros((num_frames, num_keypoints, 2))
        for f_idx in range(tracking_data.keypoint_range[0], tracking_data.keypoint_range[1]):
            for kp_idx in range(num_keypoints):
                tracked_kp = tracking_data.keypoints[tracking_data.keypoints[:, 0] == f_idx][kp_idx][1:]
                annotated_kp = annotated_keypoints[annotated_keypoints[:, 0] == f_idx][kp_idx][1:]
                bothway_tracked_keypoints_reshaped[f_idx, kp_idx] = tracked_kp
                rmse = np.sqrt(np.mean((tracked_kp - annotated_kp) ** 2))
                bothway_rmse[f_idx, kp_idx] = rmse
        
        # Track from middle to start
        # Track from the current frame to the first frame
        logging.debug(f"Tracking keypoints from frame {middle_frame_idx} to frame {0}")
        keypoints_to_track = annotated_keypoints[annotated_keypoints[:, 0] == middle_frame_idx]
        keypoints_to_track[:, 0] = 0
        tracking_data = TrackingBenchmark.tracker.track(TrackingWorker.TrackingWorkerData(
            tracker="cotracker",
            video=video[0:middle_frame_idx+1],
            keypoints=keypoints_to_track,
            keypoint_features=annotated_keypoints_info[annotated_keypoints_info['frame_idx'] == middle_frame_idx],
            keypoint_order_idx=range(num_keypoints),
            keypoint_range=(0, middle_frame_idx+1),
            backward_tracking=True
        ))
        # calculate bothway_rmse from tracking_data
        for f_idx in range(tracking_data.keypoint_range[0], tracking_data.keypoint_range[1]):
            for kp_idx in range(num_keypoints):
                tracked_kp = tracking_data.keypoints[tracking_data.keypoints[:, 0] == f_idx][kp_idx][1:]
                annotated_kp = annotated_keypoints[annotated_keypoints[:, 0] == f_idx][kp_idx][1:]
                bothway_tracked_keypoints_reshaped[f_idx, kp_idx] = tracked_kp
                rmse = np.sqrt(np.mean((tracked_kp - annotated_kp) ** 2))
                bothway_rmse[f_idx, kp_idx] = rmse

        # insert the tracked keypoints one by one
        self.insert1(key, skip_duplicates=True)
        for kp_idx, (kp_subject_id, kp_name, kp_color) in enumerate(keypoint_order):
            self.TrackedKeypoint.insert1({
                **key,
                'subject_id': kp_subject_id,
                'segment_name': kp_name,
                'pose': tracked_keypoints_reshaped[:, kp_idx],
                'rmse': rmse_values[:, kp_idx],
                'forward_rmse': forward_rmse[:, kp_idx],
                'forward_pose': forward_tracked_keypoints_reshaped[:, kp_idx],
                'start_frame_idx': start_frame_idx,
                'bothway_rmse': bothway_rmse[:, kp_idx],
                'bothway_pose': bothway_tracked_keypoints_reshaped[:, kp_idx],
                'middle_frame_idx': middle_frame_idx,
                'tracking_events': tracking_events[:, kp_idx]
            })


    def render_overlay(key, output_filepath, pose_key_name='pose'):
        import matplotlib.colors as mcolors
        video = (Video & key).fetch1('video')
        keypoint_data = (TrackingBenchmark.TrackedKeypoint & key).fetch(as_dict=True)
        keypoint_colors = [(Session.Segment & kp_key).fetch1('segment_color') for kp_key in keypoint_data]
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        height, width = video[0].shape[:2]
        out = cv2.VideoWriter(output_filepath, fourcc, 10.0, (width, height))
        for frame_idx, frame in enumerate(video):
            frame = frame.copy()  # Make the frame writable
            for keypoint_datum, keypoint_color in zip(keypoint_data, keypoint_colors):
                kp = keypoint_datum[pose_key_name][frame_idx]
                if not np.isnan(kp).any():
                    color = tuple(int(c * 255) for c in mcolors.hex2color(keypoint_color))
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, color, -1)
            out.write(frame)
        out.release()

    def plot_rmse_landscape(key):
        # plot the forward_rmse and bothway_rmse for each keypoint
        keypoint_data = (TrackingBenchmark.TrackedKeypoint & key).fetch(as_dict=True)
        num_keypoints = len(keypoint_data)
        fig, axes = plt.subplots(num_keypoints, 1, figsize=(20, 6 * num_keypoints), sharex=True)
        if num_keypoints == 1:
            axes = [axes]

        for kp_idx, keypoint_datum in enumerate(keypoint_data):
            ax = axes[kp_idx]
            kp_name = keypoint_datum['segment_name']
            kp_color = (Session.Segment & keypoint_datum & f'segment_name="{kp_name}"').fetch1('segment_color')
            ax.plot(keypoint_datum['forward_rmse'], label='Forward RMSE', color='blue')
            ax.plot(keypoint_datum['bothway_rmse'], label='Bothway RMSE', color='green')
            ax.axvline(keypoint_datum['start_frame_idx'], linestyle='--', color='black', label='Start Frame')
            ax.axvline(keypoint_datum['middle_frame_idx'], linestyle='--', color='black', label='Middle Frame')
            ax.set_ylabel('RMSE')
            ax.set_title(f'RMSE of {kp_name} Over Frames')
            ax.legend()
            ax.set_xticks(np.arange(0, len(keypoint_datum['forward_rmse']), 5))
            ax.set_xticklabels(np.arange(0, len(keypoint_datum['forward_rmse']), 5))
    
    def plot_keypoint_rmse(key):
        config_plots()
        keypoint_data = (TrackingBenchmark.TrackedKeypoint & key).fetch(as_dict=True)
        window_size, rmse_threshold = key.fetch1('window_size', 'rmse_threshold')

        num_keypoints = len(keypoint_data)

        fig, axes = plt.subplots(num_keypoints, 1, figsize=(20, 6 * num_keypoints), sharex=True)
        if num_keypoints == 1:
            axes = [axes]

        for kp_idx, keypoint_datum in enumerate(keypoint_data):
            num_frames = len(keypoint_datum['rmse'])
            ax = axes[kp_idx]
            kp_name = keypoint_datum['segment_name']
            kp_color = (Session.Segment & keypoint_datum & f'segment_name="{kp_name}"').fetch1('segment_color')
            ax.plot(keypoint_datum['rmse'], label=kp_name, color=kp_color)
            event_frames = np.where(keypoint_datum['tracking_events'])[0]
            for event_frame in event_frames:
                ax.axvline(event_frame, linestyle='--', color='black')
            ax.axhline(rmse_threshold, linestyle='--', color='red', label='Threshold')
            ax.set_ylabel('RMSE')
            ax.set_title(f'RMSE of {kp_name} Over Frames (window_size={window_size})')
            ax.legend()
            ax.set_xticks(np.arange(0, num_frames, 5))
            ax.set_xticklabels(np.arange(0, num_frames, 5))

        plt.xlabel('Frames')
        plt.show()

    def stat_table(keys):
        if not isinstance(keys, list):
            keys = [keys]

        # Initialize a dictionary to store aggregated statistics
        aggregated_stats = {}

        for key in keys:
            keypoint_data = (TrackingBenchmark.TrackedKeypoint & key).fetch(as_dict=True)
            window_size, rmse_threshold = (TrackingBenchmarkSettingsLookup & key).fetch1('window_size', 'rmse_threshold')

            min_events = np.inf
            for keypoint_datum in keypoint_data:
                num_events = np.sum(keypoint_datum['tracking_events'])
                if num_events < min_events:
                    min_events = num_events

            for keypoint_datum in keypoint_data:
                start_frame_idx = keypoint_datum['start_frame_idx']
                forward_pose = keypoint_datum['forward_pose']
                forward_rmse = keypoint_datum['forward_rmse']
                try:
                    forward_break_off_duration = np.where(forward_rmse[start_frame_idx:] > rmse_threshold)[0][0]
                except:
                    forward_break_off_duration = len(forward_rmse) - start_frame_idx

                middle_frame_idx = keypoint_datum['middle_frame_idx']
                bothway_pose = keypoint_datum['bothway_pose']
                bothway_rmse = keypoint_datum['bothway_rmse']
                try:
                    bothway_break_off_forward_duration = np.where(bothway_rmse[middle_frame_idx:] > rmse_threshold)[0][0]
                except:
                    bothway_break_off_forward_duration = len(bothway_rmse) - middle_frame_idx

                try:
                    bothway_break_off_backward_duration = len(bothway_rmse[:middle_frame_idx]) - np.where(bothway_rmse[:middle_frame_idx] > rmse_threshold)[0][::-1][0]
                except:
                    bothway_break_off_backward_duration = middle_frame_idx

                subject_id = keypoint_datum['subject_id']
                segment_name = keypoint_datum['segment_name']
                avg_rmse = np.nanmean(keypoint_datum['rmse'])
                num_events = np.sum(keypoint_datum['tracking_events']) - min_events

                if segment_name not in aggregated_stats:
                    aggregated_stats[segment_name] = {
                        'avg_rmse': [],
                        'num_events': [],
                        'forward_break_off_duration': [],
                        'bothway_break_off_forward_duration': [],
                        'bothway_break_off_backward_duration': []
                    }

                aggregated_stats[segment_name]['avg_rmse'].append(avg_rmse)
                aggregated_stats[segment_name]['num_events'].append(num_events)
                aggregated_stats[segment_name]['forward_break_off_duration'].append(forward_break_off_duration)
                aggregated_stats[segment_name]['bothway_break_off_forward_duration'].append(bothway_break_off_forward_duration)
                aggregated_stats[segment_name]['bothway_break_off_backward_duration'].append(bothway_break_off_backward_duration)

        # Calculate the average statistics for each segment
        table_data = []
        for segment_name, stats in aggregated_stats.items():
            avg_rmse = np.nanmean(stats['avg_rmse'])
            num_events = np.mean(stats['num_events'])
            forward_break_off_duration = np.mean(stats['forward_break_off_duration'])
            bothway_break_off_forward_duration = np.mean(stats['bothway_break_off_forward_duration'])
            bothway_break_off_backward_duration = np.mean(stats['bothway_break_off_backward_duration'])

            table_data.append([segment_name, avg_rmse, num_events, forward_break_off_duration, bothway_break_off_forward_duration, bothway_break_off_backward_duration])

        avg_text = 'Average' if len(keys) > 1 else ''
        # Create a DataFrame for better visualization
        df = pd.DataFrame(table_data, columns=['Segment Name', f'{avg_text} RMSE', f'{avg_text} Number of Events', f'{avg_text} Hold Duration Forward', f'{avg_text} Hold Duration Bothway Forward', f'{avg_text} Hold Duration Bothway Backward'])

        # Plot the average number of events
        plt.figure(figsize=(10, 6))
        plt.bar(df['Segment Name'], df[f'{avg_text} Number of Events'], color='skyblue')
        plt.xlabel('Segment Name')
        plt.ylabel(f'{avg_text} Number of Events')
        plt.title(f'{avg_text} Number of Events per Segment')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Plot the average RMSE
        plt.figure(figsize=(10, 6))
        plt.bar(df['Segment Name'], df[f'{avg_text} RMSE'], color='salmon')
        plt.xlabel('Segment Name')
        plt.ylabel(f'{avg_text} RMSE')
        plt.title(f'{avg_text} RMSE per Segment')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Plot the average hold durations
        plt.figure(figsize=(10, 6))
        plt.bar(df['Segment Name'], df[f'{avg_text} Hold Duration Forward'], color='salmon')
        plt.xlabel('Segment Name')
        plt.ylabel(f'{avg_text} Hold Duration Forward')
        plt.title(f'{avg_text} Hold Duration Forward per Segment')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.bar(df['Segment Name'], df[f'{avg_text} Hold Duration Bothway Forward'], color='salmon')
        plt.xlabel('Segment Name')
        plt.ylabel(f'{avg_text} Hold Duration Bothway Forward')
        plt.title(f'{avg_text} Hold Duration Bothway Forward per Segment')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.bar(df['Segment Name'], df[f'{avg_text} Hold Duration Bothway Backward'], color='salmon')
        plt.xlabel('Segment Name')
        plt.ylabel(f'{avg_text} Hold Duration Bothway Backward')
        plt.title(f'{avg_text} Hold Duration Bothway Backward per Segment')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        if len(keys) == 1:
            # Plot the total number of events per frame
            keypoint_data = (TrackingBenchmark.TrackedKeypoint & key).fetch(as_dict=True)
            total_events_per_frame = np.sum([keypoint_datum['tracking_events'] for keypoint_datum in keypoint_data], axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(total_events_per_frame, color='purple')
            plt.xlabel('Frames')
            plt.ylabel('Total Number of Events')
            plt.title('Total Number of Events per Frame')
            plt.tight_layout()
            plt.show()

        return df





            

if __name__ == "__main__":
    session_to_benchmark = Session * (TrackingBenchmarkSettingsLookup & 'setting_id=1')
    TrackingBenchmark.populate(session_to_benchmark)
    # TrackingBenchmark.render_overlay(session_to_benchmark, 'tracked.mp4')
