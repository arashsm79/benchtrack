import os
import glob

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from benchtrack.benchtrack_dj import Project, Subject, Session, Video, VideoInfo

def import_from_horse30(dir_path: str):
    project_id = "proj1"
    Project.insert1({
        'project_id': project_id,
        'project_name': 'benchtrack',
        'project_description': f'Benchmarking tracking algorithms'
    }, skip_duplicates=True)

    for i, subject_name in enumerate(os.listdir(dir_path)):
        subject_path = os.path.join(dir_path, subject_name)
        if os.path.isdir(subject_path):
            subject_id = f"sub-{i+1}"
            session_id = f"ses-{i+1}"
            
            Subject.insert1({
                'subject_id': subject_id,
                'subject_name': subject_name,
                'subject_description': f'Subject {subject_name}'
            }, skip_duplicates=True)
            
            Session.insert1({
                'project_id': project_id,
                'session_id': session_id,
                'session_description': f'Session for {subject_name}'
            }, skip_duplicates=True)
            
            # Associate Subject with Session
            Session.Subject.insert1({
                'project_id': project_id,
                'session_id': session_id,
                'subject_id': subject_id
            }, skip_duplicates=True)
            
            # Concatenate images into a matrix
            image_files = sorted(glob.glob(os.path.join(subject_path, '*.png')))
            frames = [cv2.imread(img) for img in image_files]
            video = np.stack(frames, axis=0)
            
            video_id = f"vid-{i+1}_{subject_name}"
            Video.insert1({
                'project_id': project_id,
                'session_id': session_id,
                'video_id': video_id,
                'video': video
            }, skip_duplicates=True)
            
            VideoInfo.insert1({
                'project_id': project_id,
                'session_id': session_id,
                'video_id': video_id,
                'description': f'Video for {subject_name}',
                'fps': 30.0,
                'height': frames[0].shape[0],
                'width': frames[0].shape[1],
                'num_frames': len(frames)
            }, skip_duplicates=True)
            
            # Parse H5 for body parts and annotations
            hdf_path = os.path.join(subject_path, 'CollectedData_Byron.h5')
            df = pd.read_hdf(hdf_path)
            df = df.get(df.columns.levels[0][0]) # remove scorer name
            body_parts = sorted(df.columns.levels[0].to_list())

            def generate_bright_colors(n):
                from matplotlib.colors import to_hex
                colors = plt.cm.hsv(np.linspace(0, 1, n))  # 'hsv' provides distinct hues
                return [to_hex(color) for color in colors]

            for segment, color in zip(body_parts, generate_bright_colors(len(body_parts))):
                Session.Segment.insert1({
                    'project_id': project_id,
                    'session_id': session_id,
                    'segment_name': segment,
                    'segment_color': color
                }, skip_duplicates=True)

            # Prepare data for AnnotatedKeypoints
            num_frames = len(df)
            assert num_frames == len(video)
            frame_indices = range(num_frames)
            num_bodyparts = len(body_parts)
            for segment in body_parts:
                pose = np.zeros((num_frames, 2))
                for frame_idx, (index, row) in enumerate(df.iterrows()):
                    pose[frame_idx, 0] = row[segment, 'x']
                    pose[frame_idx, 1] = row[segment, 'y']
                Session.AnnotatedKeypoint.insert1({
                    'project_id': project_id,
                    'session_id': session_id,
                    'subject_id': subject_id,
                    'segment_name': segment,
                    'frame_idx': frame_indices,
                    'pose': pose,
                    'likelihood': None
                }, skip_duplicates=True)

            print(f"Imported {subject_name}.")
            return