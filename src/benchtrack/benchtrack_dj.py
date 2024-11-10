"""
Inspired from elements
"""
import datajoint as dj
import os
import glob
import cv2
import pandas as pd
import numpy as np

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

    class Segments(dj.Part):
        definition = """
        -> master
        -> Session.Subject
        ---
        segment_names: longblob
        """

    class AnnotatedKeypoints(dj.Part):
        """
        Attributes:
        - frame_idx: list of frame indices
        - pose: ndarray of shape (num_frames, num_parts, keypoint dimension)
        - likelihood: ndarray of shape (num_frames, num_parts)
        """
        definition = """
        -> master
        -> Session.Segments
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

            Session.Segments.insert1({
                'project_id': project_id,
                'session_id': session_id,
                'subject_id': subject_id,
                'segment_names': body_parts
            }, skip_duplicates=True)

            # Prepare data for AnnotatedKeypoints
            num_frames = len(df)
            assert num_frames == len(video)
            frame_indices = range(num_frames)
            num_bodyparts = len(body_parts)
            pose = np.zeros((num_frames, num_bodyparts, 2))
            for frame_idx, (index, row) in enumerate(df.iterrows()):
                for part_idx, part in enumerate(body_parts):
                    pose[frame_idx, part_idx, 0] = row[part, 'x']
                    pose[frame_idx, part_idx, 1] = row[part, 'y']
            
            Session.AnnotatedKeypoints.insert1({
                'project_id': project_id,
                'session_id': session_id,
                'subject_id': subject_id,
                'frame_idx': frame_indices,
                'pose': pose,
                'likelihood': None
            }, skip_duplicates=True)
            print(f"Imported {subject_name}.")


if __name__ == "__main__":
    import_from_horse30("/home/arash/dev/benchtrack/data/horse10/labeled-data")