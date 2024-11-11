"""
Inspired from elements
"""
import os
import glob

import datajoint as dj
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

class TrackingBenchmark(dj.Computed):