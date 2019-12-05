import cv2
import numpy as np


class VideoWriter:

    def __init__(self, outfile=None):
        self.outfile = outfile
        self.frame_resolution = None
        self.fps = 30  # TODO: read from video file eventually
        self.writer = None

    def write(self, frame_2d, frame_3d):
        if not self.outfile:
            return

        # lazy-init when first frame is written
        if not self.writer:
            self.init_writer(frame_2d)

        frame = self.combine(frame_2d, frame_3d)
        self.writer.write(frame)

    def init_writer(self, frame_2d):
        print(f"Output will be written to {self.outfile}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # appears to work fine when outfile is *.mp4 or *.mov
        height, width, _ = frame_2d.shape

        self.frame_resolution = height, width  # numpy notation
        resolution_combined = (width * 2, height)  # opencv notation

        self.writer = cv2.VideoWriter(self.outfile, fourcc, self.fps, resolution_combined)

    def release(self):
        if self.outfile and self.writer:
            self.writer.release()
            self.writer = None
            print(f"Finished writing out file to {self.outfile}")

    def combine(self, frame_2d, frame_3d):
        height, width = self.frame_resolution

        frame_3d = self.pad(frame_3d, height, width)
        frame_3d = self.crop(frame_3d, height, width)
        return  np.concatenate((frame_2d, frame_3d), axis=1)

    @staticmethod
    def pad(frame, height, width):
        pad_top = pad_bottom = 0
        pad_left = pad_right = 0

        if height > frame.shape[0]:
            pad_top = int((height - frame.shape[0]) / 2)
            pad_bottom = height - frame.shape[0] - pad_top

        if width > frame.shape[1]:
            pad_left = int((width - frame.shape[1]) / 2)
            pad_right = width - frame.shape[1] - pad_left

        return cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    @staticmethod
    def crop(frame, height, width):

        crop_top = crop_bottom = 0
        crop_left = crop_right = 0

        if frame.shape[0] > height:
            crop_top = int((frame.shape[0] - height) / 2)
            crop_bottom = frame.shape[0] - height - crop_top

        if frame.shape[1] > width:
            crop_left = int((frame.shape[1] - width) / 2)
            crop_right = frame.shape[1] - width - crop_left

        return frame[crop_top:frame.shape[0]-crop_bottom, crop_left:frame.shape[1]-crop_right, :]