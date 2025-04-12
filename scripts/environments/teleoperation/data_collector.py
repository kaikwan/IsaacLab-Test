import cv2
import numpy as np

class DataCollector:
    """
    A simple data collector that records frames.
    At the end of a demonstration, it writes out a video (using OpenCV).
    """
    def __init__(self, demo_id, fps=30):
        self.demo_id = demo_id
        self.frames = []
        self.fps = fps
        self.video_filename = f"videos/demo_{demo_id}.mp4"
        print(f"[INFO] DataCollector initialized for demo #{demo_id}")

    def record(self, frame):
        """Record a frame (RGB image)."""
        self.frames.append(frame)

    def discard(self):
        """Discard everything collected so far."""
        self.frames = []

    def finalize(self):
        """Finalize the current demonstration by saving video."""
        if len(self.frames) == 0:
            print("[WARNING] No frames recorded for this demo, skipping video creation.")
            return

        # Save the video
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(self.video_filename, fourcc, self.fps, (width, height))

        for frame in self.frames:
            # Ensure the frame is in uint8 format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_out.write(frame_bgr)
        video_out.release()
        print(f"[INFO] Video saved to {self.video_filename}")
