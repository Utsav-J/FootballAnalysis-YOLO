from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video("input_videos/input.mp4")
    
    tracker = Tracker(model_path='models/best.pt')
    tracker.get_object_tracks(video_frames)

    save_video(video_frames,"output_videos/output.avi")

if __name__ == "__main__":
    main()