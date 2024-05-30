from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video("input_videos/input.mp4")
    
    tracker = Tracker(model_path='models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks_stubs.pkl")
    # draw output
    # draw object tracks

    output_video_frames = tracker.draw_annotations(video_frames=video_frames, tracks=tracks)

    save_video(output_video_frames,"output_videos/output.avi")

if __name__ == "__main__":
    main()