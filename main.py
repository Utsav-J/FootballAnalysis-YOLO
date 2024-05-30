from utils import read_video, save_video
from trackers import Tracker
import cv2 as cv

def main():
    video_frames = read_video("input_videos/input.mp4")
    tracker = Tracker(model_path='models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks_stubs.pkl")
    #save the cropped iamge of the player

    # for track_id,player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]
    #     # cropped_frame = first_frame[startY:endY, startX:endX]
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
    #     #save the cropped image
    #     cv.imwrite(f"output_videos/cropped_img.jpg",cropped_image)
    #     break

    # draw output and object tracks
    output_video_frames = tracker.draw_annotations(video_frames=video_frames, tracks=tracks)
    
    save_video(output_video_frames,"output_videos/output.avi")

if __name__ == "__main__":
    main()