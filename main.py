from utils import read_video, save_video
from team_assigner import TeamAssigner
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
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],track['bbox'],player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # draw output and object tracks
    output_video_frames = tracker.draw_annotations(video_frames=video_frames, tracks=tracks)

                                    
    
    save_video(output_video_frames,"output_videos/output.avi")

if __name__ == "__main__":
    main()