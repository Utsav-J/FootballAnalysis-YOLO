from ultralytics import YOLO
import supervision as sv
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        detections = [] 
        batch_size = 20 # 20 frames will be detected at a time
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
            break ##JUST FOR NOW
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referee': [],
            'ball': []
        }

        # tracks = {
        #     'players': [
        #         {0:{1,2,3,4}, <track_id>: <bbox>, } # for the first frame
        #         {<track_id>: <bbox> for each player present in frame } # for the second frame
        #         # if player moves out of frame in consecutive frames, its tracker id wont be shown
        #     ],
        #     #same format for referees   
        #     'referee': [],
        #     'ball': []
        # }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names # 2:player 1:goalkeeper 0:ball
            class_names_inversed = {v:k for k,v in class_names.items()} # player:2 goalkeeper:1 ball:0

            # converting into supervision detection format
            supervision_detection = sv.Detections.from_ultralytics(detection)

            # convert the goalkeeper into player
            for object_index, class_id in enumerate(supervision_detection.class_id):
                if class_names[class_id] == 'goalkeeper':
                    supervision_detection.class_id[object_index] = class_names_inversed['player']

            # with Tracks
            detection_with_tracks = self.tracker.update_with_detections(supervision_detection)
            # details about the above detection with tracks object
                # 0 contains the bounding boxes
                # 1 contains the mask attribute
                # 2 contains the confidence arrays
                # 3 contains the class id
                # 4 containst he tracker id
                # 5 contains the data dictionary

            tracks['players'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inversed['player']:
                    tracks['players'][frame_num][track_id] = {"bbox":bbox}

                if class_id == class_names_inversed['referee']:
                    tracks['referee'][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in supervision_detection:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inversed['ball']:
                    tracks['ball'][frame_num][1] = {"bbox":bbox}

            # print(detection_with_tracks) 
        return tracks # a dictionary of list of dictionaries
    