from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        pass

    def get_clustering_model(self, top_half):
        top_half_2d = top_half.reshape(-1,3)
        kmeans = KMeans(n_clusters=2,init='k-means++', n_init=1, random_state=0)
        kmeans.fit(top_half)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        top_half_image = image[0: int(image.shape[1]//2),:]


        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [clustered_image[0,0],clustered_image[-1,0],clustered_image[0,-1],clustered_image[-1,-1]]
        background_cluster = max(corner_clusters, key=corner_clusters.count)
        player_cluster = 1 - background_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color        

    def assign_team_color(self, frame,player_detections):
        player_color = []
        for track_id,player in player_detections:
            bbox=  player['bbox']
            player_color = self.get_player_color(frame, bbox)
