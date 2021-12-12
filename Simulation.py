import numpy as np
from flask.json import JSONEncoder
from sklearn.datasets import make_blobs

map_colors = {
    0: [126, 200, 80, 255],
    1: [212, 241, 249, 255],
    2: [79, 134, 247, 255],
    3: [186, 140, 99, 255],
}

class Simulator(object):
    def __init__(self, gx, seed):
        self.organisms = []
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.u, self.v = np.meshgrid(np.arange(gx), np.arange(gx))
        self.game_map = self.init_rand_map(gx)
        
        self.time_step = 0
        self.is_summer = True
        self.is_day = True
        
    def make_clustered_spawns(self, num_cent, num, closeness, gx):
        spawn_locs = np.stack([self.rng.integers(0, gx, size=num_cent), self.rng.integers(0, gx, size=num_cent)]).T
        zone_indices = make_blobs(n_samples=num, centers=spawn_locs, cluster_std=closeness, random_state=self.seed)[0].astype(int)
        zones = np.zeros((gx, gx), dtype=bool)
        for x, y in zone_indices: 
            try: zones[x, y] = True
            except: continue
        return zones

    def generate_water_blobs(self, gx):
        radius1 = self.rng.integers(10, 25)
        radius2 = self.rng.integers(20, 40)
        center1 = self.rng.integers(radius1, (gx - radius1, gx - radius1))
        c2theta, c2rad = -2 * np.pi * (self.rng.random() * 360), (radius1 + 10)
        center2 = [center1[0] + c2rad * np.cos(c2theta), center1[1] + c2rad * np.sin(c2theta)]
        
        dist_circ1 = np.sqrt((self.u - center1[0]) ** 2 + (self.v - center1[1]) ** 2) - radius1
        dist_circ2 = np.sqrt((self.u - center2[0]) ** 2 + (self.v - center2[1]) ** 2) - radius2
        k = 15
        h = np.clip(0.5 + 0.5 * (dist_circ1 - dist_circ2) / k, 0, 1)
        dist_scene = dist_circ1 * (1 - h) + dist_circ2 * h - k * h * (1 - h)
        return dist_scene <= 0
        
    def init_rand_map(self, gx):
        # 0 is land
        # 1 is water
        # 2 is food
        # 3 is trees
        map = np.zeros((gx, gx), dtype=np.uint8)
        map[self.make_clustered_spawns(20, 150, 4, gx)] = 3 # Spawn tree pods
        map[self.make_clustered_spawns(30, 150, 25, gx)] = 2 # Spawn food pods
        map[self.generate_water_blobs(gx)] = 1 # Create the water blob
        return map
    
    def render_map(self):
        map_render = np.zeros((*self.game_map.shape, 4), dtype=np.uint8)
        
        # Recolor
        for col_idx, color in map_colors.items():
            map_render[self.game_map == col_idx] = color
        
        return map_render.ravel().tolist()

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Simulator):
            return {
                'gameMap': obj.game_map.tolist(),
                "organisms": obj.organisms,
                'timeStep': obj.time_step,
                'isSummer': obj.is_summer,
                "isDay": obj.is_day,
                "renderedMap": obj.render_map()
            }
        return super(CustomJSONEncoder, self).default(obj)
