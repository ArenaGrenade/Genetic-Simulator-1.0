import numpy as np
from flask.json import JSONEncoder
from sklearn.datasets import make_blobs
from Organism import Organism

map_colors = {
    0: [126, 200, 80, 255],
    1: [212, 241, 249, 255],
    2: [79, 134, 247, 255],
    3: [186, 140, 99, 255],
}

class Simulator(object):
    def __init__(self, gx, seed):
        self.gx = gx
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.u, self.v = np.meshgrid(np.arange(gx), np.arange(gx))
        self.game_map = self.init_rand_map()
        
        self.organisms = []
        self.init_organisms()
        
        self.time_step = 0
        self.day_num = 0
        self.is_summer = True
        self.is_day = True
        
        self.waterph_map = np.zeros((gx, gx), dtype=np.float32)
        self.treeph_map = np.zeros((gx, gx), dtype=np.float32)
        self.berryph_map = np.zeros((gx, gx), dtype=np.float32)
    
    def init_organisms(self):
        for i in range(0, self.gx, 11):
            for j in range(0, self.gx, 11):
                if self.game_map[i:i+11,j:j+11].shape == (11, 11) and np.sum(self.game_map[i:i+11,j:j+11]) == 0:
                    self.organisms.append(Organism(
                        [i, j],
                        96,
                        self.rng
                    ))
        print(len(self.organisms))
        
    def make_clustered_spawns(self, num_cent, num, closeness):
        spawn_locs = np.stack([self.rng.integers(0, self.gx, size=num_cent), self.rng.integers(0, self.gx, size=num_cent)]).T
        zone_indices = make_blobs(n_samples=num, centers=spawn_locs, cluster_std=closeness, random_state=self.seed)[0].astype(int)
        zones = np.zeros((self.gx, self.gx), dtype=bool)
        for x, y in zone_indices: 
            try: zones[x, y] = True
            except: continue
        return zones

    def generate_water_blobs(self):
        radius1 = self.rng.integers(10, 25)
        radius2 = self.rng.integers(20, 40)
        center1 = self.rng.integers(radius1, (self.gx - radius1, self.gx - radius1))
        c2theta, c2rad = -2 * np.pi * (self.rng.random() * 360), (radius1 + 10)
        center2 = [center1[0] + c2rad * np.cos(c2theta), center1[1] + c2rad * np.sin(c2theta)]
        
        dist_circ1 = np.sqrt((self.u - center1[0]) ** 2 + (self.v - center1[1]) ** 2) - radius1
        dist_circ2 = np.sqrt((self.u - center2[0]) ** 2 + (self.v - center2[1]) ** 2) - radius2
        k = 15
        h = np.clip(0.5 + 0.5 * (dist_circ1 - dist_circ2) / k, 0, 1)
        dist_scene = dist_circ1 * (1 - h) + dist_circ2 * h - k * h * (1 - h)
        return dist_scene <= 0
        
    def init_rand_map(self):
        # 0 is land
        # 1 is water
        # 2 is food
        # 3 is trees
        map = np.zeros((self.gx, self.gx), dtype=np.uint8)
        map[self.make_clustered_spawns(20, 150, 4)] = 3 # Spawn tree pods
        map[self.make_clustered_spawns(30, 150, 25)] = 2 # Spawn food pods
        map[self.generate_water_blobs()] = 1 # Create the water blob
        return map
    
    def render_map(self):
        map_render = np.zeros((*self.game_map.shape, 4), dtype=np.uint8)
        
        # Recolor
        for col_idx, color in map_colors.items():
            map_render[self.game_map == col_idx] = color
        
        # Render the organisms
        for organism in self.organisms:
            x, y = organism.location
            # print(y, y + 11)
            # print(map_render[x:x+11, y:y+11].shape)
            body_render = organism.render_body()
            # print(body_render.shape)
            map_render[x:x+11, y:y+11][organism.body != 0] = body_render[organism.body != 0]
        
        return map_render.ravel().tolist()
    
    def simulate_step(self):
        # Time related Variables Update
        if self.time_step >= 96:
            self.time_step = 0
            self.day_num += 1
        elif 0 <= self.time_step <= 48: 
            self.is_day = True
            self.time_step += 1
        else:
            self.is_day = False
            self.time_step += 1
        
        if (self.day_num % 8) < 4: self.is_summer = True
        else: self.is_summer = False
        
        # Simulate organisms for 1 time step
        for organism in self.organisms:
            action_map = organism.forward(self.gx, (0, 0, 0), (0, 0, 0), (0, 0, 0), self.is_day, self.is_summer, self.time_step, self.day_num)
            actions_taken = 0
            for action, value in zip(organism.outputs_names, action_map):
                if action == "O_EmitPh" and (1 / (1 + np.exp(-value))) > 0.5:
                    organism.food -= 8
                    organism.water -= 8
                    actions_taken += 1
                    continue
                elif action == "O_MoveX" and np.abs(value) > 0.5:
                    organism.location[0] = np.clip(np.sign(value) + organism.location[0], 0, self.gx - 11)
                    organism.food -= 2
                    organism.water -= 2
                    actions_taken += 1
                elif action == "O_MoveY" and np.abs(value) > 0.5:
                    organism.location[1] = np.clip(np.sign(value), 0, self.gx)
                    organism.food -= 2
                    organism.water -= 2
                    actions_taken += 1
                elif action == "O_MoveRand" and (1 / (1 + np.exp(-value))) >= 0.5:
                    organism.location[0] = np.clip(self.rng.integers(-1, 2) + organism.location[0], 0, self.gx - 11)
                    organism.location[1] = np.clip(self.rng.integers(-1, 2) + organism.location[1], 0, self.gx - 11)
                    organism.food -= 5
                    organism.water -= 5
                    actions_taken += 1
                elif action == "O_Mouth" and (1 / (1 + np.exp(-value))) > 0.5:
                    organism.food -= 1
                    organism.water -= 1
                    actions_taken += 1
                    continue
                elif action == "O_Razor" and (1 / (1 + np.exp(-value))) > 0.5:
                    organism.food -= 15
                    organism.water -= 15
                    actions_taken += 1
                    continue
            if actions_taken == 0:
                organism.food -= 1
                organism.water -= 1
            organism.food = max(0, organism.food)
            organism.water = max(0, organism.water)
        
        # Update organism collision avoidance
        for _ in range(3):
            for organismA in self.organisms:
                for organismB in self.organisms:
                    dist = [abs(a - b) for a, b in zip(organismA.location, organismB.location)]
                    if all(x < 11 for x in dist):
                        orgAmap = organismA.body[:-dist[0], :-dist[1]]
                        orgBmap = organismB.body[:dist[0], :dist[1]]
                        # print(orgAmap.shape, orgBmaps.hape)
                        # print(orgAmap + orgBmap)
                        
        new_organisms = []
        dead_organisms = []
        for org_id, organism in enumerate(self.organisms):
            # Update organism reproduction
            if organism.food >= 500 and organism.water >= 500:
                new_organisms.append(organism.reproduce())
        
            # Update organism death
            # ! need to handle case of carnivores
            if organism.food == 0 or organism.water == 0:
                dead_organisms.append(org_id)
        
        self.organisms = [org for idx, org in enumerate(self.organisms) if idx not in dead_organisms]
        
        # ! have to append new organisms to population with a new location and all that

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Simulator):
            return {
                'gameMap': obj.game_map.tolist(),
                # "organisms": obj.organisms,
                'timeStep': obj.time_step,
                'isSummer': obj.is_summer,
                "isDay": obj.is_day,
                "renderedMap": obj.render_map()
            }
        return super(CustomJSONEncoder, self).default(obj)
