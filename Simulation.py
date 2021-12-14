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
        
        self.steps_per_day = 96
        self.days_per_iter = 32 / 32
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.u, self.v = np.meshgrid(np.arange(gx), np.arange(gx))
        self.game_map = self.init_rand_map()
        
        self.generation_num = 1
        self.generation_organisms = []
        
        self.organisms = []
        self.init_organisms()
        
        self.time_step = 1
        self.day_num = 1
        self.is_summer = True
        self.is_day = True
        
        self.waterph_map = np.zeros((gx, gx), dtype=np.float32)
        self.treeph_map = np.zeros((gx, gx), dtype=np.float32)
        self.berryph_map = np.zeros((gx, gx), dtype=np.float32)
    
    def init_organisms(self):
        for i in range(0, self.gx, 11):
            for j in range(0, self.gx, 11):
                if self.game_map[i:i+11,j:j+11].shape == (11, 11) and self.rng.random() < 0.5:
                    organism = Organism(
                        [i, j],
                        self.steps_per_day,
                        self.rng
                    )
                    self.organisms.append(organism)
                    self.generation_organisms.append(organism)
        
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
            try:
                map_render[x:x+11, y:y+11][organism.body != 0] = body_render[organism.body != 0]
            except:
                print(body_render.shape)
                print(organism.body)
                exit()
        
        return map_render.ravel().tolist()

    def next_generation(self):
        # Generate scores for each organism in generation
        best_organisms = sorted([(org, org.age + org.food + org.wood + org.water) for org in self.generation_organisms], key=lambda x: x[1])
        
        # Reset all generation variables
        self.generation_num += 1
        self.generation_organisms = []
        self.organisms = []
        
        self.time_step = 0
        self.day_num = 0
        self.is_summer = True
        self.is_day = True
        
        self.waterph_map = np.zeros((self.gx, self.gx), dtype=np.float32)
        self.treeph_map = np.zeros((self.gx, self.gx), dtype=np.float32)
        self.berryph_map = np.zeros((self.gx, self.gx), dtype=np.float32)
        
        # Create new organisms to replace old ones
        for i in range(0, self.gx, 11):
            for j in range(0, self.gx, 11):
                if self.game_map[i:i+11,j:j+11].shape == (11, 11) and best_organisms and self.rng.random() < 0.5: # np.sum(self.game_map[i:i+11,j:j+11]) == 0 and 
                    parent_organism =  best_organisms.pop(0)[0]
                    organism = parent_organism.reproduce([i, j])
                    if organism is None: continue
                    self.organisms.append(organism)
                    self.generation_organisms.append(organism)
                    if self.rng.random() < 0.1:
                        if self.rng.random() < 0.25:
                            best_organisms.append((parent_organism, 0))
                        else:
                            best_organisms.append((organism, 0))
    
    def simulate_step(self):
        # Time related Variables Update
        if self.time_step >= self.steps_per_day:
            self.time_step = 0
            self.day_num += 1
            if self.day_num >= self.days_per_iter:
                self.next_generation()
                return
        elif 0 <= self.time_step <= (self.steps_per_day // 2): 
            self.is_day = True
            self.time_step += 1
        else:
            self.is_day = False
            self.time_step += 1
        
        if (self.day_num % 8) < 4: self.is_summer = True
        else: self.is_summer = False
        
        # Simulate organisms for 1 time step
        for organism in self.organisms:
            # Calculate the gradients for pheromones
            # For water
            water_pheromone = self.waterph_map[organism.location[0]:organism.location[0]+11, organism.location[1]:organism.location[1]+11]
            water_grad_arr = np.gradient(water_pheromone)
            water_pheromone_data = (water_grad_arr[0].mean(), water_grad_arr[1].mean(), water_pheromone.mean())
            
            action_map = organism.forward(self.gx, water_pheromone_data, (0, 0, 0), (0, 0, 0), self.is_day, self.is_summer, self.time_step, self.day_num, self.steps_per_day, self.days_per_iter)
            actions_taken = 0
            for action, value in zip(organism.outputs_names, action_map):
                if action[:8] == "O_EmitPh" and (1 / (1 + np.exp(-value))) > 0.5:
                    organism.food -= 8
                    organism.water -= 8
                    actions_taken += 1
                    if action[-1] == "W":
                        self.waterph_map[organism.location[0]:organism.location[0]+11, organism.location[1]:organism.location[1]+11] += np.abs(value)
                        self.waterph_map = np.clip(self.waterph_map, 0, 10)
                    continue
                elif action == "O_MoveX" and np.abs(value) > 0.5:
                    organism.location[0] = int(np.clip(np.sign(value) + organism.location[0], 0, self.gx - 11))
                    organism.last_movement = [np.sign(value), 0]
                    organism.food -= 2
                    organism.water -= 2
                    actions_taken += 1
                elif action == "O_MoveY" and np.abs(value) > 0.5:
                    organism.location[1] = int(np.clip(organism.location[1] + np.sign(value), 0, self.gx))
                    organism.last_movement = [0, np.sign(value)]
                    organism.food -= 2
                    organism.water -= 2
                    actions_taken += 1
                elif action == "O_MoveRand" and (1 / (1 + np.exp(-value))) > 0.5:
                    x_shift = self.rng.integers(-1, 2)
                    y_shift = self.rng.integers(-1, 2)
                    organism.last_movement = [x_shift, y_shift]
                    organism.location[0] = int(np.clip(x_shift + organism.location[0], 0, self.gx - 11))
                    organism.location[1] = int(np.clip(y_shift + organism.location[1], 0, self.gx - 11))
                    organism.food -= 5
                    organism.water -= 5
                    actions_taken += 1
                elif action == "O_Mouth" and (1 / (1 + np.exp(-value))) >= 0.5:
                    organism.food -= 1
                    organism.water -= 1
                    actions_taken += 1
                    mouth_offsets = np.array(np.where(organism.body == 2)).T
                    for mouth_offset in mouth_offsets:
                        new_x = organism.location[0] + mouth_offset[0]
                        new_y = organism.location[1] + mouth_offset[1]
                        if self.game_map[new_x, new_y] == 1: # For consuming water
                            organism.water += 10
                            print("Drank Water")
                        elif self.game_map[new_x, new_y] == 2: # For consuming food
                            organism.food += 10
                            print("Ate Food")
                elif action == "O_Razor" and (1 / (1 + np.exp(-value))) > 0.5:
                    organism.food -= 15
                    organism.water -= 15
                    actions_taken += 1
                    razor_offsets = np.array(np.where(organism.body == 3)).T
                    for razor_offset in razor_offsets:
                        new_x = organism.location[0] + razor_offset[0]
                        new_y = organism.location[1] + razor_offset[1]
                        if self.game_map[new_x, new_y] == 3: # For cutting tree
                            organism.water += 15
                            print("Cut Tree")
            if actions_taken == 0:
                organism.food -= 1
                organism.water -= 1
            organism.food = max(0, organism.food)
            organism.water = max(0, organism.water)
        
        # Update organism collision avoidance
        # for _ in range(3):
        #     for organismA in self.organisms:
        #         for organismB in self.organisms:
        #             dist = [abs(a - b) for a, b in zip(organismA.location, organismB.location)]
        #             if all(x < 11 for x in dist):
        #                 try:
        #                     orgAmap = organismA.body[:-dist[0], :-dist[1]]
        #                     orgBmap = organismB.body[:dist[0], :dist[1]]
        #                 except:
        #                     print(organismA.body, organismA.location, organismB.location)
        #                     print(orgAmap.shape, orgBmap.shape)
                            # print(orgAmap + orgBmap)
                        
        new_organisms = []
        dead_organisms = []
        for org_id, organism in enumerate(self.organisms):
            # Update organism reproduction
            if organism.food >= 1500 and organism.water >= 1500:
                print("\n\nNEW ORGANISM\n\n")
                new_org = organism.reproduce()
                if new_org is not None: new_organisms.append(new_org)
        
            # Update organism death
            # ! need to handle case of carnivores
            if organism.food == 0 or organism.water == 0:
                dead_organisms.append(org_id)
            
            # Update organism age
            organism.age += 1
        
        self.organisms = [org for idx, org in enumerate(self.organisms) if idx not in dead_organisms]
        
        # ! have to append new organisms to population with a new location and all that

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Simulator):
            # Calculate some statistics
            stats = {
                "age": 0,
                "food": 0,
                "water": 0,
                "wood": 0,
                "connections": 0,
            }
            if len(obj.organisms) > 0:
                for organism in obj.organisms:
                    stats["age"] += organism.age
                    stats["food"] += organism.food
                    stats["wood"] += organism.wood
                    stats["water"] += organism.water
                    stats["connections"] += len(organism.connections)
                statistics = {k:f"{v / len(obj.organisms):.0f}" for k, v in stats.items()}
            return {
                'timeStep': obj.time_step,
                'dayNum': obj.day_num,
                'generation': obj.generation_num,
                "renderedMap": obj.render_map(),
                "stats": statistics
            }
        return super(CustomJSONEncoder, self).default(obj)
