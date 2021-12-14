import numpy as np
from copy import deepcopy

body_colors = {
    0: [0, 0, 0, 0],
    1: [0, 0, 0, 255],
    2: [255, 165, 0, 255],
    3: [221, 160, 221, 255]
}

class Neuron(object):
    def __init__(self, enabled=False, output=0, name=None):
        self.enabled = enabled
        self.output = output
        self.name = name
        
class Connection(object):
    def __init__(self, src, src_id, sink, sink_id, weight=0):
        self.sink = sink
        self.sink_id = sink_id
        self.source = src
        self.src_id = src_id
        self.weight = weight

# Body Cells - None, Mover, Mouth, Wedge
class Organism(object):
    def __init__(self, spawn_loc, day_night_steps, rng):
        self.rng = rng
        self.body = np.zeros((11, 11), dtype=np.uint8)
        self.body[5, 5] = 1
        
        self.age = 0
        self.food = 1000
        self.water = 1000
        self.wood = 0
        
        self.oscillator_period = day_night_steps
        self.oscillator_state = 0
        
        self.location = spawn_loc
        self.last_movement = [0, 0]
        
        self.input_names = ["I_Age", "I_Food", "I_Water", "I_Wood", "I_Oscillator", "I_WorldX", "I_WorldY", "I_LastX", "I_Last Y", "I_Rand", "I_BorderX", "I_BorderY", "I_BorderMin", "I_PhGradWX", "I_PhGradWY", "I_PhW", "I_PhGradTX", "I_PhGradTY", "I_PhT", "I_PhGradBX", "I_PhGradBY", "I_PhB", "I_IsDay", "I_IsSummer", "I_TimeStep", "I_DayNum"]
        self.outputs_names = ["O_OscSet", "O_EmitPhW", "O_EmitPhT", "O_EmitPhB", "O_MoveX", "O_MoveY", "O_MoveRand",  "O_Mouth", "O_Razor"]
        self.neurons = [Neuron(enabled=True, name=name) for name in self.outputs_names]
        self.connections = [Connection("inp", 9, "out", 6)]
    
    def render_body(self):
        rendered_org = np.zeros((*self.body.shape, 4), np.uint8)
        
        for col_idx, color in body_colors.items():
            rendered_org[self.body == col_idx] = color
        
        return rendered_org

    def forward(self, max_loc, waterph, treeph, berryph, is_day, is_summer, time_step, day_num, max_ts_day, max_days):
        # Handle Oscillator input
        oscillator_input = 0
        if self.oscillator_state >= self.oscillator_period:
            oscillator_input = 1
            self.oscillator_state = 0
        else: self.oscillator_state += 1
        
        # Construct the input matrix
        border_dist = [min(x, max_loc - x) for x in self.location]
        inputs = np.array([self.age, self.food, self.water, self.wood, oscillator_input, *self.location, *self.last_movement, self.rng.random(), *border_dist, min(border_dist), *waterph, *treeph, *berryph, is_day, is_summer, time_step, day_num])
        input_maximas = np.array([16 * 96, 1000, 1000, 300, self.oscillator_period, max_loc, max_loc, 1, 1, 1, max_loc, max_loc, max_loc, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, max_ts_day, max_days])
        inputs_scaled = inputs / input_maximas
        
        # Initialize state variables
        action_map = np.zeros(len(self.outputs_names))
        neuron_acc = np.zeros(len(self.neurons))
        outputs_computed = False
        
        # Perform feed forward
        for conn in self.connections:
            if conn.sink == "out" and not outputs_computed:
                for neuron, acc in zip(self.neurons, neuron_acc):
                    if neuron.enabled: neuron.output = np.tanh(acc)
                outputs_computed = True
            
            if conn.source == "inp": input_val = inputs_scaled[conn.src_id]
            else: input_val = self.neurons[conn.src_id].output
            
            if conn.sink == "out": action_map[conn.sink_id] += input_val * conn.weight
            else: neuron_acc += input_val * conn.weight
            
        # Handle the oscillator update
        if (1 / (1 + np.exp(-action_map[0]))) >= 0.5:
            self.oscillator_state += 1
            self.oscillator_state = int(self.oscillator_state % self.oscillator_period)
        
        return action_map

    def reproduce(self, new_loc):
        new_organism = deepcopy(self)
        
        # Reset all required variables
        new_organism.location = new_loc
        new_organism.last_movement = [0, 0]
        new_organism.oscillator_state = 0
        
        new_organism.age = 0
        new_organism.food = 1000
        new_organism.water = 1000
        new_organism.wood = 0
        
        # Mutate the body a bit
        dijs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        if new_organism.rng.random() < 1: # 0.3
            substrates = []
            for i in range(11):
                for j in range(11):
                    # Body Cells - None, Mover, Mouth, Wedge
                    bdn = new_organism.body
                    if bdn[i, j] == 0:
                        num_valid = sum(1 for di, dj in dijs if 0 <= (i + di) < 11 and 0 <= (j + dj) < 11 and bdn[i + di, j + dj] == 1)
                        if num_valid > 0: substrates.append((i, j))
            # Delete elements that are useless
            if len(substrates) == 0: return None
            chx, chy = self.rng.choice(substrates, 1)[0]
            if self.rng.random() < 0.03: # 0.03
                new_organism.body[chx, chy] = 3
            elif self.rng.random() < 0.05: # 0.05
                new_organism.body[chx, chy] = 2
            elif self.rng.random() < 0.1: # 0.1
                new_organism.body[chx, chy] = 1
        
        # Remove body configuration at any cell
        if new_organism.body.sum() > 0 and new_organism.rng.random() < 0.01:
            i = new_organism.rng.integers(0, 11)
            j = new_organism.rng.integers(0, 11)
            new_organism.body[i, j] = 0
        
        # Add neurons mutation
        if self.rng.random() < 0.1: # 0.1
            hidden_indices = [int(neuron.name[1:]) for neuron in new_organism.neurons if neuron.name[0] == "H"]
            for nidx in hidden_indices:
                if nidx < len(hidden_indices): hidden_indices[nidx - 1]
            for aidx, nidx in enumerate(hidden_indices):
                if nidx > 0:
                    new_organism.neurons.append(Neuron(enabled=True, name=f"H{aidx + 1}"))
                    break
                
        # Disable Neuron mutation
        for neuron in new_organism.neurons:
            if new_organism.rng.random() < 0.05: # 0.05
                neuron.enabled = False
                
        # Add connection mutation
        if new_organism.rng.random() < 0.1: # 0.1
            # Choose the first neuron
            if new_organism.rng.random() < 0.5: # Choose from inputs
                first_neuron = new_organism.rng.choice(len(new_organism.input_names), 1)[0]
                first_neuron_type = 'inp'
            else: # Choose from one of the neurons
                first_neuron = new_organism.rng.choice(len(new_organism.neurons), 1)[0]
                first_neuron_type = 'hid' if new_organism.neurons[first_neuron].name[0] == "H" else 'out'
            
            # Choose the second neuron
            sec_neuron = new_organism.rng.choice(len(new_organism.neurons), 1)[0]
            sec_neuron_type = 'hid' if new_organism.neurons[sec_neuron].name[0] == "H" else 'out'
            
            # Create the connection
            new_organism.connections.append(Connection(first_neuron_type, first_neuron, sec_neuron_type, sec_neuron))
        
        # Delete Connection Mutation
        if len(new_organism.connections) > 0 and new_organism.rng.random() < 0.01: # Delete Connection
            del new_organism.connections[new_organism.rng.choice(len(new_organism.connections), 1)[0]]
        
        # Add weight perturbation
        for conn in new_organism.connections:
            conn.weight = (conn.weight + (new_organism.rng.random() * 2 - 1) * 0.5) * new_organism.rng.random()
        
        return new_organism
