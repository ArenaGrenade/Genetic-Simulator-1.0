import numpy as np

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
    def __init__(self, sink, src, sink_id, src_id, weight=0):
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
        self.food = 100
        self.water = 100
        self.wood = 0
        
        self.oscillator_period = day_night_steps
        self.oscillator_state = 0
        
        self.location = spawn_loc
        self.last_movement = [0, 0]
        
        self.input_names = ["I_Age", "I_Food", "I_Water", "I_Wood", "I_Oscillator", "I_WorldX", "I_WorldY", "I_LastX", "I_Last Y", "I_Rand", "I_BorderX", "I_BorderY", "I_BorderMin", "I_PhGradWX", "I_PhGradWY", "I_PhW", "I_PhGradTX", "I_PhGradTY", "I_PhT", "I_PhGradBX", "I_PhGradBY", "I_PhB", "I_IsDay", "I_IsSummer", "I_TimeStep", "I_DayNum"]
        self.outputs_names = ["O_OscSet", "O_EmitPh", "O_MoveX", "O_MoveY", "O_MoveRand",  "O_Mouth", "O_Razor"]
        self.neurons = [Neuron(enabled=True, name=name) for name in self.outputs_names]
        self.connections = []
    
    def render_body(self):
        rendered_org = np.zeros((*self.body.shape, 4), np.uint8)
        
        for col_idx, color in body_colors.items():
            rendered_org[self.body == col_idx] = color
        
        return rendered_org

    def forward(self, max_loc, waterph, treeph, berryph, is_day, is_summer, time_step, day_num):
        # Handle Oscillator input
        oscillator_input = 0
        if self.oscillator_state >= self.oscillator_period:
            oscillator_input = 1
            self.oscillator_state = 0
        else: self.oscillator_state += 1
        
        # Construct the input matrix
        border_dist = [min(x, max_loc - x) for x in self.location]
        inputs = np.array([self.age, self.food, self.water, self.wood, oscillator_input, *self.location, *self.last_movement, self.rng.random(), *border_dist, min(border_dist), *waterph, *treeph, *berryph, is_day, is_summer, time_step, day_num])
        input_maximas = np.array([16 * 96, 1000, 1000, 300, self.oscillator_period, max_loc, max_loc, 1, 1, 1, max_loc, max_loc, max_loc, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 96, 32])
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

    def reproduce(self):
        pass
