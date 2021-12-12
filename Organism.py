import numpy as np

#### Neural Network Inputs
# Vision (1x5)
# PheromoneGradA - Water
# PheromoneGradB - Tree
# PheromoneGradC - Berries
# PheromoneA - Water
# PheromoneB - Tree
# PheromoneC - Berries
# Age
# Random
# Oscillator - Daily
# Last Movement - X, Y
# Map Edge Distance - X, Y
# Nearest Border Distance
# World Location - X, Y
####
#### Neural Network Outputs
# Set Oscillator Period
# Emit Pheromone
# Move - X, Y
# Move Random
# Use Mouth
# Use Razor
class Organism(object):
    def __init__(self):
        self.body = np.array((11, 11))
        self.network = np.array(())
        
        self.age = 0
        self.food = 100
        self.water = 100
