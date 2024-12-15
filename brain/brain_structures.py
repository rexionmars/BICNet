from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_activity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Neurotransmitter:
    level: float = 1.0
    baseline: float = 1.0
    decay_rate: float = 0.01
    
    def update(self, stimulus: float):
        self.level += stimulus
        # Retorno à linha base
        self.level += (self.baseline - self.level) * self.decay_rate
        logger.debug(f"Neurotransmitter updated - Level: {self.level:.3f}")

class Neuron:
    def __init__(self, id: int):
        self.id = id
        self.membrane_potential = -70.0  # mV
        self.threshold = -55.0
        self.is_active = False
        self.connections = []
        self.activity_history = []
        
    def activate(self, input_current: float) -> float:
        self.membrane_potential += input_current
        
        if self.membrane_potential >= self.threshold:
            self.is_active = True
            output = 1.0
            self.membrane_potential = -70.0
        else:
            self.is_active = False
            output = 0.0
            
        self.activity_history.append(output)
        return output

class NeuralRegion:
    def __init__(self, name: str, num_neurons: int, position: Tuple[float, float, float]):
        self.name = name
        self.position = position
        self.neurons = [Neuron(i) for i in range(num_neurons)]
        
        self.neurotransmitters = {
            'dopamine': Neurotransmitter(),
            'serotonin': Neurotransmitter(),
            'glutamate': Neurotransmitter(),
            'gaba': Neurotransmitter()
        }
        
        self.connections_to = {}
        self.activation_matrix = np.zeros((num_neurons, 1))
        logger.info(f"Created neural region {name} with {num_neurons} neurons")

    def connect_to(self, target: 'NeuralRegion', weight_matrix: Optional[np.ndarray] = None):
        if weight_matrix is None:
            weight_matrix = np.random.normal(0, 0.1, (len(target.neurons), len(self.neurons)))
        self.connections_to[target.name] = {
            'target': target,
            'weights': weight_matrix
        }
        logger.debug(f"Connected {self.name} to {target.name}")