from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from logging import Logger

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_activity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


from brain_structures import NeuralRegion, Neurotransmitter
from brain_visualization import BrainVisualizer

def create_rat_brain():
    regions = {
        'hippocampus': NeuralRegion('hippocampus', 100, (0, 0, 0)),
        'amygdala': NeuralRegion('amygdala', 50, (1, 0, 0)),
        'prefrontal': NeuralRegion('prefrontal', 80, (0, 1, 0)),
        'motor_cortex': NeuralRegion('motor_cortex', 40, (1, 1, 0))
    }
    
    # Estabelece conexões
    regions['hippocampus'].connect_to(regions['prefrontal'])
    regions['amygdala'].connect_to(regions['prefrontal'])
    regions['prefrontal'].connect_to(regions['motor_cortex'])
    
    logger.info("Rat brain created with all regions and connections")
    return regions

def simulate_and_visualize(timesteps: int = 100):
    brain = create_rat_brain()
    visualizer = BrainVisualizer()
    
    # Plota estrutura inicial
    visualizer.plot_brain_network(brain)
    
    # Simula e visualiza atividade
    for t in range(timesteps):
        # Simula input sensorial
        sensory_input = np.random.normal(0, 1, (100, 1))
        
        # Atualiza cada região
        for region in brain.values():
            # Simula ativação
            region.activation_matrix = np.tanh(np.random.normal(0, 1, (len(region.neurons), 1)))
            
            # Atualiza neurotransmissores
            for nt in region.neurotransmitters.values():
                nt.update(np.random.normal(0, 0.1))
            
            if t % 10 == 0:  # Salva visualizações a cada 10 passos
                visualizer.plot_region_activity(region, t)
                visualizer.plot_neurotransmitter_levels(region, t)
                
        logger.debug(f"Completed timestep {t}")
        
    return brain

if __name__ == "__main__":
    brain = simulate_and_visualize()