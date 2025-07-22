from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

from brain_structures import NeuralRegion, Neurotransmitter


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_activity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



class BrainVisualizer:
    def __init__(self, save_dir: str = "brain_visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_brain_network(self, regions: Dict[str, NeuralRegion]):
        G = nx.Graph()
        
        # Adiciona nós
        for name, region in regions.items():
            G.add_node(name, pos=region.position[:2])  # Usa apenas x,y para visualização 2D
            
        # Adiciona conexões
        for region_name, region in regions.items():
            for target_name in region.connections_to:
                G.add_edge(region_name, target_name)
        
        plt.figure(figsize=(12, 8))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Desenha a rede
        nx.draw(G, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                font_size=10,
                font_weight='bold')
        
        plt.savefig(self.save_dir / 'brain_network.png')
        plt.close()
        
    def plot_region_activity(self, region: NeuralRegion, timestep: int):
        plt.figure(figsize=(10, 6))
        
        # Matriz de ativação
        sns.heatmap(region.activation_matrix.reshape(-1, 1),
                   cmap='viridis',
                   xticklabels=False,
                   yticklabels=False)
        
        plt.title(f'{region.name} Activity - Timestep {timestep}')
        plt.savefig(self.save_dir / f'{region.name}_activity_{timestep}.png')
        plt.close()
        
    def plot_neurotransmitter_levels(self, region: NeuralRegion, timestep: int):
        levels = {name: nt.level for name, nt in region.neurotransmitters.items()}
        
        plt.figure(figsize=(8, 4))
        plt.bar(levels.keys(), levels.values())
        plt.title(f'{region.name} Neurotransmitter Levels - Timestep {timestep}')
        plt.ylim(0, 2)
        plt.savefig(self.save_dir / f'{region.name}_neurotransmitters_{timestep}.png')
        plt.close()