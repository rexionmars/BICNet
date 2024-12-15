import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import logging
from pathlib import Path
import seaborn as sns
from collections import deque
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='integrated_brain.log'
)
logger = logging.getLogger(__name__)

class TimeDelayBuffer:
    """Implementa delays temporais para sinais biológicos"""
    def __init__(self, delay_steps: int):
        self.buffer = deque(maxlen=delay_steps)
        self.delay_steps = delay_steps
        
    def add(self, value):
        self.buffer.append(value)
        
    def get(self):
        if len(self.buffer) < self.delay_steps:
            return 0.0
        return self.buffer[0]

class EpigeneticRegulation:
    def __init__(self):
        self.methylation_state = {}
        self.histone_modifications = {}
        self.chromatin_state = {}
        self.delay_buffer = TimeDelayBuffer(10)  # 10 timesteps delay
        
    def update_state(self, gene: str, stress_level: float, activity: float):
        # Atualiza metilação com delay
        self.delay_buffer.add(stress_level)
        delayed_stress = self.delay_buffer.get()
        
        self.methylation_state[gene] = np.clip(
            self.methylation_state.get(gene, 0.5) + 0.1 * delayed_stress - 0.05 * activity,
            0, 1
        )
        
        # Atualiza modificações de histonas
        self.histone_modifications[gene] = np.clip(
            self.histone_modifications.get(gene, 0.5) + 0.1 * activity,
            0, 1
        )
        
        # Atualiza estado da cromatina
        self.chromatin_state[gene] = (
            self.methylation_state[gene] * 0.7 +
            self.histone_modifications[gene] * 0.3
        )
        
        return self.chromatin_state[gene]

class SignalingPathway:
    def __init__(self, name: str):
        self.name = name
        self.components = {}
        self.activation_history = []
        self.delay_buffer = TimeDelayBuffer(5)
        
    def activate(self, signal_strength: float):
        self.delay_buffer.add(signal_strength)
        delayed_signal = self.delay_buffer.get()
        
        for component in self.components:
            current_level = self.components[component]
            # Dinâmica não-linear com saturação
            new_level = current_level + (
                0.1 * delayed_signal * (1 - current_level) -
                0.05 * current_level
            )
            self.components[component] = np.clip(new_level, 0, 1)
        
        self.activation_history.append(
            np.mean(list(self.components.values()))
        )

class GeneNetwork:
    def __init__(self):
        self.epigenetics = EpigeneticRegulation()
        self.expression_levels = {}
        self.regulatory_network = nx.DiGraph()
        self._initialize_network()
        
    def _initialize_network(self):
        # Genes do desenvolvimento
        self.add_gene_group('development', [
            'Sox2', 'Nestin', 'Pax6', 'Dcx', 'Reelin'
        ])
        
        # Genes de plasticidade
        self.add_gene_group('plasticity', [
            'Arc', 'Homer1a', 'BDNF', 'TrkB', 'CaMKII'
        ])
        
        # Genes do ciclo circadiano
        self.add_gene_group('circadian', [
            'Per1', 'Per2', 'Cry1', 'Bmal1', 'Clock'
        ])
        
    def add_gene_group(self, group: str, genes: List[str]):
        for gene in genes:
            self.expression_levels[gene] = 0.0
            self.regulatory_network.add_node(
                gene, 
                group=group,
                expression=0.0
            )
        
        # Adiciona interações dentro do grupo
        for i, gene1 in enumerate(genes):
            for gene2 in genes[i+1:]:
                if np.random.random() < 0.3:  # 30% chance de interação
                    self.regulatory_network.add_edge(
                        gene1, gene2,
                        weight=np.random.normal(0.5, 0.1)
                    )

class BrainRegion:
    def __init__(self, name: str):
        self.name = name
        self.gene_network = GeneNetwork()
        self.signaling_pathways = {
            'MAPK': SignalingPathway('MAPK'),
            'CREB': SignalingPathway('CREB'),
            'calcium': SignalingPathway('calcium')
        }
        self.activity_level = 0.0
        self.stress_level = 0.0
        
    def update(self, input_signal: float, stress: float):
        self.activity_level = np.clip(
            self.activity_level * 0.9 + 0.1 * input_signal,
            0, 1
        )
        self.stress_level = np.clip(
            self.stress_level * 0.95 + 0.05 * stress,
            0, 1
        )
        
        # Atualiza vias de sinalização
        for pathway in self.signaling_pathways.values():
            pathway.activate(self.activity_level)
        
        # Atualiza expressão gênica
        self._update_gene_expression()
        
    def _update_gene_expression(self):
        for gene in self.gene_network.expression_levels:
            # Influência epigenética
            epigenetic_state = self.gene_network.epigenetics.update_state(
                gene, self.stress_level, self.activity_level
            )
            
            # Influência da rede regulatória
            regulatory_input = 0.0
            for _, neighbor, data in self.gene_network.regulatory_network.edges(gene, data=True):
                regulatory_input += (
                    data['weight'] *
                    self.gene_network.expression_levels[neighbor]
                )
            
            # Atualiza expressão
            self.gene_network.expression_levels[gene] = np.clip(
                self.gene_network.expression_levels[gene] * 0.9 +
                0.1 * (regulatory_input + self.activity_level) *
                (1 - epigenetic_state),
                0, 1
            )

class IntegratedBrain:
    def __init__(self):
        # Inicializa regiões cerebrais
        self.regions = {
            'hippocampus': BrainRegion('hippocampus'),
            'cortex': BrainRegion('cortex'),
            'amygdala': BrainRegion('amygdala')
        }
        
        self.global_state = {
            'stress': 0.0,
            'arousal': 0.0,
            'plasticity': 0.0
        }
        
    def update(self, sensory_input: Dict[str, float], dt: float):
        # Atualiza estado global
        self.global_state['stress'] = np.clip(
            self.global_state['stress'] * 0.95 +
            0.05 * np.mean(list(sensory_input.values())),
            0, 1
        )
        
        # Atualiza cada região
        for name, region in self.regions.items():
            region.update(
                sensory_input.get(name, 0.0),
                self.global_state['stress']
            )
        
        # Atualiza plasticidade global
        self.global_state['plasticity'] = np.mean([
            region.activity_level for region in self.regions.values()
        ])

class BrainVisualizer:
    def __init__(self, brain: IntegratedBrain, output_dir: str = "brain_viz"):
        self.brain = brain
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def visualize_state(self, timestep: int):
        plt.figure(figsize=(20, 15))
        
        # Estado global
        plt.subplot(331)
        self._plot_global_state()
        plt.title('Global Brain State')
        
        # Atividade regional
        plt.subplot(332)
        self._plot_regional_activity()
        plt.title('Regional Activity')
        
        # Expressão gênica
        plt.subplot(333)
        self._plot_gene_expression()
        plt.title('Gene Expression')
        
        # Rede regulatória
        plt.subplot(334)
        self._plot_regulatory_network()
        plt.title('Gene Regulatory Network')
        
        # Estado epigenético
        plt.subplot(335)
        self._plot_epigenetic_state()
        plt.title('Epigenetic State')
        
        # Vias de sinalização
        plt.subplot(336)
        self._plot_signaling_pathways()
        plt.title('Signaling Pathways')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'brain_state_{timestep}.png')
        plt.close()
        
    def _plot_global_state(self):
        plt.bar(self.brain.global_state.keys(),
                self.brain.global_state.values())
        
    def _plot_regional_activity(self):
        activities = {name: region.activity_level 
                     for name, region in self.brain.regions.items()}
        plt.bar(activities.keys(), activities.values())
        
    def _plot_gene_expression(self):
        # Pega expressão de uma região exemplo
        sample_region = self.brain.regions['hippocampus']
        expression_data = np.array(list(
            sample_region.gene_network.expression_levels.values()
        )).reshape(-1, 1)
        
        sns.heatmap(expression_data, 
                   xticklabels=['Expression'],
                   yticklabels=list(
                       sample_region.gene_network.expression_levels.keys()
                   ))
        
    def _plot_regulatory_network(self):
        # Pega rede de uma região exemplo
        sample_network = self.brain.regions['hippocampus'].gene_network.regulatory_network
        pos = nx.spring_layout(sample_network)
        nx.draw(sample_network, pos, 
                node_color='lightblue',
                node_size=500,
                with_labels=True,
                font_size=8)
        
    def _plot_epigenetic_state(self):
        # Pega estado epigenético de uma região exemplo
        sample_region = self.brain.regions['hippocampus']
        epigenetic_data = np.array(list(
            sample_region.gene_network.epigenetics.methylation_state.values()
        )).reshape(-1, 1)
        
        sns.heatmap(epigenetic_data,
                   xticklabels=['Methylation'],
                   yticklabels=list(
                       sample_region.gene_network.epigenetics.methylation_state.keys()
                   ))
        
    def _plot_signaling_pathways(self):
        # Pega ativação das vias de uma região exemplo
        sample_region = self.brain.regions['hippocampus']
        for name, pathway in sample_region.signaling_pathways.items():
            if pathway.activation_history:
                plt.plot(pathway.activation_history[-50:], label=name)
        plt.legend()

def run_simulation():
    # Inicializa cérebro e visualizador
    brain = IntegratedBrain()
    visualizer = BrainVisualizer(brain)
    
    # Parâmetros de simulação
    dt = 0.01
    total_time = 10.0  # segundos
    steps = int(total_time / dt)
    
    # Loop de simulação
    for step in range(steps):
        # Gera entrada sensorial simulada
        sensory_input = {
            'hippocampus': 0.5 + 0.3 * np.sin(2 * np.pi * step * dt),
            'cortex': 0.4 + 0.2 * np.cos(2 * np.pi * step * dt),
            'amygdala': 0.3 + 0.4 * np.sin(4 * np.pi * step * dt)
        }
        
        # Atualiza cérebro
        brain.update(sensory_input, dt)
        
        # Visualiza periodicamente
        if step % 100 == 0:
            visualizer.visualize_state(step)
            logger.info(f"Completed step {step}/{steps}")
            
if __name__ == "__main__":
    run_simulation()