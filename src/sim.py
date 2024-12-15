import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging
from pathlib import Path

# Importa os diferentes módulos
from biological_memory_v2 import MemoryNetwork, MemoryVisualizer
from complex_interactions import ComplexNeuralAssembly, ComplexInteractionVisualizer
from dense_gene_network import DenseGeneNetwork, GeneNetworkVisualizer
from advanced_neural_dynamics import EnhancedNeuralNetwork, EnhancedVisualizer

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='virtual_rat.log'
)
logger = logging.getLogger(__name__)

import json
import seaborn as sns
import networkx as nx
from datetime import datetime
import imageio
from enum import Enum

class ThoughtState(Enum):
    EXPLORING = "exploring"
    LEARNING = "learning"
    PROCESSING = "processing"
    REMEMBERING = "remembering"
    DECIDING = "deciding"

@dataclass
class RatThought:
    """Representa um pensamento do rato"""
    timestamp: float
    brain_state: Dict[str, np.ndarray]
    neural_state: np.ndarray
    gene_activity: List[float]
    internal_state: Dict[str, float]
    thought_state: ThoughtState
    active_memories: Dict[str, float]
    focus: str



class IntegratedVirtualRat:
    """Rato virtual com integração completa dos sistemas"""
    def __init__(self, position: Tuple[float, float], output_dir: str = "rat_simulation"):
        self.position = np.array(position)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.brain_regions = ['cortex', 'hippocampus', 'amygdala', 'thalamus', 'striatum']
        
        # Inicializa os diferentes sistemas
        self.memory_network = MemoryNetwork(1000)
        self.neural_assembly = ComplexNeuralAssembly(100)
        self.gene_network = DenseGeneNetwork()
        self.brain_network = EnhancedNeuralNetwork(100)
        
        # Inicializa visualizadores
        self.memory_viz = MemoryVisualizer(self.memory_network, str(self.output_dir / "memory"))
        self.interaction_viz = ComplexInteractionVisualizer(self.neural_assembly)
        self.gene_viz = GeneNetworkVisualizer(self.gene_network, str(self.output_dir / "genes"))
        self.brain_viz = EnhancedVisualizer(self.brain_network)
        
        # Estado interno
        self.current_state = {
            'energy': 1.0,
            'stress': 0.0,
            'learning': 0.0,
            'memory_consolidation': 0.0
        }
        self.thoughts_dir = self.output_dir / "thoughts"
        self.thoughts_dir.mkdir(exist_ok=True)
        self.thought_history = []
        self.time = 0.0
        self.current_thought_state = ThoughtState.EXPLORING

    def update_internal_state(self, brain_activation, neural_state, gene_changes):
        """Atualiza estado interno baseado nas atividades dos sistemas"""
        # Calculate mean brain activation
        mean_brain_activation = np.mean(brain_activation)
        
        self.current_state['energy'] -= 0.01 * mean_brain_activation
        self.current_state['stress'] = 0.8 * self.current_state['stress'] + \
                                     0.2 * np.max(neural_state)
        self.current_state['learning'] = np.mean(gene_changes) if gene_changes else 0
        self.current_state['memory_consolidation'] += \
            0.1 * self.current_state['learning']
    
    def record_thought(self, state: Dict):
        """Registra um pensamento do rato"""
        thought = RatThought(
            timestamp=self.time,
            brain_state=state['brain_state'],
            neural_state=state['neural_state'],
            gene_activity=state['gene_activity'],
            internal_state=state['internal_state'],
            thought_state=self.current_thought_state,
            active_memories={
                k: float(np.mean(v)) for k, v in self.memory_network.memories.items()
            },
            focus=self.determine_focus(state)
        )
        
        self.thought_history.append(thought)
        self._visualize_thought(thought, len(self.thought_history)-1)
        
    def determine_focus(self, state: Dict) -> str:
        """Determina o foco atual do rato"""
        # Calculate mean brain activity across all regions
        brain_activity = np.mean([np.mean(activity) for activity in state['brain_state'].values()])
        
        if state['internal_state']['energy'] < 0.3:
            return "searching_food"
        elif state['internal_state']['stress'] > 0.7:
            return "avoiding_threat"
        elif brain_activity > 0.7:
            return "intense_learning"
        else:
            return "exploring"
            
    def _visualize_thought(self, thought: RatThought, index: int):
        """Visualiza o pensamento atual"""
        plt.figure(figsize=(20, 15))
        
        # Atividade Cerebral
        plt.subplot(331)
        self._plot_brain_activity(thought.brain_state)
        plt.title('Brain Activity')
        
        # Estado Neural
        plt.subplot(332)
        plt.imshow(thought.neural_state.reshape(10, 10), cmap='viridis')
        plt.title('Neural State')
        
        # Atividade Gênica
        plt.subplot(333)
        if thought.gene_activity:
            plt.bar(range(len(thought.gene_activity)), thought.gene_activity)
        plt.title('Gene Activity')
        
        # Estado Interno
        plt.subplot(334)
        plt.bar(thought.internal_state.keys(), thought.internal_state.values())
        plt.xticks(rotation=45)
        plt.title('Internal State')
        
        # Estado do Pensamento
        plt.subplot(335)
        plt.text(0.5, 0.5, f"Thought State:\n{thought.thought_state.value}",
                ha='center', va='center')
        plt.axis('off')
        plt.title('Current Thought State')
        
        # Memórias Ativas
        plt.subplot(336)
        if thought.active_memories:
            plt.bar(range(len(thought.active_memories)), 
                   list(thought.active_memories.values()))
        plt.title('Active Memories')
        
        # Foco Atual
        plt.subplot(337)
        plt.text(0.5, 0.5, f"Focus:\n{thought.focus}", ha='center', va='center')
        plt.axis('off')
        plt.title('Current Focus')
        
        plt.tight_layout()
        plt.savefig(self.thoughts_dir / f'thought_{index:04d}.png')
        plt.close()
        
    def _plot_brain_activity(self, brain_state: Dict[str, np.ndarray]):
        """Visualiza atividade cerebral como rede"""
        G = nx.Graph()
        
        # Agora brain_state é garantidamente um dicionário
        for region, activity in brain_state.items():
            activity_mean = float(np.mean(activity))
            G.add_node(region, activity=activity_mean)
            
        # Adiciona conexões
        regions = list(brain_state.keys())
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                G.add_edge(regions[i], regions[j])
                
        # Desenha o grafo
        pos = nx.spring_layout(G)
        nx.draw(G, pos,
                node_color=[G.nodes[n]['activity'] for n in G.nodes()],
                cmap='hot',
                node_size=2000,
                with_labels=True)
                
    def create_thought_animation(self):
        """Cria animação dos pensamentos"""
        images = []
        for i in range(len(self.thought_history)):
            image_path = self.thoughts_dir / f'thought_{i:04d}.png'
            if image_path.exists():
                images.append(imageio.imread(str(image_path)))
                
        if images:
            imageio.mimsave(
                str(self.thoughts_dir / 'thought_animation.gif'),
                images,
                duration=0.5
            )
            
    def save_thought_history(self):
        """Salva histórico de pensamentos"""
        history = []
        for thought in self.thought_history:
            thought_dict = {
                'timestamp': thought.timestamp,
                'thought_state': thought.thought_state.value,
                'focus': thought.focus,
                'internal_state': thought.internal_state
            }
            history.append(thought_dict)
            
        with open(self.thoughts_dir / 'thought_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
    def update(self, sensory_input: np.ndarray, dt: float):
        """Atualiza estado completo do rato"""
        self.time += dt
        
        # Processa entrada sensorial no cérebro
        brain_activation = self.brain_network.simulate_step(dt, sensory_input)
        
        # Convert brain activation to a dictionary with regions
        brain_state = {
            region: brain_activation[i:i+20] 
            for i, region in enumerate(self.brain_regions)
        }
        
        # Atualiza assembleias neurais
        neural_state = self.neural_assembly.update(brain_activation, dt)
        
        # Armazena na memória se relevante
        if np.max(neural_state) > 0.5:
            self.memory_network.store_memory(
                f"memory_{len(self.memory_network.memories)}",
                neural_state
            )
        
        # Ativa genes baseado na atividade neural
        gene_changes = []
        if np.mean(brain_activation) > 0.3:
            gene_changes.append(
                self.gene_network.activate_pathway('Activity_Dependent_Plasticity', 0.5)
            )
        
        # Cria estado atual
        state = {
            'brain_state': brain_state,
            'neural_state': neural_state,
            'gene_activity': gene_changes,
            'internal_state': self.current_state.copy()
        }
        
        # Atualiza estado interno
        self.update_internal_state(brain_activation, neural_state, gene_changes)
        
        # Atualiza estado do pensamento
        self.update_thought_state(state)
        
        # Registra pensamento
        self.record_thought(state)
        
        return state
    
    def update_thought_state(self, state: Dict):
        """Atualiza estado do pensamento baseado no estado atual"""
        # Calculate mean brain activity across all regions
        brain_activity = np.mean([np.mean(activity) for activity in state['brain_state'].values()])
        
        if brain_activity > 0.7:
            self.current_thought_state = ThoughtState.LEARNING
        elif state['internal_state']['memory_consolidation'] > 0.5:
            self.current_thought_state = ThoughtState.PROCESSING
        elif len(self.memory_network.memories) > 0 and np.random.random() < 0.2:
            self.current_thought_state = ThoughtState.REMEMBERING
        elif state['internal_state']['stress'] > 0.5:
            self.current_thought_state = ThoughtState.DECIDING
        else:
            self.current_thought_state = ThoughtState.EXPLORING
            
    def visualize_state(self, timestep: int):
        """Visualiza estado atual de todos os sistemas"""
        # Cria diretório para o timestep
        timestep_dir = self.output_dir / f"timestep_{timestep}"
        timestep_dir.mkdir(exist_ok=True)
        
        # Visualiza cada sistema
        if len(self.memory_network.memories) > 0:
            latest_memory = list(self.memory_network.memories.keys())[-1]
            self.memory_viz.visualize_memory(latest_memory, timestep)
            
        self.interaction_viz.visualize_state(timestep)
        self.gene_viz.create_network_visualization(timestep)
        self.brain_viz.visualize_state(timestep)
        
        # Visualiza estado interno
        self._plot_internal_state(timestep)
        
    def _plot_internal_state(self, timestep: int):
        """Plota estado interno do rato"""
        plt.figure(figsize=(10, 6))
        plt.bar(self.current_state.keys(), self.current_state.values())
        plt.title('Internal State')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'internal_state_{timestep}.png')
        plt.close()

def run_simulation(steps: int = 200):
    """Executa simulação completa"""
    # Inicializa rato
    rat = IntegratedVirtualRat((0, 0))
    
    # Parâmetros de simulação
    dt = 0.001
    
    # Histórico
    history = []
    
    # Loop principal
    for step in range(steps):
        # Gera entrada sensorial (pode ser modificada para entrada real do ambiente)
        sensory_input = np.random.normal(0, 1, 100)
        
        # Atualiza rato
        state = rat.update(sensory_input, dt)
        history.append(state)
        
        # Visualiza periodicamente
        if step % 100 == 0:
            rat.visualize_state(step)
            logger.info(f"Completed step {step}/{steps}")
            
        # Consolida memórias periodicamente
        if step % 500 == 0:
            rat.memory_network.consolidate_memories()

    rat.create_thought_animation()
    rat.save_thought_history()
    
    return history

def analyze_simulation(history: List[Dict]):
    """Analisa resultados da simulação"""
    # Extrai métricas
    brain_activity = [np.mean(state['brain_state']) for state in history]
    neural_activity = [np.mean(state['neural_state']) for state in history]
    internal_states = {
        key: [state['internal_state'][key] for state in history]
        for key in history[0]['internal_state']
    }
    
    # Plota resultados
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.plot(brain_activity)
    plt.title('Brain Activity')
    
    plt.subplot(222)
    plt.plot(neural_activity)
    plt.title('Neural Assembly Activity')
    
    plt.subplot(223)
    for key, values in internal_states.items():
        plt.plot(values, label=key)
    plt.title('Internal States')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_analysis.png')
    plt.close()

if __name__ == "__main__":
    logger.info("Starting virtual rat simulation")
    history = run_simulation()
    analyze_simulation(history)
    logger.info("Simulation completed")