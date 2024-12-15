import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import logging
from scipy.sparse import csr_matrix
from enum import Enum
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='complex_interactions.log'
)
logger = logging.getLogger(__name__)

class InteractionType(Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    ASSOCIATIVE = "associative"

@dataclass
class SynapticConnection:
    pre_id: int
    post_id: int
    weight: float = 0.0
    type: InteractionType = InteractionType.EXCITATORY
    last_spike_time: float = 0.0
    stdp_window: List[float] = None
    trace: float = 0.0
    
    def __post_init__(self):
        self.stdp_window = []

@dataclass
class NeuromodulatorState:
    dopamine: float = 1.0
    serotonin: float = 1.0
    acetylcholine: float = 1.0
    norepinephrine: float = 1.0
    
    def update(self, reward: float, attention: float, arousal: float):
        """Atualiza níveis dos neuromoduladores baseado em sinais"""
        self.dopamine = np.clip(self.dopamine + 0.1 * reward, 0, 2)
        self.norepinephrine = np.clip(self.norepinephrine + 0.1 * arousal, 0, 2)
        self.acetylcholine = np.clip(self.acetylcholine + 0.1 * attention, 0, 2)

class ComplexNeuralAssembly:
    def __init__(self, size: int):
        self.size = size
        self.neurons = np.zeros(size)
        self.connections: Dict[Tuple[int, int], SynapticConnection] = {}
        self.neuromodulators = NeuromodulatorState()
        self.calcium_levels = np.zeros(size)
        self.protein_synthesis = np.zeros(size)
        self.activation_history = []
        
        # Inicializa conexões com probabilidade esparsa
        self._initialize_connections()
        
    def _initialize_connections(self):
        """Inicializa conexões com padrão complexo"""
        for i in range(self.size):
            for j in range(self.size):
                if i != j and np.random.random() < 0.1:  # 10% de probabilidade de conexão
                    connection_type = np.random.choice(
                        [InteractionType.EXCITATORY, InteractionType.INHIBITORY],
                        p=[0.8, 0.2]  # 80% excitatórias, 20% inibitórias
                    )
                    self.connections[(i, j)] = SynapticConnection(
                        pre_id=i,
                        post_id=j,
                        weight=np.random.normal(0.5, 0.1),
                        type=connection_type
                    )

    def apply_stdp(self, pre_id: int, post_id: int, time_diff: float):
        """Aplica STDP (Spike-Timing-Dependent Plasticity)"""
        if (pre_id, post_id) in self.connections:
            conn = self.connections[(pre_id, post_id)]
            
            # Função de STDP assimétrica
            if time_diff > 0:  # Post depois de pre (potenciação)
                dw = 0.1 * np.exp(-time_diff / 20.0)
            else:  # Pre depois de post (depressão)
                dw = -0.1 * np.exp(time_diff / 20.0)
                
            # Modula STDP baseado em neuromoduladores
            dw *= self.neuromodulators.dopamine
            
            conn.weight = np.clip(conn.weight + dw, 0, 1)
            conn.stdp_window.append(time_diff)

    def lateral_inhibition(self, active_neurons: Set[int]):
        """Aplica inibição lateral"""
        inhibition = np.zeros(self.size)
        for i in active_neurons:
            for j in range(self.size):
                if (i, j) in self.connections and \
                   self.connections[(i, j)].type == InteractionType.INHIBITORY:
                    inhibition[j] += self.connections[(i, j)].weight
        return inhibition

    def calcium_dynamics(self, active_neurons: Set[int]):
        """Simula dinâmica de cálcio intracelular"""
        # Aumenta cálcio em neurônios ativos
        self.calcium_levels[list(active_neurons)] += 0.2
        
        # Decaimento natural do cálcio
        self.calcium_levels *= 0.95
        
        # Ativa síntese de proteínas baseado em níveis de cálcio
        self.protein_synthesis = np.where(
            self.calcium_levels > 0.5,
            self.calcium_levels * 0.1,
            self.protein_synthesis * 0.95
        )

    def homeostatic_plasticity(self):
        """Aplica plasticidade homeostática"""
        mean_activity = np.mean([len(act) for act in self.activation_history[-100:]])
        target_activity = self.size * 0.1  # 10% de atividade alvo
        
        # Ajusta pesos para manter atividade próxima ao alvo
        scale_factor = target_activity / (mean_activity + 1e-10)
        for conn in self.connections.values():
            if conn.type == InteractionType.EXCITATORY:
                conn.weight *= np.sqrt(scale_factor)

    def update(self, input_pattern: np.ndarray, time: float):
        """Atualiza estado da assembleia"""
        # Processa entrada
        activation = np.zeros(self.size)
        active_neurons = set(np.where(input_pattern > 0.5)[0])
        
        # Propaga ativação através das conexões
        for i in active_neurons:
            for j in range(self.size):
                if (i, j) in self.connections:
                    conn = self.connections[(i, j)]
                    if conn.type == InteractionType.EXCITATORY:
                        activation[j] += conn.weight
                    elif conn.type == InteractionType.INHIBITORY:
                        activation[j] -= conn.weight
        
        # Aplica inibição lateral
        inhibition = self.lateral_inhibition(active_neurons)
        activation -= inhibition
        
        # Aplica função de ativação
        activation = F.relu(torch.tensor(activation)).numpy()
        
        # Atualiza STDP
        new_active = set(np.where(activation > 0.5)[0])
        for pre in active_neurons:
            for post in new_active:
                self.apply_stdp(pre, post, time)
        
        # Atualiza dinâmica de cálcio
        self.calcium_dynamics(new_active)
        
        # Registra ativação
        self.activation_history.append(new_active)
        
        # Periodicamente aplica plasticidade homeostática
        if len(self.activation_history) % 100 == 0:
            self.homeostatic_plasticity()
        
        return activation

class ComplexInteractionVisualizer:
    def __init__(self, assembly: ComplexNeuralAssembly):
        self.assembly = assembly
        
    def visualize_state(self, timestep: int):
        """Visualiza estado atual da assembleia"""
        plt.figure(figsize=(20, 10))
        
        # Matriz de conectividade
        plt.subplot(231)
        self._plot_connectivity_matrix()
        plt.title('Connectivity Matrix')
        
        # Grafo de conexões
        plt.subplot(232)
        self._plot_connection_graph()
        plt.title('Connection Graph')
        
        # Níveis de cálcio
        plt.subplot(233)
        plt.plot(self.assembly.calcium_levels)
        plt.title('Calcium Levels')
        
        # Distribuição de pesos
        plt.subplot(234)
        weights = [conn.weight for conn in self.assembly.connections.values()]
        plt.hist(weights, bins=20)
        plt.title('Weight Distribution')
        
        # Neuromoduladores
        plt.subplot(235)
        self._plot_neuromodulators()
        plt.title('Neuromodulator Levels')
        
        # Atividade recente
        plt.subplot(236)
        self._plot_recent_activity()
        plt.title('Recent Activity')
        
        plt.tight_layout()
        plt.savefig(f'complex_interactions_{timestep}.png')
        plt.close()
        
    def _plot_connectivity_matrix(self):
        matrix = np.zeros((self.assembly.size, self.assembly.size))
        for (i, j), conn in self.assembly.connections.items():
            matrix[i, j] = conn.weight if conn.type == InteractionType.EXCITATORY else -conn.weight
        plt.imshow(matrix, cmap='RdBu_r')
        plt.colorbar()
        
    def _plot_connection_graph(self):
        G = nx.DiGraph()
        for (i, j), conn in self.assembly.connections.items():
            G.add_edge(i, j, weight=conn.weight, 
                      color='g' if conn.type == InteractionType.EXCITATORY else 'r')
        
        pos = nx.spring_layout(G)
        edge_colors = [G[u][v]['color'] for u,v in G.edges()]
        nx.draw(G, pos, edge_color=edge_colors, with_labels=True, node_size=100)
        
    def _plot_neuromodulators(self):
        levels = [
            self.assembly.neuromodulators.dopamine,
            self.assembly.neuromodulators.serotonin,
            self.assembly.neuromodulators.acetylcholine,
            self.assembly.neuromodulators.norepinephrine
        ]
        plt.bar(['DA', '5-HT', 'ACh', 'NE'], levels)
        
    def _plot_recent_activity(self):
        recent = self.assembly.activation_history[-100:]
        activity = [len(act) for act in recent]
        plt.plot(activity)

def simulate_complex_interactions():
    """Simula interações complexas ao longo do tempo"""
    assembly = ComplexNeuralAssembly(100)  # 100 neurônios
    visualizer = ComplexInteractionVisualizer(assembly)
    
    # Simula padrões de entrada variados
    for t in range(10000):
        # Gera padrão de entrada
        if t % 100 < 50:  # Alterna entre padrões
            input_pattern = np.zeros(100)
            input_pattern[20:40] = 1  # Padrão A
        else:
            input_pattern = np.zeros(100)
            input_pattern[60:80] = 1  # Padrão B
            
        # Adiciona ruído
        input_pattern += np.random.normal(0, 0.1, 100)
        
        # Atualiza assembleia
        assembly.update(input_pattern, t)
        
        # Atualiza neuromoduladores baseado em "recompensa" simulada
        assembly.neuromodulators.update(
            reward=np.sin(t/100),  # Recompensa oscilante
            attention=0.5 + 0.5*np.sin(t/50),  # Atenção variável
            arousal=0.5 + 0.3*np.cos(t/75)  # Arousal variável
        )
        
        # Visualiza periodicamente
        if t % 100 == 0:
            visualizer.visualize_state(t)
            logger.info(f"Completed timestep {t}")

if __name__ == "__main__":
    simulate_complex_interactions()