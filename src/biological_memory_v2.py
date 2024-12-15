import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import logging
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='memory_storage.log'
)
logger = logging.getLogger(__name__)

@dataclass
class Synapse:
    """Representa uma sinapse entre neurônios"""
    pre_neuron: int
    post_neuron: int
    weight: float = 0.1
    neurotransmitter_level: float = 1.0
    receptor_density: float = 1.0
    spine_size: float = 1.0
    
    def apply_ltp(self, strength: float):
        """Aplica Potenciação de Longa Duração"""
        self.weight *= (1 + strength)
        self.receptor_density *= (1 + strength * 0.5)
        self.spine_size *= (1 + strength * 0.3)
        
    def apply_ltd(self, strength: float):
        """Aplica Depressão de Longa Duração"""
        self.weight *= (1 - strength)
        self.receptor_density *= (1 - strength * 0.5)
        self.spine_size *= (1 - strength * 0.3)

class Neuron:
    def __init__(self, id: int):
        self.id = id
        self.dendrites: List[float] = []  # Comprimentos dos dendritos
        self.spine_count = 0
        self.myelination_level = 0.1
        self.protein_synthesis = 0.0
        self.gene_expression: Dict[str, float] = {
            'BDNF': 0.0,    # Fator neurotrófico
            'CREB': 0.0,    # Regulador da transcrição
            'ARC': 0.0,     # Plasticidade sináptica
            'FOS': 0.0      # Resposta imediata
        }
        
    def grow_dendrite(self, length: float):
        """Adiciona novo dendrito"""
        self.dendrites.append(length)
        
    def add_spine(self):
        """Adiciona espinha dendrítica"""
        self.spine_count += 1
        
    def increase_myelination(self, amount: float):
        """Aumenta nível de mielinização"""
        self.myelination_level = min(1.0, self.myelination_level + amount)

class MemoryNetwork:
    def __init__(self, num_neurons: int):
        self.neurons = [Neuron(i) for i in range(num_neurons)]
        self.synapses: Dict[Tuple[int, int], Synapse] = {}
        self.assemblies: Dict[str, Set[int]] = {}  # Grupos neurais por memória
        self.memories: Dict[str, np.ndarray] = {}  # Padrões de ativação
        
    def create_synapse(self, pre: int, post: int):
        """Cria nova sinapse entre neurônios"""
        if (pre, post) not in self.synapses:
            self.synapses[(pre, post)] = Synapse(pre, post)
            
    def strengthen_synapse(self, pre: int, post: int, strength: float):
        """Fortalece sinapse existente"""
        if (pre, post) in self.synapses:
            self.synapses[(pre, post)].apply_ltp(strength)
            
    def weaken_synapse(self, pre: int, post: int, strength: float):
        """Enfraquece sinapse existente"""
        if (pre, post) in self.synapses:
            self.synapses[(pre, post)].apply_ltd(strength)
            
    def store_memory(self, memory_id: str, pattern: np.ndarray):
        """Armazena novo padrão de memória"""
        # Cria assembleia neural para a memória
        assembly = set()
        active_neurons = np.where(pattern > 0.5)[0]
        
        for neuron_id in active_neurons:
            assembly.add(neuron_id)
            # Crescimento estrutural
            self.neurons[neuron_id].grow_dendrite(1.0)
            self.neurons[neuron_id].add_spine()
            
            # Ativa genes relacionados à memória
            self.neurons[neuron_id].gene_expression['BDNF'] += 0.2
            self.neurons[neuron_id].gene_expression['CREB'] += 0.3
            self.neurons[neuron_id].gene_expression['ARC'] += 0.4
            self.neurons[neuron_id].gene_expression['FOS'] += 0.5
            
        # Cria e fortalece sinapses entre neurônios ativos
        for i in active_neurons:
            for j in active_neurons:
                if i != j:
                    self.create_synapse(i, j)
                    self.strengthen_synapse(i, j, 0.5)
                    
        self.assemblies[memory_id] = assembly
        self.memories[memory_id] = pattern
        
        logger.info(f"Stored memory {memory_id} involving {len(assembly)} neurons")
        
    def retrieve_memory(self, memory_id: str, partial_pattern: Optional[np.ndarray] = None) -> np.ndarray:
        """Recupera padrão de memória armazenado"""
        if memory_id not in self.memories:
            return None
            
        assembly = self.assemblies[memory_id]
        pattern = np.zeros_like(self.memories[memory_id])
        
        # Ativa assembleia neural
        for neuron_id in assembly:
            pattern[neuron_id] = 1.0
            
            # Ativa genes relacionados à recuperação
            self.neurons[neuron_id].gene_expression['ARC'] += 0.2
            self.neurons[neuron_id].gene_expression['FOS'] += 0.3
            
        # Se fornecido padrão parcial, usa para refinamento
        if partial_pattern is not None:
            pattern = 0.7 * pattern + 0.3 * partial_pattern
            
        return pattern
        
    def consolidate_memories(self):
        """Consolida memórias através de mudanças estruturais"""
        for memory_id, assembly in self.assemblies.items():
            for neuron_id in assembly:
                # Aumenta mielinização
                self.neurons[neuron_id].increase_myelination(0.1)
                
                # Síntese de proteínas
                self.neurons[neuron_id].protein_synthesis += 0.2
                
            # Fortalece sinapses dentro da assembleia
            for pre in assembly:
                for post in assembly:
                    if pre != post and (pre, post) in self.synapses:
                        self.strengthen_synapse(pre, post, 0.2)
                        
        logger.info("Consolidated memories through structural changes")
        
    def save_state(self, filename: str):
        """Salva estado da rede"""
        state = {
            'memories': self.memories,
            'assemblies': self.assemblies,
            'synapses': self.synapses
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
            
    def load_state(self, filename: str):
        """Carrega estado da rede"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            self.memories = state['memories']
            self.assemblies = state['assemblies']
            self.synapses = state['synapses']

class MemoryVisualizer:
    def __init__(self, network: MemoryNetwork, output_dir: str = "memory_viz"):
        self.network = network
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def visualize_memory(self, memory_id: str, timestep: int):
        """Visualiza estado de uma memória específica"""
        plt.figure(figsize=(15, 5))
        
        # Padrão de ativação
        plt.subplot(131)
        plt.imshow(self.network.memories[memory_id].reshape(-1, 1), cmap='viridis')
        plt.title(f'Memory Pattern {memory_id}')
        
        # Rede neural
        plt.subplot(132)
        G = nx.Graph()
        assembly = self.network.assemblies[memory_id]
        
        # Adiciona nós
        for neuron_id in assembly:
            G.add_node(neuron_id, 
                      size=len(self.network.neurons[neuron_id].dendrites),
                      myelination=self.network.neurons[neuron_id].myelination_level)
            
        # Adiciona arestas
        for pre in assembly:
            for post in assembly:
                if pre != post and (pre, post) in self.network.synapses:
                    G.add_edge(pre, post, 
                             weight=self.network.synapses[(pre, post)].weight)
                    
        pos = nx.spring_layout(G)
        nx.draw(G, pos,
                node_size=[G.nodes[n]['size']*100 for n in G.nodes()],
                node_color=[G.nodes[n]['myelination'] for n in G.nodes()],
                edge_color=[G[u][v]['weight'] for u,v in G.edges()],
                width=2,
                cmap='viridis')
        plt.title('Neural Assembly')
        
        # Expressão gênica
        plt.subplot(133)
        example_neuron = list(assembly)[0]
        gene_levels = list(self.network.neurons[example_neuron].gene_expression.values())
        gene_names = list(self.network.neurons[example_neuron].gene_expression.keys())
        plt.bar(gene_names, gene_levels)
        plt.title('Gene Expression')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'memory_{memory_id}_{timestep}.png')
        plt.close()

def simulate_memory_formation():
    """Simula formação e consolidação de memórias"""
    # Cria rede
    network = MemoryNetwork(1000)
    visualizer = MemoryVisualizer(network)
    
    # Cria padrões de memória
    memories = {
        'memory1': np.random.binomial(1, 0.1, 1000),
        'memory2': np.random.binomial(1, 0.1, 1000),
        'memory3': np.random.binomial(1, 0.1, 1000)
    }
    
    # Armazena e visualiza memórias
    for timestep, (memory_id, pattern) in enumerate(memories.items()):
        logger.info(f"Storing memory {memory_id}")
        network.store_memory(memory_id, pattern)
        visualizer.visualize_memory(memory_id, timestep)
        
        # Consolida periodicamente
        if timestep % 2 == 0:
            network.consolidate_memories()
            
    # Salva estado final
    network.save_state('final_memory_state.pkl')

if __name__ == "__main__":
    simulate_memory_formation()