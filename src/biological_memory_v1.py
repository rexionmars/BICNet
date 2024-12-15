from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from enum import Enum
import time

class MemoryType(Enum):
    SENSORY = "sensory"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

@dataclass
class Protein:
    """Representa proteínas sinápticas"""
    name: str
    concentration: float
    half_life: float
    synthesis_rate: float
    degradation_rate: float
    last_update: float = field(default_factory=time.time)
    
    def update(self):
        """Atualiza concentração baseado no tempo"""
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        # Degradação natural
        self.concentration *= np.exp(-self.degradation_rate * elapsed)
        # Síntese contínua
        self.concentration += self.synthesis_rate * elapsed
        
        self.last_update = current_time

@dataclass
class Synapse:
    """Representa uma sinapse com seus componentes moleculares"""
    strength: float = 0.0
    num_receptors: int = 100
    neurotransmitter_count: float = 1.0
    spine_size: float = 1.0
    proteins: Dict[str, Protein] = field(default_factory=dict)
    last_activation: float = 0.0
    ltp_level: float = 0.0  # Potenciação de Longa Duração
    ltd_level: float = 0.0  # Depressão de Longa Duração

@dataclass
class Dendrite:
    """Representa um dendrito"""
    synapses: Dict[int, Synapse] = field(default_factory=dict)
    spine_density: float = 1.0
    branch_complexity: float = 1.0
    myelination: float = 0.0

class NeuralAssembly:
    """Representa uma assembleia neural que codifica informação"""
    def __init__(self, size: int):
        self.neurons: Set[int] = set()
        self.activation_pattern: np.ndarray = np.zeros(size)
        self.connection_strength: Dict[Tuple[int, int], float] = {}
        self.activation_threshold: float = 0.5
        self.refractory_period: float = 0.1
        self.last_activation: float = 0.0

class BiologicalMemorySystem:
    def __init__(self, num_neurons: int = 1000):
        self.num_neurons = num_neurons
        self.dendrites: Dict[int, List[Dendrite]] = {}
        self.assemblies: Dict[str, NeuralAssembly] = {}
        self.proteins = self._initialize_proteins()
        self.epigenetic_marks: Dict[str, float] = {}
        self.gene_expression: Dict[str, float] = {}
        
        self._initialize_structure()
    
    def _initialize_proteins(self) -> Dict[str, Protein]:
        """Inicializa proteínas importantes para a memória"""
        return {
            "CREB": Protein("CREB", 1.0, 24.0, 0.1, 0.05),
            "BDNF": Protein("BDNF", 1.0, 12.0, 0.2, 0.1),
            "CaMKII": Protein("CaMKII", 1.0, 8.0, 0.3, 0.15),
            "Arc": Protein("Arc", 1.0, 4.0, 0.4, 0.2)
        }
    
    def _initialize_structure(self):
        """Inicializa estrutura neural básica"""
        for neuron_id in range(self.num_neurons):
            self.dendrites[neuron_id] = [
                Dendrite() for _ in range(np.random.poisson(10))  # Média de 10 dendritos por neurônio
            ]
    
    def store_information(self, info_key: str, data: np.ndarray, 
                         memory_type: MemoryType) -> bool:
        """Armazena nova informação no sistema"""
        # Cria nova assembleia neural
        assembly = NeuralAssembly(self.num_neurons)
        
        # Seleciona neurônios para codificar a informação
        selected_neurons = np.random.choice(
            self.num_neurons,
            size=int(self.num_neurons * 0.1),  # 10% dos neurônios
            replace=False
        )
        assembly.neurons.update(selected_neurons)
        
        # Cria padrão de ativação
        assembly.activation_pattern[selected_neurons] = data[:len(selected_neurons)]
        
        # Fortalece conexões entre neurônios selecionados
        for i in selected_neurons:
            for j in selected_neurons:
                if i != j:
                    assembly.connection_strength[(i, j)] = np.random.random()
        
        # Aplica mudanças sinápticas
        self._modify_synapses(assembly)
        
        # Ativa genes relacionados à memória
        self._activate_memory_genes(memory_type)
        
        # Armazena a assembleia
        self.assemblies[info_key] = assembly
        
        return True
    
    def _modify_synapses(self, assembly: NeuralAssembly):
        """Modifica sinapses para armazenar informação"""
        for neuron_id in assembly.neurons:
            for dendrite in self.dendrites[neuron_id]:
                # Cria novas sinapses
                new_synapse = Synapse()
                
                # Aumenta número de receptores
                new_synapse.num_receptors += int(np.random.poisson(50))
                
                # Aumenta liberação de neurotransmissores
                new_synapse.neurotransmitter_count *= 1.5
                
                # Aumenta tamanho da espinha dendrítica
                new_synapse.spine_size *= 1.3
                
                # Adiciona proteínas específicas
                for protein in self.proteins.values():
                    protein.update()
                    new_synapse.proteins[protein.name] = Protein(
                        protein.name,
                        protein.concentration * 1.2,
                        protein.half_life,
                        protein.synthesis_rate * 1.1,
                        protein.degradation_rate
                    )
                
                # Induz LTP
                new_synapse.ltp_level = 1.0
                
                # Adiciona sinapse ao dendrito
                synapse_id = len(dendrite.synapses)
                dendrite.synapses[synapse_id] = new_synapse
                
                # Aumenta complexidade dendrítica
                dendrite.branch_complexity *= 1.1
                dendrite.spine_density *= 1.2
    
    def _activate_memory_genes(self, memory_type: MemoryType):
        """Ativa genes relacionados à formação de memória"""
        # Genes específicos para cada tipo de memória
        memory_genes = {
            MemoryType.SENSORY: ["immediate_early_genes"],
            MemoryType.SHORT_TERM: ["arc", "c-fos"],
            MemoryType.LONG_TERM: ["creb", "bdnf", "zif268"],
            MemoryType.PROCEDURAL: ["dopamine_genes", "motor_genes"],
            MemoryType.EPISODIC: ["hippocampal_genes"],
            MemoryType.SEMANTIC: ["cortical_genes"]
        }
        
        # Ativa genes específicos
        for gene in memory_genes[memory_type]:
            self.gene_expression[gene] = 1.0
            
            # Adiciona marca epigenética
            self.epigenetic_marks[gene] = 1.0
    
    def retrieve_information(self, info_key: str) -> Optional[np.ndarray]:
        """Recupera informação armazenada"""
        if info_key not in self.assemblies:
            return None
            
        assembly = self.assemblies[info_key]
        
        # Verifica se proteínas necessárias estão presentes
        for neuron_id in assembly.neurons:
            for dendrite in self.dendrites[neuron_id]:
                for synapse in dendrite.synapses.values():
                    for protein in synapse.proteins.values():
                        protein.update()
                        if protein.concentration < 0.5:
                            return None  # Memória pode estar degradada
        
        # Ativa o padrão neural
        current_time = time.time()
        if (current_time - assembly.last_activation) > assembly.refractory_period:
            assembly.last_activation = current_time
            return assembly.activation_pattern
        
        return None
    
    def strengthen_memory(self, info_key: str):
        """Fortalece uma memória existente"""
        if info_key in self.assemblies:
            assembly = self.assemblies[info_key]
            
            # Aumenta força das conexões
            for connection in assembly.connection_strength:
                assembly.connection_strength[connection] *= 1.1
            
            # Aumenta síntese de proteínas
            for neuron_id in assembly.neurons:
                for dendrite in self.dendrites[neuron_id]:
                    for synapse in dendrite.synapses.values():
                        for protein in synapse.proteins.values():
                            protein.synthesis_rate *= 1.1
            
            # Aumenta marcas epigenéticas
            for mark in self.epigenetic_marks:
                self.epigenetic_marks[mark] *= 1.1
    
    def consolidate_memories(self):
        """Consolida memórias durante período de repouso"""
        for assembly in self.assemblies.values():
            # Fortalece conexões estáveis
            strong_connections = {
                conn: strength 
                for conn, strength in assembly.connection_strength.items()
                if strength > 0.7
            }
            
            for connection in strong_connections:
                assembly.connection_strength[connection] *= 1.2
            
            # Aumenta mielinização
            for neuron_id in assembly.neurons:
                for dendrite in self.dendrites[neuron_id]:
                    dendrite.myelination += 0.1
                    
            # Sintetiza proteínas de manutenção
            for protein in self.proteins.values():
                protein.synthesis_rate *= 1.1
    
    def get_memory_state(self, info_key: str) -> Dict:
        """Retorna estado detalhado de uma memória"""
        if info_key not in self.assemblies:
            return {}
            
        assembly = self.assemblies[info_key]
        
        state = {
            "num_neurons": len(assembly.neurons),
            "connection_strength": np.mean(list(assembly.connection_strength.values())),
            "protein_levels": {},
            "epigenetic_marks": dict(self.epigenetic_marks),
            "gene_expression": dict(self.gene_expression),
            "structural_state": {
                "dendritic_complexity": [],
                "spine_density": [],
                "myelination": []
            }
        }
        
        # Analisa estado das proteínas
        for neuron_id in assembly.neurons:
            for dendrite in self.dendrites[neuron_id]:
                state["structural_state"]["dendritic_complexity"].append(
                    dendrite.branch_complexity
                )
                state["structural_state"]["spine_density"].append(
                    dendrite.spine_density
                )
                state["structural_state"]["myelination"].append(
                    dendrite.myelination
                )
                
                for synapse in dendrite.synapses.values():
                    for protein_name, protein in synapse.proteins.items():
                        if protein_name not in state["protein_levels"]:
                            state["protein_levels"][protein_name] = []
                        state["protein_levels"][protein_name].append(
                            protein.concentration
                        )
        
        # Calcula médias
        for key in state["protein_levels"]:
            state["protein_levels"][key] = np.mean(state["protein_levels"][key])
        
        for key in state["structural_state"]:
            state["structural_state"][key] = np.mean(state["structural_state"][key])
        
        return state

# Exemplo de uso:
def create_memory_system(num_neurons: int = 1000) -> BiologicalMemorySystem:
    """Cria sistema de memória"""
    return BiologicalMemorySystem(num_neurons)

def store_and_retrieve_example():
    """Exemplo de armazenamento e recuperação"""
    system = create_memory_system()
    
    # Cria dado para armazenar
    data = np.random.random(100)
    
    # Armazena informação
    system.store_information(
        "exemplo_memoria",
        data,
        MemoryType.LONG_TERM
    )
    
    # Recupera informação
    retrieved = system.retrieve_information("exemplo_memoria")
    
    # Fortalece memória
    system.strengthen_memory("exemplo_memoria")
    
    # Consolida durante "sono"
    system.consolidate_memories()
    
    # Verifica estado
    state = system.get_memory_state("exemplo_memoria")
    
    return retrieved, state