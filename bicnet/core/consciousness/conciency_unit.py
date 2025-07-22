import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SynapseType(Enum):
    EXCITATORY = 1
    INHIBITORY = 2


@dataclass
class MemoryTrace:
    """
    Armazena traços de memória para plasticidade
    """
    short_term: Dict[str, float] = field(default_factory=dict)
    long_term: Dict[str, float] = field(default_factory=dict)
    emotional: Dict[str, float] = field(default_factory=dict)
    temporal: List[float] = field(default_factory=list)


@dataclass
class SynapticState:
    """
    Estado da sinapse incluindo neurotransmissores e plasticidade
    """
    weight: float = 0.0
    type: SynapseType = SynapseType.EXCITATORY
    neurotransmitter_level: float = 1.0
    plasticity_rate: float = 1.0
    last_spike_time: float = 0.0
    dendritic_distance: float = 0.0
    temporal_trace: List[float] = field(default_factory=list)


class ConsciousnessUnit:
    def __init__(
        self,
        input_size: int,
        membrane_rest: float = -70.0,  # mV 
        threshold: float = -55.0,      # mV
        refractory_period: int = 5,    # ms
        learning_rate: float = 0.01,
        plasticity_threshold: float = 0.7,
        homeostatic_rate: float = 0.1,
        emotional_learning_rate: float = 0.05
    ):
        # Parâmetros da membrana
        self.membrane_potential = membrane_rest
        self.membrane_rest = membrane_rest
        self.threshold = threshold
        self.refractory_period = refractory_period
        self.refractory_time = 0
        
        # Estado interno
        self.consciousness_level = 0.0
        self.attention_focus = None
        self.emotional_state = 0.0
        self.calcium_concentration = 0.0
        self.protein_synthesis = 0.0
        
        # Sinapses e conexões
        self.synapses: Dict[int, SynapticState] = {}
        self.dendritic_branches: Dict[int, List[int]] = {}
        
        # Memória
        self.memory = MemoryTrace()
        
        # Taxas de aprendizado
        self.learning_rate = learning_rate
        self.plasticity_threshold = plasticity_threshold
        self.homeostatic_rate = homeostatic_rate
        self.emotional_learning_rate = emotional_learning_rate
        
        # Inicializa sinapses
        self.initialize_synapses(input_size)

    def initialize_synapses(self, input_size: int):
        """Inicializa sinapses com distribuição lognormal"""
        # Distribuição lognormal dos pesos conforme observado biologicamente
        weights = np.random.lognormal(mean=-0.5, sigma=0.1, size=input_size)
        
        # Cria sinapses com tipos aleatórios e distribuição espacial
        for i in range(input_size):
            synapse_type = SynapseType.EXCITATORY if np.random.random() > 0.2 else SynapseType.INHIBITORY
            dendritic_distance = np.random.uniform(0, 1)
            
            self.synapses[i] = SynapticState(
                weight=weights[i],
                type=synapse_type,
                dendritic_distance=dendritic_distance
            )
            
        # Agrupa sinapses em ramos dendríticos
        self.organize_dendritic_branches()
        
    def organize_dendritic_branches(self, branch_size: int = 10):
        """Organiza sinapses em ramos dendríticos"""
        synapses = list(self.synapses.keys())
        np.random.shuffle(synapses)
        
        for i in range(0, len(synapses), branch_size):
            branch = synapses[i:i + branch_size]
            branch_id = i // branch_size
            self.dendritic_branches[branch_id] = branch
            
    def compute_dendritic_integration(self, inputs: np.ndarray) -> float:
        """Integração sináptica não-linear nos dendritos"""
        branch_activations = []
        
        # Processa cada ramo dendrítico separadamente
        for branch_id, synapses in self.dendritic_branches.items():
            # Soma ponderada das entradas no ramo
            branch_input = 0
            for syn_id in synapses:
                syn = self.synapses[syn_id]
                weight = syn.weight if syn.type == SynapseType.EXCITATORY else -syn.weight
                distance_decay = np.exp(-syn.dendritic_distance)
                branch_input += inputs[syn_id] * weight * distance_decay
            
            # Não-linearidade do ramo dendrítico (sigmoid)
            if branch_input > 0:
                branch_activation = 2 / (1 + np.exp(-branch_input)) - 1
            else:
                branch_activation = 0
                
            branch_activations.append(branch_activation)
            
        # Soma das ativações dos ramos
        return np.sum(branch_activations)
        
    def update_consciousness(self, activation: float):
        """Atualiza nível de consciência baseado na atividade"""
        # Consciência emerge da integração de múltiplos fatores
        self.consciousness_level = np.tanh(
            self.calcium_concentration * 0.3 +
            self.protein_synthesis * 0.2 +
            abs(self.emotional_state) * 0.1 +
            activation * 0.4
        )
        
        # Atualiza foco de atenção
        if np.random.random() < self.consciousness_level:
            active_synapses = [(i, s.weight) for i, s in self.synapses.items()]
            if active_synapses:
                self.attention_focus = max(active_synapses, key=lambda x: x[1])[0]
                
    def update_calcium_dynamics(self, activation: float):
        """Atualiza dinâmica do cálcio baseado na atividade"""
        # Influxo de cálcio proporcional à ativação
        calcium_influx = activation * 0.1
        
        # Decaimento do cálcio
        calcium_decay = self.calcium_concentration * 0.05
        
        self.calcium_concentration += calcium_influx - calcium_decay
        
        # Síntese de proteínas dependente de cálcio
        if self.calcium_concentration > 0.8:
            self.protein_synthesis += 0.1
            
    def hebbian_learning(self, inputs: np.ndarray, activation: float):
        """
        Aprendizado Hebbiano com múltiplos mecanismos de plasticidade
        """
        for i, synapse in self.synapses.items():
            # Plasticidade Hebbiana
            hebbian = inputs[i] * activation * self.learning_rate
            
            # Plasticidade Homeostática
            current_activity = np.mean(synapse.temporal_trace[-100:]) if synapse.temporal_trace else 0
            homeostatic = self.homeostatic_rate * (0.5 - current_activity)
            
            # Competição Heterosináptica
            if abs(synapse.weight) > self.plasticity_threshold:
                heterosynaptic = -0.1 * synapse.weight
            else:
                heterosynaptic = 0
                
            # Atualização total
            weight_update = hebbian + homeostatic + heterosynaptic
            
            # Atualiza peso
            synapse.weight += weight_update
            
            # Atualiza traço temporal
            synapse.temporal_trace.append(activation)
            if len(synapse.temporal_trace) > 1000:
                synapse.temporal_trace.pop(0)
                
    def consolidate_memory(self, importance: float):
        """
        Consolida memórias de curto prazo em longo prazo
        """
        if len(self.memory.temporal) > 10:
            recent_experience = np.mean(self.memory.temporal[-10:])
            
            # Consolida baseado na importância e estado emocional
            consolidation_strength = importance * (1 + abs(self.emotional_state))
            
            for key in self.memory.short_term:
                if key not in self.memory.long_term:
                    self.memory.long_term[key] = 0
                    
                # Transferência para memória de longo prazo
                self.memory.long_term[key] = (
                    0.9 * self.memory.long_term[key] +
                    0.1 * self.memory.short_term[key] * consolidation_strength
                )
                
            # Limpa memória de curto prazo antiga
            self.memory.temporal = self.memory.temporal[-100:]
            
    def process_input(self, inputs: np.ndarray, importance: float = 1.0) -> float:
        """
        Processa entrada e atualiza estado interno
        """
        if self.refractory_time > 0:
            self.refractory_time -= 1
            return 0.0
            
        # Integração dendrítica
        dendritic_input = self.compute_dendritic_integration(inputs)
        
        # Atualiza potencial de membrana
        self.membrane_potential += dendritic_input
        
        # Atualiza dinâmica de cálcio
        self.update_calcium_dynamics(dendritic_input)
        
        # Verifica disparo
        if self.membrane_potential >= self.threshold:
            # Gera spike
            self.membrane_potential = self.membrane_rest
            self.refractory_time = self.refractory_period
            activation = 1.0
            
            # Aumenta concentração de cálcio no spike
            self.calcium_concentration += 0.5
        else:
            # Decaimento do potencial
            self.membrane_potential = (
                self.membrane_potential * 0.9 + 
                self.membrane_rest * 0.1
            )
            activation = 0.0
            
        # Atualiza estado interno
        self.update_consciousness(activation)
        
        # Aprendizado
        self.hebbian_learning(inputs, activation)
        
        # Consolidação de memória
        self.consolidate_memory(importance)
        
        return activation
        
    def get_state(self) -> Dict:
        """
        Retorna estado atual da unidade
        """
        return {
            'consciousness_level': self.consciousness_level,
            'attention_focus': self.attention_focus,
            'emotional_state': self.emotional_state,
            'membrane_potential': self.membrane_potential,
            'calcium_concentration': self.calcium_concentration,
            'protein_synthesis': self.protein_synthesis,
            'synaptic_weights': {i: s.weight for i, s in self.synapses.items()}
        }

# Exemplo de uso
if __name__ == "__main__":
    # Cria unidade com 100 entradas
    unit = ConsciousnessUnit(input_size=100)
    
    # Processa algumas entradas aleatórias
    for _ in range(10):
        inputs = np.random.random(100)
        activation = unit.process_input(inputs)
        state = unit.get_state()
        print(f"Activation: {activation:.3f}")
        print(f"Consciousness: {state['consciousness_level']:.3f}")
        print(f"Attention focus: {state['attention_focus']}")
        print("---")
