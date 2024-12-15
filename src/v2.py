import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math

class ConsciousnessLevel(Enum):
    UNCONSCIOUS = 0
    SUBCONSCIOUS = 1
    CONSCIOUS = 2
    SELF_AWARE = 3

@dataclass
class Memory:
    """Estrutura de memória"""
    short_term: Dict[str, float] = field(default_factory=dict)
    long_term: Dict[str, float] = field(default_factory=dict)
    working: Dict[str, float] = field(default_factory=dict)
    emotional: Dict[str, float] = field(default_factory=dict)
    temporal_trace: List[float] = field(default_factory=list)

@dataclass
class Synapse:
    """Estado sináptico"""
    weight: float = 0.0
    plasticity: float = 1.0
    neurotransmitter: float = 1.0
    last_activation: float = 0.0
    temporal_trace: List[float] = field(default_factory=list)

class ConsciousnessSystem:
    def __init__(
        self,
        input_size: int,
        membrane_rest: float = -70.0,
        threshold: float = -55.0,
        refractory_period: int = 5,
        learning_rate: float = 0.01,
        consciousness_threshold: float = 0.7,
        emotional_learning_rate: float = 0.05
    ):
        # Parâmetros da membrana
        self.membrane_potential = membrane_rest
        self.membrane_rest = membrane_rest
        self.threshold = threshold
        self.refractory_period = refractory_period
        self.refractory_time = 0

        # Estado de consciência
        self.consciousness_level = 0.0
        self.attention_focus = None
        self.emotional_state = 0.0
        self.arousal_level = 0.5

        # Mecanismos biológicos
        self.calcium_concentration = 0.0
        self.protein_synthesis = 0.0
        self.neurotransmitter_levels = {}
        self.dendrite_activation = {}

        # Aprendizado
        self.learning_rate = learning_rate
        self.consciousness_threshold = consciousness_threshold
        self.emotional_learning_rate = emotional_learning_rate

        # Memória e sinapses
        self.memory = Memory()
        self.synapses: Dict[int, Synapse] = {}
        self.initialize_synapses(input_size)

        # Estado interno
        self.internal_state = {
            'arousal': 0.5,
            'attention': None,
            'emotion': 0.0,
            'stress': 0.0,
            'fatigue': 0.0
        }

    def initialize_synapses(self, input_size: int):
        """Inicializa sinapses com distribuição lognormal"""
        weights = np.random.lognormal(mean=-0.5, sigma=0.1, size=input_size)
        
        for i in range(input_size):
            self.synapses[i] = Synapse(
                weight=weights[i],
                plasticity=np.random.random(),
                neurotransmitter=1.0
            )

    def update_consciousness_level(self, inputs: np.ndarray) -> ConsciousnessLevel:
        """Atualiza e retorna nível de consciência"""
        # Fatores que influenciam a consciência
        input_strength = np.mean(np.abs(inputs))
        calcium_factor = self.calcium_concentration * 0.3
        arousal_factor = self.internal_state['arousal'] * 0.2
        emotion_factor = abs(self.emotional_state) * 0.1

        # Calcula nível de consciência
        consciousness = np.tanh(
            input_strength +
            calcium_factor +
            arousal_factor +
            emotion_factor +
            self.protein_synthesis * 0.1
        )

        self.consciousness_level = consciousness

        # Determina nível discreto
        if consciousness < 0.3:
            return ConsciousnessLevel.UNCONSCIOUS
        elif consciousness < 0.5:
            return ConsciousnessLevel.SUBCONSCIOUS
        elif consciousness < 0.8:
            return ConsciousnessLevel.CONSCIOUS
        else:
            return ConsciousnessLevel.SELF_AWARE

    def process_dendrites(self, inputs: np.ndarray) -> float:
        """Processa integração dendrítica"""
        total_activation = 0.0
        dendrite_groups = self.group_synapses_by_dendrite()

        for dendrite_id, synapse_group in dendrite_groups.items():
            # Soma ponderada das entradas no dendrito
            dendrite_input = 0.0
            for syn_id in synapse_group:
                synapse = self.synapses[syn_id]
                dendrite_input += inputs[syn_id] * synapse.weight * synapse.neurotransmitter

            # Não-linearidade dendrítica
            if dendrite_input > 0:
                dendrite_activation = 2.0 / (1.0 + np.exp(-dendrite_input)) - 1.0
            else:
                dendrite_activation = 0.0

            self.dendrite_activation[dendrite_id] = dendrite_activation
            total_activation += dendrite_activation

        return total_activation

    def update_synaptic_plasticity(self, inputs: np.ndarray, output: float):
        """Atualiza plasticidade sináptica"""
        for i, synapse in self.synapses.items():
            # Plasticidade Hebbiana
            hebbian = inputs[i] * output * self.learning_rate
            
            # Plasticidade Homeostática
            recent_activity = np.mean(synapse.temporal_trace[-100:]) if synapse.temporal_trace else 0
            homeostatic = self.learning_rate * (0.5 - recent_activity)
            
            # Competição Heterosináptica
            if abs(synapse.weight) > self.consciousness_threshold:
                heterosynaptic = -0.1 * synapse.weight
            else:
                heterosynaptic = 0

            # Plasticidade emocional
            emotional = self.emotional_state * self.emotional_learning_rate
            
            # Atualização total
            synapse.weight += hebbian + homeostatic + heterosynaptic + emotional
            
            # Atualiza traço temporal
            synapse.temporal_trace.append(output)
            if len(synapse.temporal_trace) > 1000:
                synapse.temporal_trace.pop(0)

    def update_neurotransmitters(self):
        """Atualiza níveis de neurotransmissores"""
        for synapse in self.synapses.values():
            # Síntese de neurotransmissores
            synthesis = 0.1 * (1.0 - synapse.neurotransmitter)
            
            # Degradação
            degradation = 0.05 * synapse.neurotransmitter
            
            # Atualização
            synapse.neurotransmitter += synthesis - degradation

    def update_internal_state(self, inputs: np.ndarray):
        """Atualiza estado interno"""
        # Atualiza arousal
        input_strength = np.mean(np.abs(inputs))
        self.internal_state['arousal'] = 0.9 * self.internal_state['arousal'] + 0.1 * input_strength

        # Atualiza fadiga
        self.internal_state['fatigue'] += 0.001
        if self.internal_state['arousal'] > 0.8:
            self.internal_state['fatigue'] += 0.002

        # Recuperação durante baixa atividade
        if self.internal_state['arousal'] < 0.2:
            self.internal_state['fatigue'] *= 0.95

        # Atualiza stress
        self.internal_state['stress'] = 0.8 * self.internal_state['stress'] + \
                                      0.2 * (self.internal_state['arousal'] + self.internal_state['fatigue'])

    def consolidate_memory(self):
        """Consolida memória de curto prazo em longo prazo"""
        if len(self.memory.temporal_trace) > 10:
            recent_experience = np.mean(self.memory.temporal_trace[-10:])
            
            # Fator de consolidação baseado em consciência e emoção
            consolidation_strength = self.consciousness_level * (1 + abs(self.emotional_state))
            
            for key in self.memory.short_term:
                if key not in self.memory.long_term:
                    self.memory.long_term[key] = 0
                    
                # Transferência para memória de longo prazo
                self.memory.long_term[key] = (
                    0.9 * self.memory.long_term[key] +
                    0.1 * self.memory.short_term[key] * consolidation_strength
                )
            
            # Limpa memória antiga
            self.memory.temporal_trace = self.memory.temporal_trace[-100:]

    def process_input(self, inputs: np.ndarray) -> Tuple[float, ConsciousnessLevel]:
        """Processa entrada e retorna ativação e nível de consciência"""
        if self.refractory_time > 0:
            self.refractory_time -= 1
            return 0.0, ConsciousnessLevel.UNCONSCIOUS

        # Integração dendrítica
        dendritic_activation = self.process_dendrites(inputs)
        
        # Atualiza potencial de membrana
        self.membrane_potential += dendritic_activation
        
        # Atualiza dinâmica de cálcio
        self.calcium_concentration += abs(dendritic_activation) * 0.1
        self.calcium_concentration *= 0.95  # Decay
        
        # Gera spike se atingir threshold
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = self.membrane_rest
            self.refractory_time = self.refractory_period
            output = 1.0
            self.calcium_concentration += 0.5
        else:
            # Decaimento do potencial
            self.membrane_potential = (
                self.membrane_potential * 0.9 +
                self.membrane_rest * 0.1
            )
            output = 0.0

        # Atualiza estados
        self.update_internal_state(inputs)
        consciousness_level = self.update_consciousness_level(inputs)
        self.update_synaptic_plasticity(inputs, output)
        self.update_neurotransmitters()
        self.consolidate_memory()

        return output, consciousness_level

    def get_state(self) -> Dict:
        """Retorna estado completo do sistema"""
        return {
            'consciousness_level': self.consciousness_level,
            'attention_focus': self.attention_focus,
            'emotional_state': self.emotional_state,
            'membrane_potential': self.membrane_potential,
            'calcium_concentration': self.calcium_concentration,
            'protein_synthesis': self.protein_synthesis,
            'internal_state': self.internal_state,
            'memory_state': {
                'short_term_size': len(self.memory.short_term),
                'long_term_size': len(self.memory.long_term),
                'working_memory_size': len(self.memory.working)
            }
        }

    def group_synapses_by_dendrite(self, branch_size: int = 10) -> Dict[int, List[int]]:
        """Agrupa sinapses em ramos dendríticos"""
        dendrites = {}
        synapses = list(self.synapses.keys())
        np.random.shuffle(synapses)
        
        for i in range(0, len(synapses), branch_size):
            dendrite_id = i // branch_size
            dendrites[dendrite_id] = synapses[i:i + branch_size]
            
        return dendrites