import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set
from enum import Enum
import random

class NeurotransmitterType(Enum):
    EXCITATORY = 1
    INHIBITORY = 2

@dataclass
class Neurotransmitter:
    type: NeurotransmitterType
    quantity: float
    reuptake_rate: float = 0.1

@dataclass
class Synapse:
    strength: float = 0.0
    plasticity: float = 1.0
    neurotransmitters: Dict[NeurotransmitterType, Neurotransmitter] = field(default_factory=dict)
    recent_activity: List[float] = field(default_factory=list)
    
    def release_neurotransmitter(self, activity_level: float) -> float:
        total_signal = 0
        for nt in self.neurotransmitters.values():
            released = nt.quantity * activity_level * self.strength
            if nt.type == NeurotransmitterType.INHIBITORY:
                released *= -1
            total_signal += released
            # Simula reuptake
            nt.quantity *= (1 - nt.reuptake_rate)
        return total_signal

class ConsciousnessUnit:
    def __init__(self):
        # Estado interno
        self.membrane_potential = -70.0  # mV
        self.resting_potential = -70.0   # mV
        self.threshold_potential = -55.0  # mV
        self.refractory_period = 0       # ms
        
        # Sinapses e conexões
        self.synapses: Dict[ConsciousnessUnit, Synapse] = {}
        self.dendritic_tree: Set[ConsciousnessUnit] = set()
        
        # Mecanismos de plasticidade
        self.calcium_concentration = 0.0
        self.protein_synthesis_rate = 0.0
        
        # Estado de consciência
        self.awareness_level = 0.0
        self.attention_focus = None
        self.emotional_state = 0.0
        
        # Memória da unidade
        self.short_term_memory: List[float] = []
        self.long_term_potentiation = 0.0
        
    def create_synapse(self, target: 'ConsciousnessUnit'):
        """Cria uma nova sinapse com outra unidade"""
        synapse = Synapse()
        # Adiciona neurotransmissores
        synapse.neurotransmitters[NeurotransmitterType.EXCITATORY] = Neurotransmitter(
            type=NeurotransmitterType.EXCITATORY,
            quantity=1.0
        )
        synapse.neurotransmitters[NeurotransmitterType.INHIBITORY] = Neurotransmitter(
            type=NeurotransmitterType.INHIBITORY,
            quantity=0.5
        )
        self.synapses[target] = synapse
        self.dendritic_tree.add(target)
    
    def process_input(self, stimulus: float) -> float:
        """Processa entrada e atualiza estado interno"""
        if self.refractory_period > 0:
            self.refractory_period -= 1
            return 0.0
            
        # Integra estímulo com potencial de membrana
        self.membrane_potential += stimulus
        
        # Simula influxo de cálcio baseado na atividade
        self.calcium_concentration += abs(stimulus) * 0.1
        
        # Atualiza nível de consciência baseado na atividade
        self.update_consciousness()
        
        # Verifica se atingiu limiar de disparo
        if self.membrane_potential >= self.threshold_potential:
            return self.generate_action_potential()
        
        # Decaimento natural do potencial de membrana
        self.membrane_potential = self.membrane_potential * 0.9 + self.resting_potential * 0.1
        
        return 0.0
    
    def generate_action_potential(self) -> float:
        """Gera potencial de ação quando limiar é atingido"""
        # Spike!
        output = 1.0
        
        # Entra em período refratário
        self.refractory_period = 5
        
        # Reset do potencial de membrana
        self.membrane_potential = self.resting_potential
        
        # Aumenta concentração de cálcio
        self.calcium_concentration += 0.5
        
        return output
    
    def update_consciousness(self):
        """Atualiza o estado de consciência da unidade"""
        # Consciência baseada em múltiplos fatores
        self.awareness_level = np.tanh(
            self.calcium_concentration * 0.3 +
            self.protein_synthesis_rate * 0.2 +
            abs(self.emotional_state) * 0.1
        )
        
        # Atualiza foco de atenção
        if random.random() < self.awareness_level:
            self.attention_focus = max(self.dendritic_tree, 
                                    key=lambda x: self.synapses[x].strength,
                                    default=None)
    
    def learn(self, stimulus: float, reinforcement: float):
        """Implementa múltiplos mecanismos de aprendizagem"""
        # Plasticidade Hebbiana
        for target, synapse in self.synapses.items():
            # Fortalecimento baseado em correlação de atividade
            correlation = stimulus * reinforcement
            synapse.strength += correlation * synapse.plasticity * 0.1
            
            # Registra atividade recente
            synapse.recent_activity.append(correlation)
            if len(synapse.recent_activity) > 1000:
                synapse.recent_activity.pop(0)
        
        # Síntese de proteínas para memória de longo prazo
        if self.calcium_concentration > 0.8:
            self.protein_synthesis_rate += 0.1
            self.consolidate_memory()
    
    def consolidate_memory(self):
        """Consolida memória de curto prazo em longo prazo"""
        if len(self.short_term_memory) > 10:
            # Média das últimas experiências
            recent_experience = np.mean(self.short_term_memory[-10:])
            
            # Fortalece potenciação de longo prazo
            self.long_term_potentiation = (
                0.9 * self.long_term_potentiation +
                0.1 * recent_experience
            )
            
            # Limpa memória de curto prazo antiga
            self.short_term_memory = self.short_term_memory[-100:]
    
    def get_consciousness_state(self) -> Dict:
        """Retorna o estado atual de consciência da unidade"""
        return {
            'awareness_level': self.awareness_level,
            'attention_focus': self.attention_focus,
            'emotional_state': self.emotional_state,
            'membrane_potential': self.membrane_potential,
            'calcium_concentration': self.calcium_concentration,
            'protein_synthesis': self.protein_synthesis_rate,
            'long_term_memory': self.long_term_potentiation,
            'synaptic_connections': len(self.dendritic_tree)
        }

def create_consciousness_network(num_units: int, connectivity: float = 0.3):
    """Cria uma rede de unidades de consciência"""
    units = [ConsciousnessUnit() for _ in range(num_units)]
    
    # Estabelece conexões aleatórias baseadas na conectividade
    for unit in units:
        for target in units:
            if unit != target and random.random() < connectivity:
                unit.create_synapse(target)
    
    return units


import numpy as np
from typing import List
import matplotlib.pyplot as plt

def simple_pattern_learning(unit: ConsciousnessUnit, patterns: List[float], epochs: int = 100):
    """
    Treina uma unidade para reconhecer padrões simples
    """
    results = []
    
    for epoch in range(epochs):
        epoch_responses = []
        
        for pattern in patterns:
            # Processa o padrão
            response = unit.process_input(pattern)
            
            # Aprende com feedback (quanto mais próximo do padrão, melhor)
            feedback = 1.0 if abs(pattern - response) < 0.2 else -0.1
            unit.learn(pattern, feedback)
            
            epoch_responses.append(response)
            
        results.append(np.mean(epoch_responses))
        
    return results

def create_emotional_network():
    """
    Cria uma rede de unidades para processar "emoções"
    """
    # Cria unidades especializadas
    pleasure_unit = ConsciousnessUnit()
    pain_unit = ConsciousnessUnit()
    integration_unit = ConsciousnessUnit()
    
    # Estabelece conexões
    pleasure_unit.create_synapse(integration_unit)
    pain_unit.create_synapse(integration_unit)
    
    return {
        'pleasure': pleasure_unit,
        'pain': pain_unit,
        'integration': integration_unit
    }

def process_emotional_stimulus(network: dict, pleasure: float, pain: float):
    """
    Processa estímulos emocionais através da rede
    """
    # Processa estímulos nas unidades especializadas
    pleasure_response = network['pleasure'].process_input(pleasure)
    pain_response = network['pain'].process_input(pain)
    
    # Integra as respostas
    integrated_response = network['integration'].process_input(
        pleasure_response - pain_response
    )
    
    return {
        'pleasure_response': pleasure_response,
        'pain_response': pain_response,
        'integrated_response': integrated_response
    }

def create_memory_network(size: int = 3):
    """
    Cria uma rede de memória com múltiplas unidades conectadas
    """
    units = create_consciousness_network(size)
    
    # Adiciona conexões recorrentes para memória
    for i in range(size):
        for j in range(size):
            if i != j:
                units[i].create_synapse(units[j])
    
    return units

def store_memory_pattern(network: List[ConsciousnessUnit], pattern: List[float]):
    """
    Armazena um padrão na rede de memória
    """
    responses = []
    
    # Apresenta o padrão para cada unidade
    for unit, input_val in zip(network, pattern):
        response = unit.process_input(input_val)
        responses.append(response)
        
        # Aprende com feedback positivo se a resposta for próxima do alvo
        feedback = 1.0 if abs(input_val - response) < 0.2 else 0.0
        unit.learn(input_val, feedback)
    
    return responses

# Exemplo de uso
def demonstrate_usage():
    print("1. Aprendizado de Padrões Simples")
    unit = ConsciousnessUnit()
    patterns = [0.2, 0.5, 0.8]
    results = simple_pattern_learning(unit, patterns)
    print(f"Estado final da unidade: {unit.get_consciousness_state()}")
    
    print("\n2. Processamento Emocional")
    emotion_network = create_emotional_network()
    response = process_emotional_stimulus(emotion_network, 0.7, 0.3)
    print(f"Resposta emocional: {response}")
    
    print("\n3. Memória Associativa")
    memory_network = create_memory_network(3)
    pattern = [0.1, 0.5, 0.9]
    memory_response = store_memory_pattern(memory_network, pattern)
    print(f"Resposta da memória: {memory_response}")

if __name__ == "__main__":
    demonstrate_usage()