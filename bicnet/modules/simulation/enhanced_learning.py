import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Set, Optional
from enum import Enum
from dataclasses import dataclass, field
import networkx as nx
from complex_neural import (ComplexNeuralAssembly, InteractionType, 
                           SynapticConnection, NeuromodulatorState)

class LearningRule(Enum):
    """Tipos de regras de aprendizado disponíveis"""
    STDP = "stdp"                        # Spike-Timing-Dependent Plasticity (já implementada)
    BCM = "bcm"                          # Bienenstock-Cooper-Munro
    REINFORCEMENT = "reinforcement"      # Aprendizado por Reforço
    HEBBIAN = "hebbian"                  # Aprendizado Hebbiano Clássico
    OJA = "oja"                          # Regra de Oja (componentes principais)
    COMPETITIVE = "competitive"          # Aprendizado Competitivo
    CONTRASTIVE = "contrastive"          # Divergência Contrastiva

@dataclass
class LearningParameters:
    """Parâmetros para as regras de aprendizado"""
    # Parâmetros comuns
    learning_rate: float = 0.01
    
    # Parâmetros específicos de BCM
    bcm_threshold: float = 0.5
    bcm_time_constant: float = 100.0
    
    # Parâmetros de aprendizado por reforço
    eligibility_decay: float = 0.95
    reward_discount: float = 0.9
    
    # Parâmetros de aprendizado competitivo
    winner_strength: float = 1.2
    inhibition_strength: float = 0.8
    
    # Parâmetros de divergência contrastiva
    cd_steps: int = 5
    
    # Metaplasticidade
    metaplasticity_rate: float = 0.001
    metaplasticity_threshold: float = 0.5
    
    # Ativação de grupos neurais
    assembly_activation_threshold: float = 0.3


class EnhancedLearningAssembly(ComplexNeuralAssembly):
    """Versão ampliada da assembleia neural com algoritmos de aprendizado avançados"""
    
    def __init__(self, size: int):
        super().__init__(size)
        
        # Configuração de aprendizado
        self.learning_parameters = LearningParameters()
        self.active_learning_rules = {
            LearningRule.STDP: True,       # já implementado
            LearningRule.BCM: False,
            LearningRule.REINFORCEMENT: False,
            LearningRule.HEBBIAN: False,
            LearningRule.OJA: False,
            LearningRule.COMPETITIVE: False,
            LearningRule.CONTRASTIVE: False
        }
        
        # Estado para aprendizado por reforço
        self.eligibility_traces = {}
        self.reward_history = []
        self.value_estimate = 0.0
        
        # Estado para BCM
        self.activity_averages = np.zeros(size)
        
        # Estado para aprendizado competitivo
        self.competition_winners = set()
        
        # Estado para divergência contrastiva
        self.positive_phase = np.zeros(size)
        self.negative_phase = np.zeros(size)
        
        # Grupos de neurônios (assembleias funcionais)
        self.neural_assemblies = []
        
        # Estado para metaplasticidade
        self.plasticity_thresholds = np.ones(size) * self.learning_parameters.metaplasticity_threshold
    
    def set_learning_rule(self, rule: LearningRule, active: bool):
        """Ativa ou desativa uma regra de aprendizado específica"""
        self.active_learning_rules[rule] = active
    
    def update_bcm(self, active_neurons: Set[int]):
        """Implementa a regra BCM (Bienenstock-Cooper-Munro)"""
        # Atualiza médias de atividade (representando o limiar variável)
        activity = np.zeros(self.size)
        activity[list(active_neurons)] = 1.0
        
        # Atualiza médias de atividade com fator de decaimento
        tau = self.learning_parameters.bcm_time_constant
        self.activity_averages = (1 - 1/tau) * self.activity_averages + (1/tau) * activity**2
        
        # Aplica regra BCM para todas as conexões
        for (pre_id, post_id), conn in self.connections.items():
            if conn.type == InteractionType.EXCITATORY and post_id in active_neurons:
                # Obtém atividade pré e pós sináptica
                pre_activity = 1.0 if pre_id in active_neurons else 0.0
                post_activity = 1.0  # Se post_id está em active_neurons
                
                # Calcula mudança de peso baseada na regra BCM
                threshold = self.activity_averages[post_id]
                dw = pre_activity * post_activity * (post_activity - threshold)
                
                # Modula com neuromoduladores
                dw *= self.neuromodulators.acetylcholine
                
                # Aplica mudança com rate de aprendizado
                conn.weight += self.learning_parameters.learning_rate * dw
                conn.weight = np.clip(conn.weight, 0, 1)
    
    def update_reinforcement_learning(self, reward: float, active_neurons: Set[int]):
        """Implementa aprendizado por reforço"""
        # Registra recompensa
        self.reward_history.append(reward)
        
        # Calcula TD-error
        prev_value = self.value_estimate
        self.value_estimate = reward + self.learning_parameters.reward_discount * self.value_estimate
        td_error = self.value_estimate - prev_value
        
        # Atualiza traços de elegibilidade para conexões ativas
        for pre_id in active_neurons:
            for post_id in active_neurons:
                key = (pre_id, post_id)
                if key in self.connections:
                    # Cria traço se não existir
                    if key not in self.eligibility_traces:
                        self.eligibility_traces[key] = 0.0
                    
                    # Aumenta traço para a conexão ativa
                    self.eligibility_traces[key] += 1.0
        
        # Decai todos os traços
        for key in list(self.eligibility_traces.keys()):
            self.eligibility_traces[key] *= self.learning_parameters.eligibility_decay
            
            # Remove traços pequenos para eficiência
            if self.eligibility_traces[key] < 0.01:
                del self.eligibility_traces[key]
        
        # Atualiza pesos baseado em TD-error e traços de elegibilidade
        for (pre_id, post_id), trace in self.eligibility_traces.items():
            if (pre_id, post_id) in self.connections:
                conn = self.connections[(pre_id, post_id)]
                
                # Modula com dopamina para aprendizado por reforço
                dw = td_error * trace * self.neuromodulators.dopamine
                conn.weight += self.learning_parameters.learning_rate * dw
                conn.weight = np.clip(conn.weight, 0, 1)
    
    def update_hebbian(self, active_neurons: Set[int]):
        """Implementa regra Hebbiana clássica"""
        for pre_id in active_neurons:
            for post_id in active_neurons:
                if (pre_id, post_id) in self.connections:
                    conn = self.connections[(pre_id, post_id)]
                    
                    # Aplica regra Hebbiana simples
                    dw = self.learning_parameters.learning_rate
                    
                    # Modula com acetilcolina para aprendizado associativo
                    dw *= self.neuromodulators.acetylcholine
                    
                    conn.weight += dw
                    conn.weight = np.clip(conn.weight, 0, 1)
    
    def update_oja(self, active_neurons: Set[int]):
        """Implementa regra de Oja (aproximação para PCA)"""
        for post_id in range(self.size):
            # Calcula ativação com soma ponderada
            post_activation = 0.0
            for pre_id in active_neurons:
                if (pre_id, post_id) in self.connections:
                    post_activation += self.connections[(pre_id, post_id)].weight
            
            # Normaliza ativação
            post_activation = 1.0 / (1.0 + np.exp(-post_activation))  # Sigmoide
            
            # Atualiza pesos para todas as conexões de entrada
            for pre_id in range(self.size):
                if (pre_id, post_id) in self.connections:
                    conn = self.connections[(pre_id, post_id)]
                    
                    # Regra de Oja: combina Hebbian com normalização
                    pre_activation = 1.0 if pre_id in active_neurons else 0.0
                    dw = post_activation * (pre_activation - post_activation * conn.weight)
                    
                    # Aplica mudança
                    conn.weight += self.learning_parameters.learning_rate * dw
                    conn.weight = np.clip(conn.weight, 0, 1)
    
    def update_competitive(self, active_neurons: Set[int]):
        """Implementa aprendizado competitivo"""
        # Calcula ativação para todos os neurônios
        activations = np.zeros(self.size)
        
        # Propaga ativação de neurônios ativos
        for pre_id in active_neurons:
            for post_id in range(self.size):
                if (pre_id, post_id) in self.connections:
                    conn = self.connections[(pre_id, post_id)]
                    if conn.type == InteractionType.EXCITATORY:
                        activations[post_id] += conn.weight
        
        # Encontra os vencedores (top 10% das ativações)
        if np.sum(activations) > 0:  # Evita divisão por zero
            n_winners = max(1, int(0.1 * self.size))
            winner_indices = np.argsort(activations)[-n_winners:]
            self.competition_winners = set(winner_indices)
            
            # Atualiza pesos apenas para os vencedores (fortalece)
            for post_id in winner_indices:
                for pre_id in active_neurons:
                    if (pre_id, post_id) in self.connections:
                        conn = self.connections[(pre_id, post_id)]
                        dw = self.learning_parameters.learning_rate * self.learning_parameters.winner_strength
                        conn.weight += dw
                        conn.weight = np.clip(conn.weight, 0, 1)
            
            # Enfraquece conexões para neurônios não-vencedores
            for post_id in range(self.size):
                if post_id not in winner_indices:
                    for pre_id in active_neurons:
                        if (pre_id, post_id) in self.connections:
                            conn = self.connections[(pre_id, post_id)]
                            dw = -self.learning_parameters.learning_rate * self.learning_parameters.inhibition_strength
                            conn.weight += dw
                            conn.weight = np.clip(conn.weight, 0, 1)
    
    def update_contrastive_divergence(self, input_pattern: np.ndarray):
        """Implementa Divergência Contrastiva"""
        # Fase positiva: aplica o padrão de entrada
        self.positive_phase = np.zeros(self.size)
        self.positive_phase[input_pattern > 0.5] = 1.0
        
        # Fase negativa: realiza etapas de amostragem de Gibbs
        self.negative_phase = self.positive_phase.copy()
        
        for _ in range(self.learning_parameters.cd_steps):
            # Calcula ativações para cada neurônio
            activations = np.zeros(self.size)
            
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) in self.connections and self.negative_phase[i] > 0:
                        activations[j] += self.connections[(i, j)].weight
            
            # Aplica função de ativação sigmoide
            probabilities = 1.0 / (1.0 + np.exp(-activations))
            
            # Amostra nova fase negativa
            self.negative_phase = (np.random.random(self.size) < probabilities).astype(float)
        
        # Atualiza pesos baseado na diferença entre fases
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in self.connections:
                    conn = self.connections[(i, j)]
                    
                    # Diferença entre correlações nas fases positiva e negativa
                    pos_correlation = self.positive_phase[i] * self.positive_phase[j]
                    neg_correlation = self.negative_phase[i] * self.negative_phase[j]
                    
                    dw = self.learning_parameters.learning_rate * (pos_correlation - neg_correlation)
                    conn.weight += dw
                    conn.weight = np.clip(conn.weight, 0, 1)
    
    def update_metaplasticity(self, active_neurons: Set[int]):
        """Implementa metaplasticidade"""
        # Atualiza limiares de plasticidade baseado na atividade
        activity = np.zeros(self.size)
        activity[list(active_neurons)] = 1.0
        
        # Limiares se adaptam à atividade: aumentam com alta atividade
        delta_threshold = self.learning_parameters.metaplasticity_rate * (activity - self.plasticity_thresholds)
        self.plasticity_thresholds += delta_threshold
        
        # Limita limiares em intervalos razoáveis
        self.plasticity_thresholds = np.clip(self.plasticity_thresholds, 0.1, 0.9)
    
    def detect_neural_assemblies(self):
        """Detecta grupos de neurônios fortemente conectados (assembleias)"""
        # Cria matriz de adjacência ponderada
        adj_matrix = np.zeros((self.size, self.size))
        for (i, j), conn in self.connections.items():
            adj_matrix[i, j] = conn.weight
        
        # Usa clustering baseado em modularidade para detectar comunidades
        try:
            import networkx as nx
            
            G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            
            # Remove arestas com pesos abaixo do limiar
            threshold = self.learning_parameters.assembly_activation_threshold
            edges_to_remove = [(i, j) for i, j, w in G.edges(data='weight') if w < threshold]
            G.remove_edges_from(edges_to_remove)
            
            # Detecta comunidades
            try:
                from networkx.algorithms import community
                communities = community.greedy_modularity_communities(G)
                self.neural_assemblies = [set(c) for c in communities]
                return self.neural_assemblies
            except:
                # Fallback se o algoritmo falhar (ex: grafo vazio)
                return []
        except Exception as e:
            print(f"Error detecting neural assemblies: {e}")
            return []
    
    def update(self, input_pattern: np.ndarray, time: float, reward: float = 0.0):
        """Versão ampliada do método update incorporando algoritmos avançados"""
        # Primeiro chama o update original para processamento básico
        activation = super().update(input_pattern, time)
        
        # Determina neurônios ativos
        active_neurons = set(np.where(activation > 0.5)[0])
        
        # Aplica algoritmos de aprendizado adicionais se ativados
        if self.active_learning_rules[LearningRule.BCM]:
            self.update_bcm(active_neurons)
            
        if self.active_learning_rules[LearningRule.REINFORCEMENT]:
            self.update_reinforcement_learning(reward, active_neurons)
            
        if self.active_learning_rules[LearningRule.HEBBIAN]:
            self.update_hebbian(active_neurons)
            
        if self.active_learning_rules[LearningRule.OJA]:
            self.update_oja(active_neurons)
            
        if self.active_learning_rules[LearningRule.COMPETITIVE]:
            self.update_competitive(active_neurons)
            
        if self.active_learning_rules[LearningRule.CONTRASTIVE]:
            self.update_contrastive_divergence(input_pattern)
        
        # Atualiza metaplasticidade
        self.update_metaplasticity(active_neurons)
        
        # Periodicamente detecta assembleias neurais
        if time % 100 == 0:
            self.detect_neural_assemblies()
        
        return activation


# Função auxiliar para testes
def demonstrate_advanced_learning():
    """Demonstra o uso dos algoritmos de aprendizado avançados"""
    import matplotlib.pyplot as plt
    
    # Cria assembleia com aprendizado ampliado
    assembly = EnhancedLearningAssembly(100)
    
    # Ativa múltiplos mecanismos de aprendizado
    assembly.set_learning_rule(LearningRule.BCM, True)
    assembly.set_learning_rule(LearningRule.REINFORCEMENT, True)
    assembly.set_learning_rule(LearningRule.COMPETITIVE, True)
    
    # Padrões de entrada para demonstração
    patterns = {
        'A': np.zeros(100),
        'B': np.zeros(100),
        'C': np.zeros(100)
    }
    patterns['A'][10:30] = 1
    patterns['B'][40:60] = 1
    patterns['C'][70:90] = 1
    
    # Função de recompensa baseada em padrão
    def get_reward(pattern_name, time):
        if pattern_name == 'A':
            return 0.5 + 0.5 * np.sin(time / 200)
        elif pattern_name == 'B':
            return 1.0 - 0.5 * np.sin(time / 200)
        else:
            return 0.1
    
    # Simula aprendizado
    for t in range(5000):
        # Seleciona padrão baseado no tempo
        if t % 200 < 100:
            pattern_name = 'A'
        elif t % 200 < 150:
            pattern_name = 'B'
        else:
            pattern_name = 'C'
        
        pattern = patterns[pattern_name].copy()
        
        # Adiciona ruído
        pattern += np.random.normal(0, 0.1, 100)
        
        # Calcula recompensa baseada no padrão e tempo
        reward = get_reward(pattern_name, t)
        
        # Atualiza assembleia
        assembly.update(pattern, t, reward)
        
        # Periodicamente mostra assembleias detectadas
        if t % 500 == 0:
            assemblies = assembly.detect_neural_assemblies()
            print(f"Timestep {t}, Detected {len(assemblies)} neural assemblies")
            for i, neurons in enumerate(assemblies):
                print(f"Assembly {i}: {len(neurons)} neurons")
    
    # Verifica se o aprendizado ocorreu visualizando a matriz de pesos
    plt.figure(figsize=(12, 10))
    
    # Visualiza matriz de pesos
    plt.subplot(221)
    weights_matrix = np.zeros((assembly.size, assembly.size))
    for (i, j), conn in assembly.connections.items():
        weights_matrix[i, j] = conn.weight
    plt.imshow(weights_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Weight Matrix After Learning')
    
    # Visualiza traços de elegibilidade
    plt.subplot(222)
    eligibility_matrix = np.zeros((assembly.size, assembly.size))
    for (i, j), trace in assembly.eligibility_traces.items():
        eligibility_matrix[i, j] = trace
    plt.imshow(eligibility_matrix, cmap='plasma')
    plt.colorbar()
    plt.title('Eligibility Traces')
    
    # Visualiza histórico de recompensa
    plt.subplot(223)
    plt.plot(assembly.reward_history)
    plt.title('Reward History')
    
    # Visualiza assembleias detectadas
    plt.subplot(224)
    assembly_matrix = np.zeros((assembly.size, assembly.size))
    for i, assembly_neurons in enumerate(assembly.neural_assemblies):
        for neuron in assembly_neurons:
            assembly_matrix[neuron, :] = i + 1
    plt.imshow(assembly_matrix, cmap='tab10')
    plt.title('Detected Neural Assemblies')
    
    plt.tight_layout()
    plt.savefig('advanced_learning_results.png')
    plt.show()


if __name__ == "__main__":
    demonstrate_advanced_learning()