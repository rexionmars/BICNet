import numpy as np
from typing import List, Tuple

class NeuralLearningSimulation:
    def __init__(self, input_size: int, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        # Simulando sinapses com pesos
        self.weights = np.random.randn(input_size) * 0.01
        # Simulando plasticidade com histórico de ativações
        self.activation_history = []
        # Limiar para potenciação de longo prazo
        self.ltp_threshold = 0.7
        
    def hebbian_learning(self, input_data: np.ndarray, target: float) -> float:
        """
        Implementa regra de Hebb: "Neurons that fire together, wire together"
        Simula plasticidade sináptica
        """
        activation = np.dot(input_data, self.weights)
        error = target - activation
        
        # Atualização dos pesos baseada na correlação entre entrada e erro
        delta_weights = self.learning_rate * error * input_data
        self.weights += delta_weights
        
        # Registra histórico de ativações para simular plasticidade
        self.activation_history.append(activation)
        
        return activation
    
    def long_term_potentiation(self) -> None:
        """
        Simula potenciação de longo prazo (LTP)
        Fortalece conexões frequentemente ativadas
        """
        if len(self.activation_history) > 10:
            recent_activations = self.activation_history[-10:]
            mean_activation = np.mean(recent_activations)
            
            if mean_activation > self.ltp_threshold:
                # Fortalece pesos que contribuem para ativações fortes
                self.weights *= 1.1
    
    def synaptic_pruning(self, threshold: float = 0.1) -> None:
        """
        Simula poda sináptica
        Remove conexões fracas ou pouco utilizadas
        """
        weak_connections = np.abs(self.weights) < threshold
        self.weights[weak_connections] = 0
    
    def homeostatic_plasticity(self) -> None:
        """
        Simula plasticidade homeostática
        Mantém estabilidade da rede ajustando pesos globalmente
        """
        if len(self.activation_history) > 100:
            mean_activity = np.mean(self.activation_history[-100:])
            target_activity = 0.5
            
            # Ajusta todos os pesos para manter atividade média desejada
            scale_factor = target_activity / (mean_activity + 1e-10)
            self.weights *= scale_factor

def demonstrate_learning():
    # Exemplo de uso
    neural_sim = NeuralLearningSimulation(input_size=4)
    
    # Dados de treino simulando padrões de entrada
    training_data = [
        (np.array([1, 0, 1, 0]), 1),  # Padrão A -> Saída 1
        (np.array([0, 1, 0, 1]), 0),  # Padrão B -> Saída 0
    ]
    
    # Ciclo de aprendizagem
    for epoch in range(100):
        for input_pattern, target in training_data:
            # Aprendizado hebbiano
            activation = neural_sim.hebbian_learning(input_pattern, target)
            
            # Processos de plasticidade
            if epoch % 10 == 0:
                neural_sim.long_term_potentiation()
                neural_sim.homeostatic_plasticity()
            
            if epoch % 20 == 0:
                neural_sim.synaptic_pruning()
    
    return neural_sim

# Executa demonstração
model = demonstrate_learning()