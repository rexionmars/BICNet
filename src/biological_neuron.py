import numpy as np
from scipy import signal
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple

class BiologicalNeuron:
    def __init__(self, gene_expression: Dict[str, float] = None):
        """
        Simula um neurônio biológico com expressão gênica
        
        Parameters:
        gene_expression: dicionário com níveis de expressão de genes importantes
        """
        # Genes importantes para funcionamento neural
        self.gene_expression = gene_expression or {
            'SNAP25': 1.0,  # Liberação de neurotransmissores
            'SYN1': 1.0,    # Função sináptica
            'BDNF': 1.0,    # Plasticidade
            'GRIN1': 1.0,   # Receptores NMDA
            'CAMK2A': 1.0,  # Potenciação de longo prazo
        }
        
        # Estado do neurônio
        self.membrane_potential = -70.0  # mV
        self.threshold = -55.0          # mV
        self.refractory_period = 0      # ms
        
        # Sinapses
        self.synapses = []
        
        # Neurotransmissores disponíveis
        self.neurotransmitters = {
            'glutamate': 100.0,
            'GABA': 100.0,
        }

    def update_gene_expression(self, activity_level: float):
        """
        Atualiza expressão gênica baseado na atividade neural
        """
        # Plasticidade dependente de atividade
        self.gene_expression['BDNF'] *= (1.0 + 0.1 * activity_level)
        self.gene_expression['CAMK2A'] *= (1.0 + 0.05 * activity_level)
        
        # Homeostase sináptica
        if activity_level > 1.5:
            self.gene_expression['GRIN1'] *= 0.95  # Downregulation
        elif activity_level < 0.5:
            self.gene_expression['GRIN1'] *= 1.05  # Upregulation

    def simulate_action_potential(self, input_current: float) -> float:
        """
        Simula potencial de ação usando modelo Hodgkin-Huxley simplificado
        """
        if self.refractory_period > 0:
            self.refractory_period -= 1
            return 0.0
            
        # Influência da expressão gênica na excitabilidade
        threshold_mod = self.threshold * (1.0 - 0.1 * self.gene_expression['GRIN1'])
        
        self.membrane_potential += input_current * self.gene_expression['SNAP25']
        
        if self.membrane_potential >= threshold_mod:
            # Disparo
            spike = 1.0 * self.gene_expression['SYN1']
            self.membrane_potential = -70.0
            self.refractory_period = 5
            return spike
        
        # Decaimento natural do potencial
        self.membrane_potential += (-70.0 - self.membrane_potential) * 0.1
        return 0.0

class NeuralNetwork:
    def __init__(self, num_neurons: int):
        self.neurons = [BiologicalNeuron() for _ in range(num_neurons)]
        self.connectivity = nx.watts_strogatz_graph(num_neurons, 4, 0.1)
        
        # Matriz de pesos sinápticos
        self.weights = np.random.normal(0.5, 0.1, (num_neurons, num_neurons))
        
    def simulate_step(self, external_input: np.ndarray) -> np.ndarray:
        """
        Simula um passo temporal da rede
        """
        # Vetor de ativações
        activations = np.zeros(len(self.neurons))
        
        # Atualiza cada neurônio
        for i, neuron in enumerate(self.neurons):
            # Soma entradas sinápticas
            synaptic_input = 0.0
            for j in self.connectivity[i]:
                synaptic_input += self.weights[j,i] * activations[j]
            
            # Adiciona input externo
            total_input = synaptic_input + external_input[i]
            
            # Simula potencial de ação
            activations[i] = neuron.simulate_action_potential(total_input)
            
            # Atualiza expressão gênica baseado na atividade
            neuron.update_gene_expression(activations[i])
            
        return activations

def analyze_network_activity(network: NeuralNetwork, 
                           simulation_steps: int = 1000) -> Dict:
    """
    Analisa atividade da rede ao longo do tempo
    """
    activity_history = []
    gene_expression_history = []
    
    for step in range(simulation_steps):
        # Input aleatório para simular atividade espontânea
        external_input = np.random.normal(0, 0.1, len(network.neurons))
        
        # Simula um passo
        activations = network.simulate_step(external_input)
        
        # Registra atividade
        activity_history.append(np.mean(activations))
        
        # Registra expressão gênica média
        avg_expression = {gene: np.mean([n.gene_expression[gene] 
                         for n in network.neurons])
                         for gene in network.neurons[0].gene_expression}
        gene_expression_history.append(avg_expression)
    
    return {
        'activity': np.array(activity_history),
        'gene_expression': pd.DataFrame(gene_expression_history)
    }

# Exemplo de uso
def main():
    # Cria rede com 100 neurônios
    network = NeuralNetwork(100)
    
    # Simula e analisa
    results = analyze_network_activity(network)
    
    # Imprime alguns resultados
    print("Atividade média da rede:", np.mean(results['activity']))
    print("\nMudanças na expressão gênica:")
    print(results['gene_expression'].describe())

if __name__ == "__main__":
    main()