import numpy as np
from typing import List, Dict
from enum import Enum

from .conciency_unit import ConsciousnessUnit

class CorticalLayerType(Enum):
    SENSORY = 1  # Camadas sensoriais
    PROCESSING = 2  # Camadas de processamento
    INTEGRATION = 3  # Camadas de integração
    MOTOR = 4  # Camadas motoras
    MEMORY = 5  # Camadas de memória

class NeuralStructure:
    def __init__(self):
        # Diferentes regiões corticais
        self.regions = {
            'sensory': [],
            'processing': [], 
            'integration': [],
            'memory': [],
            'motor': []
        }
        
        # Conexões entre regiões
        self.connections = {}
        
        # Estado de ativação
        self.activation_state = {}
        
        # Mecanismos de plasticidade
        self.plasticity_rules = {}
        
    def create_cortical_column(self, region: str, size: int):
        """Cria uma coluna cortical em uma região específica"""
        column = []
        for _ in range(size):
            # Cria neurônios com diferentes funções na coluna
            layer_neurons = {
                'sensory': ConsciousnessUnit(input_size=10),
                'integrator': ConsciousnessUnit(input_size=20),
                'output': ConsciousnessUnit(input_size=5)
            }
            column.append(layer_neurons)
            
        self.regions[region].append(column)
        
    def create_connection_pattern(self):
        """Define padrões de conexão entre regiões"""
        # Conexões feedforward
        self.add_connections('sensory', 'processing', 0.6)
        self.add_connections('processing', 'integration', 0.4)
        self.add_connections('integration', 'memory', 0.3)
        self.add_connections('integration', 'motor', 0.5)
        
        # Conexões feedback 
        self.add_connections('integration', 'processing', 0.2)
        self.add_connections('processing', 'sensory', 0.1)
        
        # Conexões laterais
        self.add_lateral_connections('processing', 0.3)
        self.add_lateral_connections('integration', 0.2)
        
    def add_connections(self, source: str, target: str, probability: float):
        """Adiciona conexões entre regiões com probabilidade dada"""
        if (source, target) not in self.connections:
            self.connections[(source, target)] = []
            
        for src_col in self.regions[source]:
            for tgt_col in self.regions[target]:
                if np.random.random() < probability:
                    self.connections[(source, target)].append((src_col, tgt_col))
                    
    def add_lateral_connections(self, region: str, probability: float):
        """Adiciona conexões laterais dentro de uma região"""
        columns = self.regions[region]
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                if np.random.random() < probability:
                    self.connections[(region, region)].append((col1, col2))
                    
    def update_activation(self):
        """Atualiza estado de ativação da rede"""
        # Atualiza ativação por região na ordem hierárquica
        for region in ['sensory', 'processing', 'integration', 'memory', 'motor']:
            region_activation = self.compute_region_activation(region)
            self.activation_state[region] = region_activation
            
    def compute_region_activation(self, region: str) -> np.ndarray:
        """Computa ativação de uma região considerando entradas de outras regiões"""
        region_input = np.zeros(len(self.regions[region]))
        
        # Soma entradas de todas as regiões conectadas
        for (src, tgt), connections in self.connections.items():
            if tgt == region:
                for src_col, tgt_col in connections:
                    # Propaga ativação através da conexão
                    connection_activity = self.propagate_activity(src_col, tgt_col)
                    region_input += connection_activity
                    
        return self.activate_region(region, region_input)
    
    def propagate_activity(self, source_column, target_column):
        """Propaga atividade entre colunas"""
        # Obtém ativação da coluna fonte
        source_activity = np.array([n.get_activation() for n in source_column])
        
        # Aplica pesos sinápticos
        connection_weights = self.get_connection_weights(source_column, target_column)
        propagated_activity = source_activity @ connection_weights
        
        return propagated_activity
        
    def activate_region(self, region: str, region_input: np.ndarray):
        """Ativa uma região com base nas entradas"""
        # Diferentes funções de ativação por região
        if region == 'sensory':
            return np.tanh(region_input)  # Sensorial: resposta rápida
        elif region == 'memory':
            return 1/(1 + np.exp(-region_input))  # Memória: resposta sigmoidal
        else:
            return np.maximum(0, region_input)  # Demais: ReLU
            
    def update_plasticity(self):
        """Atualiza regras de plasticidade"""
        for (src, tgt), connections in self.connections.items():
            for src_col, tgt_col in connections:
                # Aplica regras de plasticidade específicas para cada tipo de conexão
                if src == 'sensory':
                    self.update_sensory_plasticity(src_col, tgt_col)
                elif 'memory' in (src, tgt):
                    self.update_memory_plasticity(src_col, tgt_col)
                else:
                    self.update_standard_plasticity(src_col, tgt_col)
                    
    def process_sensory_input(self, input_data: np.ndarray):
        """Processa entrada sensorial"""
        # Atualiza região sensorial
        for i, column in enumerate(self.regions['sensory']):
            column_input = input_data[i] if i < len(input_data) else 0
            self.activate_sensory_column(column, column_input)
            
        # Propaga ativação pela rede
        self.update_activation()
        
        # Atualiza plasticidade
        self.update_plasticity()
        
    def get_motor_output(self) -> np.ndarray:
        """Obtém saída motora"""
        return self.activation_state['motor']
        
    def get_network_state(self) -> Dict:
        """Retorna estado atual da rede"""
        return {
            'activation': self.activation_state,
            'connections': len(self.connections),
            'regions': {r: len(cols) for r, cols in self.regions.items()}
        }
