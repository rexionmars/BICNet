import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random

@dataclass
class Position:
    x: float
    y: float

class Environment:
    def __init__(self, size: Tuple[int, int] = (20, 20)):
        self.size = size
        self.food_positions: List[Position] = []
        self.obstacles: List[Position] = []
        self.grid = np.zeros(size)
        
    def add_food(self, position: Position):
        self.food_positions.append(position)
        self.grid[int(position.x)][int(position.y)] = 1
        
    def add_obstacle(self, position: Position):
        self.obstacles.append(position)
        self.grid[int(position.x)][int(position.y)] = -1

class BrainRegion:
    def __init__(self, name: str, num_neurons: int):
        self.name = name
        self.num_neurons = num_neurons
        self.activation = np.zeros(num_neurons)
        self.connections = {}
        
        # Níveis de neurotransmissores
        self.neurotransmitters = {
            'dopamine': 1.0,    # Recompensa
            'serotonin': 1.0,   # Humor/satisfação
            'glutamate': 1.0,   # Excitação
            'gaba': 1.0         # Inibição
        }
        
    def connect_to(self, other_region: 'BrainRegion', weight_matrix: np.ndarray):
        self.connections[other_region.name] = {
            'region': other_region,
            'weights': weight_matrix
        }
        
    def update(self, input_signal: np.ndarray) -> np.ndarray:
        # Atualiza ativação baseado no input
        self.activation = np.tanh(input_signal)
        
        # Modula baseado em neurotransmissores
        self.activation *= self.neurotransmitters['glutamate']
        self.activation *= (1 - 0.5 * self.neurotransmitters['gaba'])
        
        return self.activation

class RatBrain:
    def __init__(self):
        # Inicializa regiões cerebrais principais
        self.hippocampus = BrainRegion('hippocampus', 100)    # Memória espacial
        self.amygdala = BrainRegion('amygdala', 50)          # Emoções/motivação
        self.prefrontal = BrainRegion('prefrontal', 80)      # Decisões
        self.striatum = BrainRegion('striatum', 60)          # Recompensa
        self.motor_cortex = BrainRegion('motor_cortex', 40)  # Movimento
        
        # Conecta regiões
        self._setup_connections()
        
        # Estado interno
        self.hunger_level = 0.5
        self.stress_level = 0.0
        self.memory = []
        
    def _setup_connections(self):
        # Conexões entre regiões (simplificadas)
        self.hippocampus.connect_to(
            self.prefrontal,
            np.random.normal(0, 0.1, (self.prefrontal.num_neurons, self.hippocampus.num_neurons))
        )
        self.prefrontal.connect_to(
            self.motor_cortex,
            np.random.normal(0, 0.1, (self.motor_cortex.num_neurons, self.prefrontal.num_neurons))
        )
        self.amygdala.connect_to(
            self.prefrontal,
            np.random.normal(0, 0.1, (self.prefrontal.num_neurons, self.amygdala.num_neurons))
        )
        
    def process_sensory_input(self, visual_input: np.ndarray, olfactory_input: np.ndarray) -> np.ndarray:
        # Processa entrada sensorial
        combined_input = np.concatenate([visual_input.flatten(), olfactory_input.flatten()])
        
        # Ativa hipocampo (memória espacial)
        hippocampus_activation = self.hippocampus.update(combined_input[:self.hippocampus.num_neurons])
        
        # Ativa amígdala (resposta emocional)
        amygdala_activation = self.amygdala.update(combined_input[:self.amygdala.num_neurons])
        
        # Integra no córtex prefrontal
        prefrontal_input = np.concatenate([hippocampus_activation, amygdala_activation])
        prefrontal_activation = self.prefrontal.update(prefrontal_input)
        
        # Gera comando motor
        motor_command = self.motor_cortex.update(prefrontal_activation)
        
        return motor_command

class Rat:
    def __init__(self, start_position: Position):
        self.position = start_position
        self.brain = RatBrain()
        self.velocity = Position(0, 0)
        self.memory_positions = []
        
    def update_state(self, environment: Environment) -> Tuple[Position, Dict]:
        # Coleta input sensorial
        visual_input = self._get_visual_input(environment)
        olfactory_input = self._get_olfactory_input(environment)
        
        # Processa no cérebro
        motor_command = self.brain.process_sensory_input(visual_input, olfactory_input)
        
        # Atualiza posição baseado no comando motor
        self._update_position(motor_command, environment)
        
        # Registra posição na memória
        self.memory_positions.append(Position(self.position.x, self.position.y))
        
        # Coleta dados de atividade cerebral
        brain_activity = self._collect_brain_activity()
        
        return self.position, brain_activity
    
    def _get_visual_input(self, environment: Environment) -> np.ndarray:
        # Simula campo visual (5x5 ao redor do rato)
        visual_field = np.zeros((5, 5))
        x, y = int(self.position.x), int(self.position.y)
        
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= x+i < environment.size[0] and 0 <= y+j < environment.size[1]:
                    visual_field[i+2][j+2] = environment.grid[x+i][y+j]
                    
        return visual_field
    
    def _get_olfactory_input(self, environment: Environment) -> np.ndarray:
        # Simula detecção de odor (diminui com a distância da comida)
        olfactory_signal = np.zeros(environment.size)
        
        for food in environment.food_positions:
            distance = np.sqrt((self.position.x - food.x)**2 + (self.position.y - food.y)**2)
            strength = 1 / (1 + distance)
            olfactory_signal[int(food.x)][int(food.y)] = strength
            
        return olfactory_signal
    
    def _update_position(self, motor_command: np.ndarray, environment: Environment):
        # Converte comando motor em movimento
        dx = motor_command[0] * 0.5
        dy = motor_command[1] * 0.5
        
        # Verifica colisões e atualiza posição
        new_x = np.clip(self.position.x + dx, 0, environment.size[0]-1)
        new_y = np.clip(self.position.y + dy, 0, environment.size[1]-1)
        
        # Verifica obstáculos
        if Position(new_x, new_y) not in environment.obstacles:
            self.position.x = new_x
            self.position.y = new_y
    
    def _collect_brain_activity(self) -> Dict:
        return {
            'hippocampus': self.brain.hippocampus.activation.copy(),
            'amygdala': self.brain.amygdala.activation.copy(),
            'prefrontal': self.brain.prefrontal.activation.copy(),
            'motor_cortex': self.brain.motor_cortex.activation.copy(),
            'neurotransmitters': {
                region.name: region.neurotransmitters.copy()
                for region in [self.brain.hippocampus, self.brain.amygdala, 
                             self.brain.prefrontal, self.brain.motor_cortex]
            }
        }

def run_experiment(steps: int = 1000):
    # Configura ambiente
    env = Environment()
    env.add_food(Position(15, 15))
    env.add_obstacle(Position(10, 10))
    
    # Cria rato
    rat = Rat(Position(2, 2))
    
    # Registros
    positions = []
    brain_activities = []
    
    # Executa simulação
    for _ in range(steps):
        pos, activity = rat.update_state(env)
        positions.append((pos.x, pos.y))
        brain_activities.append(activity)
    
    return positions, brain_activities

def analyze_results(positions: List, brain_activities: List):
    # Análise de trajetória
    positions = np.array(positions)
    
    plt.figure(figsize=(15, 5))
    
    # Plot da trajetória
    plt.subplot(121)
    plt.plot(positions[:,0], positions[:,1], 'b-', alpha=0.5)
    plt.plot(positions[0,0], positions[0,1], 'go', label='Start')
    plt.plot(positions[-1,0], positions[-1,1], 'ro', label='End')
    plt.title('Trajetória do Rato')
    plt.legend()
    
    # Plot da atividade cerebral
    plt.subplot(122)
    activities = np.array([a['hippocampus'].mean() for a in brain_activities])
    plt.plot(activities, label='Hipocampo')
    plt.title('Atividade Cerebral Média')
    plt.legend()
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('rat_simulation.png')

if __name__ == "__main__":
    positions, brain_activities = run_experiment()
    analyze_results(positions, brain_activities)
