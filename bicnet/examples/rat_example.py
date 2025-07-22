import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from enum import Enum
import matplotlib.pyplot as plt
from scipy.spatial import distance





class EmotionalState(Enum):
    CURIOUS = "curious"
    AFRAID = "afraid"
    HUNGRY = "hungry"
    SATISFIED = "satisfied"
    TIRED = "tired"
    ALERT = "alert"

@dataclass
class SensoryInput:
    """Representa entrada sensorial completa"""
    visual: np.ndarray  # Matriz 2D representando campo visual
    olfactory: np.ndarray  # Vetor de intensidades de odores
    tactile: np.ndarray  # Estado dos bigodes e tato
    auditory: np.ndarray  # Espectro de frequências auditivas
    proprioceptive: np.ndarray  # Estado corporal interno

@dataclass
class MotorOutput:
    """Representa saída motora completa"""
    velocity: Tuple[float, float]  # Velocidade (x, y)
    head_angle: float  # Ângulo da cabeça
    whisker_movement: float  # Movimento dos bigodes
    body_posture: np.ndarray  # Postura corporal

@dataclass
class HomeostasisState:
    """Estado homeostático do rato"""
    energy: float = 1.0  # Nível de energia
    hydration: float = 1.0  # Nível de hidratação
    temperature: float = 37.0  # Temperatura corporal
    fatigue: float = 0.0  # Nível de fadiga
    stress: float = 0.0  # Nível de estresse

class Environment:
    """Ambiente virtual para o rato"""
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        self.objects = {}  # Objetos no ambiente
        self.food_sources = []  # Fontes de comida
        self.water_sources = []  # Fontes de água
        self.odor_map = np.zeros(size)  # Mapa de odores
        self.obstacles = set()  # Obstáculos
        self.temperature_map = np.ones(size) * 22  # Temperatura ambiente
        
    def add_food(self, position: Tuple[int, int], amount: float):
        self.food_sources.append({"position": position, "amount": amount})
        self._update_odor_map()
        
    def add_water(self, position: Tuple[int, int], amount: float):
        self.water_sources.append({"position": position, "amount": amount})
        
    def add_obstacle(self, position: Tuple[int, int]):
        self.obstacles.add(position)
        
    def _update_odor_map(self):
        """Atualiza mapa de odores baseado em fontes"""
        self.odor_map = np.zeros(self.size)
        for food in self.food_sources:
            pos = food["position"]
            amount = food["amount"]
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    dist = np.sqrt((i-pos[0])**2 + (j-pos[1])**2)
                    self.odor_map[i,j] += amount * np.exp(-dist/10)

class VirtualRat:
    """Simulação completa de rato virtual"""
    def __init__(self, position: Tuple[float, float]):
        self.position = np.array(position, dtype=float)
        self.orientation = 0.0  # Ângulo em radianos
        self.velocity = np.zeros(2)
        self.homeostasis = HomeostasisState()
        self.emotional_state = EmotionalState.CURIOUS
        self.memory = {}
        self.brain = self._initialize_brain()
        
    def _initialize_brain(self):
        """Inicializa estruturas cerebrais"""
        return {
            'sensory_cortex': np.zeros((75, 1)),  # Ajustado para corresponder ao input total
            'motor_cortex': np.zeros((50, 50)),
            'hippocampus': np.zeros((30, 30)),
            'amygdala': np.zeros((20, 20)),
            'prefrontal_cortex': np.zeros((40, 40))
        }

        
    def sense_environment(self, environment: Environment) -> SensoryInput:
        """Coleta informações sensoriais do ambiente"""
        # Visão - campo visual de 180 graus
        visual_field = self._process_visual_input(environment)
        
        # Olfato - detecta gradientes químicos
        olfactory_input = self._process_olfactory_input(environment)
        
        # Tato - incluindo bigodes
        tactile_input = self._process_tactile_input(environment)
        
        # Audição
        auditory_input = self._process_auditory_input(environment)
        
        # Propriocepção
        proprioceptive_input = self._process_proprioceptive_input()
        
        return SensoryInput(
            visual=visual_field,
            olfactory=olfactory_input,
            tactile=tactile_input,
            auditory=auditory_input,
            proprioceptive=proprioceptive_input
        )
        
    def _process_visual_input(self, environment: Environment) -> np.ndarray:
        """Processa entrada visual"""
        visual_field = np.zeros((30, 180))  # 180 graus, 30 níveis de profundidade
        
        # Detecta objetos no campo visual
        for obstacle in environment.obstacles:
            angle = np.arctan2(obstacle[1]-self.position[1], 
                             obstacle[0]-self.position[0])
            dist = distance.euclidean(self.position, obstacle)
            if dist < 30:
                angle_idx = int((angle + np.pi) * 180/np.pi)
                depth_idx = int(dist)
                if 0 <= angle_idx < 180 and 0 <= depth_idx < 30:
                    visual_field[depth_idx, angle_idx] = 1
                    
        return visual_field
        
    def _process_olfactory_input(self, environment: Environment) -> np.ndarray:
        """Processa entrada olfativa"""
        pos = tuple(map(int, self.position))
        local_odor = environment.odor_map[pos[0], pos[1]]
        
        # Detecta gradiente de odor
        gradient_x = 0
        gradient_y = 0
        if pos[0] > 0 and pos[0] < environment.size[0]-1:
            gradient_x = environment.odor_map[pos[0]+1, pos[1]] - \
                        environment.odor_map[pos[0]-1, pos[1]]
        if pos[1] > 0 and pos[1] < environment.size[1]-1:
            gradient_y = environment.odor_map[pos[0], pos[1]+1] - \
                        environment.odor_map[pos[0], pos[1]-1]
                        
        return np.array([local_odor, gradient_x, gradient_y])
        
    def update_state(self, environment: Environment, dt: float):
        """Atualiza estado interno do rato"""
        # Atualiza homeostase
        self._update_homeostasis(dt)
        
        # Coleta informação sensorial
        sensory_input = self.sense_environment(environment)
        
        # Processa no cérebro
        motor_output = self._process_brain(sensory_input)
        
        # Atualiza posição e estado físico
        self._update_physical_state(motor_output, environment, dt)
        
        # Atualiza estado emocional
        self._update_emotional_state(sensory_input)
        
    def _update_homeostasis(self, dt: float):
        """Atualiza estado homeostático"""
        # Gasta energia
        self.homeostasis.energy -= 0.1 * dt
        self.homeostasis.hydration -= 0.05 * dt
        
        # Aumenta fadiga
        self.homeostasis.fatigue += 0.02 * dt
        
        # Regula temperatura
        if self.homeostasis.temperature > 37.0:
            self.homeostasis.temperature -= 0.1 * dt
        elif self.homeostasis.temperature < 37.0:
            self.homeostasis.temperature += 0.1 * dt
            
    def _process_brain(self, sensory_input: SensoryInput) -> MotorOutput:
        """Processa informação no cérebro e gera saída motora"""
        # Processa informação sensorial
        self.brain['sensory_cortex'] = self._process_sensory_cortex(sensory_input)
        
        # Atualiza memória espacial
        self._update_hippocampus()
        
        # Avalia ameaças/recompensas
        self._process_amygdala(sensory_input)
        
        # Toma decisões
        motor_commands = self._process_motor_cortex()
        
        return motor_commands
        
    def _update_emotional_state(self, sensory_input: SensoryInput):
        """Atualiza estado emocional baseado em entradas sensoriais e estado interno"""
        # Medo/ansiedade baseado em ameaças detectadas
        threat_level = np.max(sensory_input.visual)  # Simplificação
        
        # Fome/sede baseado em homeostase
        if self.homeostasis.energy < 0.3:
            self.emotional_state = EmotionalState.HUNGRY
        elif self.homeostasis.fatigue > 0.8:
            self.emotional_state = EmotionalState.TIRED
        elif threat_level > 0.7:
            self.emotional_state = EmotionalState.AFRAID
        else:
            self.emotional_state = EmotionalState.CURIOUS

    def _process_tactile_input(self, environment: Environment) -> np.ndarray:
        """Processa entrada tátil, incluindo bigodes"""
        tactile_input = np.zeros(24)  # 20 bigodes + 4 patas
        
        # Simula bigodes
        for i in range(20):
            angle = self.orientation + (i - 10) * np.pi/20
            whisker_tip = self.position + np.array([
                np.cos(angle), 
                np.sin(angle)
            ]) * 5  # Comprimento do bigode
            
            # Verifica colisões
            whisker_pos = tuple(map(int, whisker_tip))
            if (whisker_pos in environment.obstacles or 
                not (0 <= whisker_pos[0] < environment.size[0] and 
                     0 <= whisker_pos[1] < environment.size[1])):
                tactile_input[i] = 1.0
                
        # Simula tato nas patas
        pos = tuple(map(int, self.position))
        for i in range(4):
            if pos in environment.obstacles:
                tactile_input[20 + i] = 1.0
                
        return tactile_input

    def _process_auditory_input(self, environment: Environment) -> np.ndarray:
        """Processa entrada auditiva"""
        # Simula 8 frequências diferentes (simplificado)
        auditory_input = np.zeros(8)
        
        # Simula sons do ambiente
        for obj in environment.objects.values():
            if 'sound' in obj:
                distance = np.linalg.norm(self.position - obj['position'])
                intensity = obj['sound']['amplitude'] * np.exp(-distance/50)
                frequency_idx = obj['sound']['frequency']
                auditory_input[frequency_idx] += intensity
                
        return auditory_input

    def _process_proprioceptive_input(self) -> np.ndarray:
        """Processa entrada proprioceptiva"""
        # 10 valores representando diferentes aspectos do estado corporal
        proprioceptive_input = np.zeros(10)
        
        # Posição da cabeça
        proprioceptive_input[0] = np.sin(self.orientation)
        proprioceptive_input[1] = np.cos(self.orientation)
        
        # Velocidade
        proprioceptive_input[2:4] = self.velocity
        
        # Estado das patas (posição relativa ao corpo)
        proprioceptive_input[4:8] = 0.5  # Posição neutra
        
        # Tensão muscular (simplificado)
        proprioceptive_input[8] = np.linalg.norm(self.velocity)
        
        # Equilíbrio/postura
        proprioceptive_input[9] = 1.0  # Postura normal
        
        return proprioceptive_input

    def _update_physical_state(self, motor_output: MotorOutput, environment: Environment, dt: float):
        """Atualiza estado físico do rato"""
        # Atualiza posição
        velocity_array = np.array(motor_output.velocity)
        new_position = self.position + velocity_array * dt
        
        # Verifica colisões
        new_pos_tuple = tuple(map(int, new_position))
        if (0 <= new_pos_tuple[0] < environment.size[0] and 
            0 <= new_pos_tuple[1] < environment.size[1] and 
            new_pos_tuple not in environment.obstacles):
            self.position = new_position
            
        # Atualiza orientação
        self.orientation = motor_output.head_angle
        
        # Atualiza bigodes
        self.whisker_state = np.sin(np.linspace(0, 2*np.pi, 20) + 
                                  motor_output.whisker_movement)
        
        # Gasta energia baseado no movimento
        movement_energy = np.linalg.norm(motor_output.velocity) * 0.01
        self.homeostasis.energy -= movement_energy * dt

    def _update_hippocampus(self):
        """Atualiza memória espacial no hipocampo"""
        # Posição atual em coordenadas de grade
        grid_pos = tuple(map(int, self.position))
        
        # Atualiza mapa cognitivo
        if 'spatial_map' not in self.memory:
            self.memory['spatial_map'] = np.zeros(self.brain['hippocampus'].shape)
            
        # Marca posição atual e decai outras posições
        self.memory['spatial_map'] *= 0.95  # Decay
        x, y = grid_pos
        if (x < self.memory['spatial_map'].shape[0] and 
            y < self.memory['spatial_map'].shape[1]):
            self.memory['spatial_map'][x, y] = 1.0

    def _process_amygdala(self, sensory_input: SensoryInput):
        """Processa emoções na amígdala"""
        # Avalia ameaças
        threat_level = np.max(sensory_input.visual)
        
        # Avalia recompensas
        reward_level = np.max(sensory_input.olfactory)
        
        # Atualiza níveis de stress
        self.homeostasis.stress = 0.8 * self.homeostasis.stress + \
                                 0.2 * threat_level

    def _process_sensory_cortex(self, sensory_input: SensoryInput) -> np.ndarray:
        """Processa informação no córtex sensorial"""
        # Combina todas as entradas sensoriais
        visual_processed = np.mean(sensory_input.visual, axis=1)  # 30 valores
        # Total de valores:
        # visual_processed: 30
        # olfactory: 3
        # tactile: 24
        # auditory: 8
        # proprioceptive: 10
        # Total: 75 valores
        
        combined_input = np.concatenate([
            visual_processed,      # 30 valores
            sensory_input.olfactory,    # 3 valores
            sensory_input.tactile,      # 24 valores
            sensory_input.auditory,     # 8 valores
            sensory_input.proprioceptive # 10 valores
        ])
        
        # Retorna já no formato correto (75,1)
        return combined_input.reshape(-1, 1)

    def _process_motor_cortex(self) -> MotorOutput:
        """Gera comandos motores baseado no estado do cérebro"""
        # Decisão de movimento baseada em estado interno
        if self.emotional_state == EmotionalState.HUNGRY:
            # Procura comida - movimento mais rápido
            velocity = np.array([0.5, 0.5])
        elif self.emotional_state == EmotionalState.AFRAID:
            # Foge - movimento rápido em direção oposta à ameaça
            velocity = np.array([-1.0, -1.0])
        else:
            # Exploração normal
            velocity = np.array([0.2, 0.2])
            
        # Ajusta orientação baseado em estímulos
        head_angle = self.orientation + np.random.normal(0, 0.1)
        
        # Movimento dos bigodes baseado em estado emocional
        whisker_movement = 1.0 if self.emotional_state == EmotionalState.ALERT else 0.5
        
        return MotorOutput(
            velocity=(float(velocity[0]), float(velocity[1])),  # Explicit float tuple
            head_angle=head_angle,
            whisker_movement=whisker_movement,
            body_posture=np.ones(4)
        )


def simulate_rat_behavior():
    """Simula comportamento do rato em ambiente virtual"""
    # Cria ambiente
    env = Environment((100, 100))
    
    # Adiciona elementos ao ambiente
    env.add_food((80, 80), 1.0)
    env.add_water((20, 20), 1.0)
    env.add_obstacle((50, 50))
    env.add_obstacle((40, 30))
    
    # Cria rato virtual
    rat = VirtualRat((10, 10))
    
    # Simulação principal
    dt = 0.1
    total_time = 240
    steps = int(total_time / dt)
    
    history = []
    
    for step in range(steps):
        # Atualiza rato
        rat.update_state(env, dt)
        
        # Registra estado
        history.append({
            'position': rat.position.copy(),
            'energy': rat.homeostasis.energy,
            'emotional_state': rat.emotional_state
        })
        
        # Visualiza periodicamente
        if step % 10 == 0:
            visualize_state(env, rat, step)
            
def visualize_state(env: Environment, rat: VirtualRat, step: int):
    """Visualiza estado atual da simulação"""
    plt.figure(figsize=(15, 5))
    
    # Ambiente e posição do rato
    plt.subplot(131)
    plt.imshow(env.odor_map)
    plt.plot(rat.position[0], rat.position[1], 'ro')
    plt.title('Environment & Rat Position')
    
    # Estado homeostático
    plt.subplot(132)
    homeostasis_values = [
        rat.homeostasis.energy,
        rat.homeostasis.hydration,
        rat.homeostasis.temperature/100,
        rat.homeostasis.fatigue,
        rat.homeostasis.stress
    ]
    plt.bar(['Energy', 'Hydration', 'Temp', 'Fatigue', 'Stress'],
            homeostasis_values)
    plt.title('Homeostatic State')
    
    # Estado emocional
    plt.subplot(133)
    plt.text(0.5, 0.5, f"Emotional State:\n{rat.emotional_state.value}",
             ha='center', va='center')
    plt.axis('off')
    plt.title('Emotional State')
    
    plt.tight_layout()
    plt.savefig(f'/home/rexionmars/estudos/MakeConciency/LOGS/rate_state_run_1/rat_state_{step}.png')
    plt.close()

if __name__ == "__main__":
    simulate_rat_behavior()