import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
import torch
import torch.nn.functional as F
from scipy import signal
import logging
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BrainRhythm:
    """Representa um ritmo cerebral específico"""
    def __init__(self, name: str, frequency: float, amplitude: float):
        self.name = name
        self.frequency = frequency  # Hz
        self.amplitude = amplitude
        self.phase = 0.0
        
    def update(self, dt: float) -> float:
        """Atualiza e retorna o valor atual do ritmo"""
        self.phase += 2 * np.pi * self.frequency * dt
        return self.amplitude * np.sin(self.phase)

class BrainRhythms:
    """Gerencia múltiplos ritmos cerebrais"""
    def __init__(self):
        self.rhythms = {
            'delta': BrainRhythm('delta', 2, 1.0),    # 1-4 Hz
            'theta': BrainRhythm('theta', 6, 0.8),    # 4-8 Hz
            'alpha': BrainRhythm('alpha', 10, 0.6),   # 8-13 Hz
            'beta': BrainRhythm('beta', 20, 0.4),     # 13-30 Hz
            'gamma': BrainRhythm('gamma', 40, 0.3)    # 30-100 Hz
        }
        
    def update(self, dt: float) -> Dict[str, float]:
        """Atualiza todos os ritmos"""
        return {name: rhythm.update(dt) 
                for name, rhythm in self.rhythms.items()}

@dataclass
class DendriticSpine:
    """Representa uma espinha dendrítica"""
    size: float = 1.0
    stability: float = 0.5
    calcium: float = 0.0
    proteins: Dict[str, float] = None
    MAX_SIZE: float = 5.0  # Limite máximo de tamanho
    
    def __post_init__(self):
        self.proteins = {
            'actin': 1.0,
            'PSD95': 1.0,
            'AMPAR': 1.0,
            'NMDAR': 1.0
        }
    
    def update_structure(self, activity: float):
        """Atualiza estrutura da espinha baseado em atividade"""
        # Limita a atividade para evitar overflow
        activity = np.clip(activity, -10.0, 10.0)
        
        # Atualiza proteínas estruturais com limites
        for protein in ['actin', 'PSD95', 'AMPAR', 'NMDAR']:
            current = self.proteins[protein]
            delta = current * 0.1 * activity
            self.proteins[protein] = np.clip(current + delta, 0.1, 10.0)
        
        # Atualiza tamanho baseado em proteínas com limite
        target_size = np.clip(np.mean(list(self.proteins.values())), 0.1, self.MAX_SIZE)
        self.size = np.clip(self.size + 0.1 * (target_size - self.size), 0.1, self.MAX_SIZE)
        
        # Atualiza estabilidade
        self.stability = np.clip(
            self.stability + 0.01 * (self.size - 1.0),
            0, 1
        )

@dataclass
class DendriticSegment:
    """Representa um segmento dendrítico"""
    length: float
    diameter: float
    spines: List[DendriticSpine]
    parent: Optional['DendriticSegment'] = None
    children: List['DendriticSegment'] = None
    local_potential: float = 0.0
    calcium_concentration: float = 0.0
    MAX_DIAMETER: float = 5.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        # Garante valores iniciais válidos
        self.diameter = np.clip(self.diameter, 0.1, self.MAX_DIAMETER)
        self.length = np.clip(self.length, 0.1, 10.0)
            
    def add_spine(self):
        """Adiciona nova espinha dendrítica"""
        self.spines.append(DendriticSpine())
        
    def remove_unstable_spines(self):
        """Remove espinhas instáveis"""
        self.spines = [spine for spine in self.spines 
                      if spine.stability > 0.2]
        
    def update_morphology(self, activity: float):
        """Atualiza morfologia baseado em atividade"""
        # Limita a atividade
        activity = np.clip(activity, -10.0, 10.0)
        
        # Atualiza diâmetro com limite
        delta_diameter = self.diameter * 0.01 * activity
        self.diameter = np.clip(self.diameter + delta_diameter, 0.1, self.MAX_DIAMETER)
        
        # Atualiza espinhas
        for spine in self.spines:
            spine.update_structure(activity)
        
        # Remove espinhas instáveis
        self.remove_unstable_spines()
        
        # Adiciona novas espinhas com probabilidade baseada em atividade
        if np.random.random() < 0.1 * abs(activity):
            self.add_spine()
    def _update_dendritic_activity(self, input_current: float):
        """Atualiza atividade em toda a árvore dendrítica"""
        def update_segment(segment: DendriticSegment, input_value: float):
            # Limita input
            input_value = np.clip(input_value, -100.0, 100.0)
            
            # Atenuação baseada em morfologia com limites
            attenuation = np.clip(
                segment.diameter / (segment.length + 1e-6),
                0.0, 1.0
            )
            
            # Contribuição das espinhas com limite
            spine_contribution = np.clip(
                sum(spine.size for spine in segment.spines),
                0.0, 100.0
            )
            
            # Atividade local com limite
            segment.local_potential = np.clip(
                (input_value + spine_contribution) * attenuation,
                -100.0, 100.0
            )
            
            # Propaga para filhos
            for child in segment.children:
                update_segment(child, segment.local_potential)
            
            # Atualiza morfologia
            segment.update_morphology(abs(segment.local_potential))
            
        update_segment(self.dendrites, input_current)

class Neuromodulator:
    """Representa um neuromodulador específico"""
    def __init__(self, name: str, baseline: float = 1.0):
        self.name = name
        self.concentration = baseline
        self.baseline = baseline
        self.decay_rate = 0.1
        
    def release(self, amount: float):
        """Libera neuromodulador"""
        self.concentration += amount
        
    def update(self):
        """Atualiza concentração"""
        self.concentration += (self.baseline - self.concentration) * self.decay_rate

class RegionalNeuromodulation:
    """Gerencia neuromodulação específica por região"""
    def __init__(self):
        self.modulators = {
            'dopamine': Neuromodulator('dopamine'),
            'serotonin': Neuromodulator('serotonin'),
            'acetylcholine': Neuromodulator('acetylcholine'),
            'norepinephrine': Neuromodulator('norepinephrine')
        }
        
    def update(self, region_activity: Dict[str, float]):
        """Atualiza neuromoduladores baseado em atividade regional"""
        for name, modulator in self.modulators.items():
            if name in region_activity:
                modulator.release(0.1 * region_activity[name])
            modulator.update()

class EnhancedNeuron:
    """Neurônio com dinâmica complexa"""
    def __init__(self, id: int):
        self.id = id
        self.dendrites = self._create_dendritic_tree()
        self.rhythms = BrainRhythms()
        self.soma_potential = 0.0
        self.calcium = 0.0
        self.activation_history = []
        
    def _create_dendritic_tree(self, depth: int = 3) -> DendriticSegment:
        """Cria árvore dendrítica complexa"""
        def create_branch(length: float, diameter: float, current_depth: int) -> DendriticSegment:
            segment = DendriticSegment(
                length=length,
                diameter=diameter,
                spines=[DendriticSpine() for _ in range(np.random.randint(5, 15))]
            )
            
            if current_depth > 0:
                num_children = np.random.randint(2, 4)
                for _ in range(num_children):
                    child_length = length * 0.8
                    child_diameter = diameter * 0.7
                    child = create_branch(child_length, child_diameter, current_depth - 1)
                    child.parent = segment
                    segment.children.append(child)
                    
            return segment
            
        return create_branch(1.0, 1.0, depth)
        
    def update(self, dt: float, input_current: float):
        """Atualiza estado do neurônio"""
        # Atualiza ritmos
        rhythm_values = self.rhythms.update(dt)
        
        # Soma influências rítmicas
        rhythmic_input = sum(rhythm_values.values())
        
        # Propaga atividade pelos dendritos
        self._update_dendritic_activity(input_current + rhythmic_input)
        
        # Atualiza potencial somático
        self.soma_potential = np.tanh(self.dendrites.local_potential)
        
        # Registra ativação
        self.activation_history.append(self.soma_potential)
        
        return self.soma_potential
        
    def _update_dendritic_activity(self, input_current: float):
        """Atualiza atividade em toda a árvore dendrítica"""
        def update_segment(segment: DendriticSegment, input_value: float):
            # Atenuação baseada em morfologia
            attenuation = segment.diameter / (segment.length + 1e-6)
            
            # Contribuição das espinhas
            spine_contribution = sum(spine.size for spine in segment.spines)
            
            # Atividade local
            segment.local_potential = (input_value + spine_contribution) * attenuation
            
            # Propaga para filhos
            for child in segment.children:
                update_segment(child, segment.local_potential)
            
            # Atualiza morfologia
            segment.update_morphology(abs(segment.local_potential))
            
        update_segment(self.dendrites, input_current)

class EnhancedNeuralNetwork:
    """Rede neural com dinâmica complexa"""
    def __init__(self, num_neurons: int):
        self.neurons = [EnhancedNeuron(i) for i in range(num_neurons)]
        self.neuromodulation = RegionalNeuromodulation()
        self.time = 0.0
        
    def simulate_step(self, dt: float, external_input: np.ndarray):
        """Simula um passo temporal"""
        # Atualiza neuromodulação
        region_activity = {
            'dopamine': np.mean([n.soma_potential for n in self.neurons[:25]]),
            'serotonin': np.mean([n.soma_potential for n in self.neurons[25:50]]),
            'acetylcholine': np.mean([n.soma_potential for n in self.neurons[50:75]]),
            'norepinephrine': np.mean([n.soma_potential for n in self.neurons[75:]])
        }
        self.neuromodulation.update(region_activity)
        
        # Atualiza cada neurônio
        activations = []
        for i, neuron in enumerate(self.neurons):
            activation = neuron.update(dt, external_input[i])
            activations.append(activation)
            
        self.time += dt
        return np.array(activations)

class EnhancedVisualizer:
    """Visualizador avançado para a rede neural"""
    def __init__(self, network: EnhancedNeuralNetwork):
        self.network = network
        
    def visualize_state(self, timestep: int):
        """Visualiza estado atual da rede"""
        plt.figure(figsize=(20, 15))
        
        # Ritmos cerebrais
        plt.subplot(331)
        self._plot_brain_rhythms()
        plt.title('Brain Rhythms')
        
        # Estrutura dendrítica
        plt.subplot(332)
        self._plot_dendritic_structure()
        plt.title('Dendritic Structure')
        
        # Neuromoduladores
        plt.subplot(333)
        self._plot_neuromodulators()
        plt.title('Neuromodulators')
        
        # Atividade neural
        plt.subplot(334)
        self._plot_neural_activity()
        plt.title('Neural Activity')
        
        # Espinhas dendríticas
        plt.subplot(335)
        self._plot_spine_distribution()
        plt.title('Spine Distribution')
        
        plt.tight_layout()
        plt.savefig(f'enhanced_neural_state_{timestep}.png')
        plt.close()
        
    def _plot_brain_rhythms(self):
        sample_neuron = self.network.neurons[0]
        rhythm_values = sample_neuron.rhythms.update(0.001)
        plt.bar(rhythm_values.keys(), rhythm_values.values())
        
    def _plot_dendritic_structure(self):
        def plot_segment(segment: DendriticSegment, x: float, y: float, dx: float, dy: float):
            if segment.parent is not None:
                plt.plot([x, x+dx], [y, y+dy], 'k-', linewidth=segment.diameter*2)
            for spine in segment.spines:
                plt.scatter(x+dx/2, y+dy/2, s=spine.size*20, c='r', alpha=0.5)
            for i, child in enumerate(segment.children):
                new_dx = dx * 0.8 * np.cos(i * np.pi/3)
                new_dy = dy * 0.8 * np.sin(i * np.pi/3)
                plot_segment(child, x+dx, y+dy, new_dx, new_dy)
                
        sample_neuron = self.network.neurons[0]
        plot_segment(sample_neuron.dendrites, 0, 0, 1, 1)
        
    def _plot_neuromodulators(self):
        modulators = self.network.neuromodulation.modulators
        plt.bar([m.name for m in modulators.values()],
                [m.concentration for m in modulators.values()])
        
    def _plot_neural_activity(self):
        activities = [n.soma_potential for n in self.network.neurons]
        plt.plot(activities)
        
    def _plot_spine_distribution(self):
        """Plota distribuição de tamanhos das espinhas"""
        all_sizes = []
        sample_neuron = self.network.neurons[0]
        
        def collect_spine_sizes(segment):
            # Filtra valores inválidos
            valid_sizes = [spine.size for spine in segment.spines 
                         if np.isfinite(spine.size) and spine.size > 0]
            all_sizes.extend(valid_sizes)
            for child in segment.children:
                collect_spine_sizes(child)
                
        collect_spine_sizes(sample_neuron.dendrites)
        
        if all_sizes:  # Verifica se há dados válidos
            # Limita o range do histograma
            plt.hist(all_sizes, bins=20, range=(0, 5))
        else:
            logger.warning("No valid spine sizes to plot")

def simulate_enhanced_network():
    """Simula rede neural aprimorada"""
    network = EnhancedNeuralNetwork(100)
    visualizer = EnhancedVisualizer(network)
    
    dt = 0.001  # 1ms
    simulation_time = 1.0  # 1 segundo
    steps = int(simulation_time / dt)
    
    for step in range(steps):
        # Gera entrada externa com múltiplas frequências
        t = step * dt
        external_input = (
            0.3 * np.sin(2 * np.pi * 1 * t) +   # 1 Hz
            0.2 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz
            0.1 * np.sin(2 * np.pi * 40 * t)    # 40 Hz
        )
        external_input = np.ones(100) * external_input
        
        # Adiciona ruído
        external_input += 0.05 * np.random.randn(100)
        
        # Simula um passo
        network.simulate_step(dt, external_input)
        
        # Visualiza periodicamente
        # Continuação da função simulate_enhanced_network():
        if step % 1000 == 0:  # A cada 1ms
            visualizer.visualize_state(step)
            
            # Análise do estado atual
            analyze_network_state(network, step)
            
            logger.info(f"Completed step {step}/{steps}")

def analyze_network_state(network: EnhancedNeuralNetwork, step: int):
    """Analisa e registra o estado atual da rede"""
    # Análise de ritmos
    rhythm_power = analyze_rhythm_power(network)
    
    # Análise dendrítica
    dendritic_stats = analyze_dendritic_structure(network)
    
    # Análise de neuromodulação
    neuromod_state = analyze_neuromodulation(network)
    
    # Salva resultados
    save_analysis_results(rhythm_power, dendritic_stats, neuromod_state, step)

def analyze_rhythm_power(network: EnhancedNeuralNetwork) -> Dict[str, float]:
    """Analisa a potência dos diferentes ritmos cerebrais"""
    rhythm_power = {}
    sample_neuron = network.neurons[0]
    
    for name, rhythm in sample_neuron.rhythms.rhythms.items():
        # Calcula potência usando a amplitude e fase atual
        power = rhythm.amplitude * abs(np.sin(rhythm.phase))
        rhythm_power[name] = power
        
    return rhythm_power

def analyze_dendritic_structure(network: EnhancedNeuralNetwork) -> Dict[str, float]:
    """Analisa a estrutura dendrítica atual"""
    stats = {
        'total_spines': 0,
        'avg_spine_size': 0.0,
        'avg_segment_diameter': 0.0,
        'total_segments': 0
    }
    
    def analyze_segment(segment: DendriticSegment):
        stats['total_spines'] += len(segment.spines)
        stats['avg_spine_size'] += sum(spine.size for spine in segment.spines)
        stats['avg_segment_diameter'] += segment.diameter
        stats['total_segments'] += 1
        
        for child in segment.children:
            analyze_segment(child)
    
    for neuron in network.neurons:
        analyze_segment(neuron.dendrites)
    
    # Calcula médias
    if stats['total_segments'] > 0:
        stats['avg_segment_diameter'] /= stats['total_segments']
    if stats['total_spines'] > 0:
        stats['avg_spine_size'] /= stats['total_spines']
    
    return stats

def analyze_neuromodulation(network: EnhancedNeuralNetwork) -> Dict[str, float]:
    """Analisa o estado atual da neuromodulação"""
    return {name: mod.concentration 
            for name, mod in network.neuromodulation.modulators.items()}

def save_analysis_results(rhythm_power: Dict[str, float],
                         dendritic_stats: Dict[str, float],
                         neuromod_state: Dict[str, float],
                         step: int):
    """Salva resultados da análise"""
    results = {
        'step': step,
        'rhythm_power': rhythm_power,
        'dendritic_stats': dendritic_stats,
        'neuromodulation': neuromod_state
    }
    
    # Salva em arquivo
    with open(f'analysis_results_{step}.json', 'w') as f:
        json.dump(results, f)

class NetworkAnalyzer:
    """Classe para análise contínua da rede neural"""
    def __init__(self, network: EnhancedNeuralNetwork):
        self.network = network
        self.rhythm_history = []
        self.spine_history = []
        self.neuromod_history = []
        
    def update(self, step: int):
        """Atualiza análise da rede"""
        # Registra estado dos ritmos
        rhythm_power = analyze_rhythm_power(self.network)
        self.rhythm_history.append(rhythm_power)
        
        # Registra estado das espinhas
        dendritic_stats = analyze_dendritic_structure(self.network)
        self.spine_history.append(dendritic_stats)
        
        # Registra estado da neuromodulação
        neuromod_state = analyze_neuromodulation(self.network)
        self.neuromod_history.append(neuromod_state)
        
        # Periodicamente gera relatório
        if step % 10000 == 0:
            self.generate_report(step)
    
    def generate_report(self, step: int):
        """Gera relatório detalhado da análise"""
        plt.figure(figsize=(20, 15))
        
        # Evolução dos ritmos
        plt.subplot(331)
        self._plot_rhythm_evolution()
        plt.title('Rhythm Power Evolution')
        
        # Evolução das espinhas
        plt.subplot(332)
        self._plot_spine_evolution()
        plt.title('Spine Evolution')
        
        # Evolução da neuromodulação
        plt.subplot(333)
        self._plot_neuromod_evolution()
        plt.title('Neuromodulator Evolution')
        
        plt.tight_layout()
        plt.savefig(f'network_analysis_{step}.png')
        plt.close()
    
    def _plot_rhythm_evolution(self):
        """Plota evolução dos ritmos cerebrais"""
        data = np.array([[r[name] for name in self.rhythm_history[0].keys()] 
                        for r in self.rhythm_history])
        plt.imshow(data.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Power')
        plt.xlabel('Time Step')
        plt.ylabel('Rhythm Type')
        
    def _plot_spine_evolution(self):
        """Plota evolução das espinhas dendríticas"""
        total_spines = [s['total_spines'] for s in self.spine_history]
        avg_size = [s['avg_spine_size'] for s in self.spine_history]
        
        plt.plot(total_spines, label='Total Spines')
        plt.plot(avg_size, label='Avg Size')
        plt.legend()
        
    def _plot_neuromod_evolution(self):
        """Plota evolução dos neuromoduladores"""
        data = np.array([[n[name] for name in self.neuromod_history[0].keys()] 
                        for n in self.neuromod_history])
        plt.imshow(data.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Concentration')
        plt.xlabel('Time Step')
        plt.ylabel('Neuromodulator')

if __name__ == "__main__":
    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Inicia simulação
    network = EnhancedNeuralNetwork(100)
    analyzer = NetworkAnalyzer(network)
    
    # Simula e analisa
    simulate_enhanced_network()