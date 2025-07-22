import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Protocol
from enum import Enum
import threading
import queue
import logging
import time
from collections import deque
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessLevel(Enum):
    UNCONSCIOUS = 0.0
    SUBCONSCIOUS = 0.3
    CONSCIOUS = 0.7
    SELF_AWARE = 1.0

@dataclass(frozen=True)  # Torna a classe imutável e hashável
class NeuralAssembly:
    """Representa uma assembleia neural que pode participar da consciência"""
    neurons: frozenset  # Usa frozenset em vez de set para ser hashável
    activation: float = 0.0
    coherence: float = 0.0
    stability: float = 0.0
    lifetime: int = 0
    
    def __post_init__(self):
        # Converte set para frozenset se necessário
        if isinstance(self.neurons, set):
            object.__setattr__(self, 'neurons', frozenset(self.neurons))
            
    def update(self, network_state: np.ndarray) -> 'NeuralAssembly':
        """Atualiza estado da assembleia baseado no estado da rede"""
        neurons_list = list(self.neurons)
        new_activation = np.mean(network_state[neurons_list])
        new_coherence = np.std(network_state[neurons_list])
        new_stability = 0.9 * self.stability + 0.1 * new_activation
        
        # Cria nova instância com valores atualizados
        return NeuralAssembly(
            neurons=self.neurons,
            activation=new_activation,
            coherence=new_coherence,
            stability=new_stability,
            lifetime=self.lifetime + 1
        )

@dataclass
class WorkingMemory:
    """Memória de trabalho com capacidade limitada"""
    capacity: int = 7
    items: Dict[str, float] = field(default_factory=dict)
    temporal_context: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def add_item(self, key: str, value: float):
        """Adiciona item, removendo o mais antigo se necessário"""
        if len(self.items) >= self.capacity:
            oldest = min(self.items.keys())
            del self.items[oldest]
        self.items[key] = value
        self.temporal_context.append((key, value))
    
    def get_context(self) -> List[Tuple[str, float]]:
        """Retorna contexto temporal recente"""
        return list(self.temporal_context)

class AttentionSystem:
    """Sistema de atenção que filtra e prioriza informações"""
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.attention_weights = np.ones(input_size) / input_size
        self.focus_strength = 0.5
        self.novelty_threshold = 0.3
        self.recent_inputs = deque(maxlen=100)
        
    def update_attention(self, input_data: np.ndarray, salience: np.ndarray):
        """Atualiza pesos de atenção baseado em saliência"""
        # Detecta novidade
        novelty = 1.0
        if self.recent_inputs:
            similarities = [np.corrcoef(input_data, prev)[0,1] 
                          for prev in self.recent_inputs]
            novelty = 1.0 - np.mean(similarities)
        
        # Atualiza pesos
        target_weights = F.softmax(torch.tensor(salience * novelty), dim=0).numpy()
        self.attention_weights = (1 - self.focus_strength) * self.attention_weights + \
                               self.focus_strength * target_weights
        
        self.recent_inputs.append(input_data)
        
    def filter_input(self, input_data: np.ndarray) -> np.ndarray:
        """Aplica filtro de atenção ao input"""
        return input_data * self.attention_weights

class GlobalWorkspace:
    """Implementa teoria do espaço global de trabalho"""
    def __init__(self, size: int):
        self.size = size
        self.content = np.zeros(size)
        self.access_threshold = 0.6
        self.active_assemblies: Set[NeuralAssembly] = set()
        self.competition_strength = 0.5
        
    def broadcast(self, input_data: np.ndarray, assemblies: Set[NeuralAssembly]) -> np.ndarray:
        """Transmite informação global baseado em competição entre assembleias"""
        # Competição entre assembleias
        for assembly in assemblies:
            if assembly.activation > self.access_threshold:
                self.active_assemblies.add(assembly)
        
        # Remove assembleias fracas
        self.active_assemblies = {a for a in self.active_assemblies 
                                if a.activation > self.access_threshold}
        
        # Calcula output
        output = np.zeros_like(self.content)
        if self.active_assemblies:
            weights = np.array([a.activation for a in self.active_assemblies])
            weights = F.softmax(torch.tensor(weights), dim=0).numpy()
            
            for assembly, weight in zip(self.active_assemblies, weights):
                output[list(assembly.neurons)] += weight
        
        self.content = (1 - self.competition_strength) * self.content + \
                      self.competition_strength * output
        
        return self.content

class ConsciousnessSystem:
    """Sistema principal de consciência"""
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.attention = AttentionSystem(input_size)
        self.working_memory = WorkingMemory()
        self.global_workspace = GlobalWorkspace(input_size)
        
        # Estado interno
        self.consciousness_level = ConsciousnessLevel.UNCONSCIOUS
        self.arousal = 0.5
        self.stability = 0.0
        
        # Neuromoduladores
        self.neuromodulators = {
            'dopamine': 1.0,
            'norepinephrine': 1.0,
            'serotonin': 1.0,
            'acetylcholine': 1.0
        }
        
        # Assembleias neurais
        self.assemblies: Dict[frozenset, NeuralAssembly] = {}  # Usa dict em vez de set
        self.assembly_threshold = 0.4
        
        # Processamento em background
        self._processing_queue = queue.Queue()
        self._running = True
        self._processing_thread = threading.Thread(target=self._background_processing)
        self._processing_thread.start()
    
    def detect_assemblies(self, network_state: np.ndarray) -> Dict[frozenset, NeuralAssembly]:
        """Detecta assembleias neurais emergentes"""
        new_assemblies = {}
        
        if len(network_state) < 3:
            return new_assemblies
        
        window_size = min(10, len(network_state) - 1)
        for i in range(0, len(network_state) - window_size + 1):
            window = network_state[i:i+window_size]
            
            if np.std(window) > 1e-6:
                window_2d = window.reshape(-1, 1)
                distances = np.abs(window_2d - window_2d.T)
                
                similar_pairs = np.argwhere(distances < self.assembly_threshold)
                
                if len(similar_pairs) > 2:
                    neurons_in_assembly = set()
                    for pair in similar_pairs:
                        neurons_in_assembly.add(i + pair[0])
                        neurons_in_assembly.add(i + pair[1])
                    
                    if len(neurons_in_assembly) > 2:
                        assembly = NeuralAssembly(frozenset(neurons_in_assembly))
                        new_assemblies[assembly.neurons] = assembly
        
        return new_assemblies
    
    def update_neuromodulators(self, arousal_delta: float):
        """Atualiza níveis de neuromoduladores"""
        # Norepinefrina relacionada com arousal
        self.neuromodulators['norepinephrine'] = np.clip(
            self.neuromodulators['norepinephrine'] + 0.1 * arousal_delta,
            0.0, 2.0
        )
        
        # Dopamina relacionada com recompensa/novidade
        self.neuromodulators['dopamine'] *= 0.95  # Decaimento natural
        
        # Serotonina relacionada com estabilidade
        self.neuromodulators['serotonin'] = np.clip(
            self.neuromodulators['serotonin'] + 0.1 * (self.stability - 0.5),
            0.0, 2.0
        )
        
        # Acetilcolina relacionada com atenção
        self.neuromodulators['acetylcholine'] = np.clip(
            self.neuromodulators['acetylcholine'] + 0.1 * (self.attention.focus_strength - 0.5),
            0.0, 2.0
        )
    
    def determine_consciousness_level(self) -> ConsciousnessLevel:
        """Determina nível de consciência baseado no estado do sistema"""
        # Fatores que influenciam o nível de consciência
        factors = {
            'arousal': self.arousal,
            'workspace_activity': np.mean(self.global_workspace.content),
            'attention': np.max(self.attention.attention_weights),
            'neuromodulation': np.mean(list(self.neuromodulators.values())),
            'assembly_coherence': np.mean([a.coherence for a in self.assemblies]) 
                                if self.assemblies else 0.0
        }
        
        # Calcula nível geral
        consciousness_value = np.mean(list(factors.values()))
        
        # Determina nível discreto
        if consciousness_value < 0.3:
            return ConsciousnessLevel.UNCONSCIOUS
        elif consciousness_value < 0.5:
            return ConsciousnessLevel.SUBCONSCIOUS
        elif consciousness_value < 0.8:
            return ConsciousnessLevel.CONSCIOUS
        else:
            return ConsciousnessLevel.SELF_AWARE
    
    def process_input(self, input_data: np.ndarray) -> Tuple[np.ndarray, ConsciousnessLevel]:
        """Processa input através do sistema de consciência"""
        if not np.isfinite(input_data).all():
            raise ValueError("Invalid input data detected")
            
        # Normaliza input
        input_data = (input_data - np.mean(input_data)) / (np.std(input_data) + 1e-6)
        
        self._processing_queue.put(input_data)
        
        return self.global_workspace.content, self.consciousness_level
    
    def _background_processing(self):
        """Processamento contínuo em background"""
        while self._running:
            try:
                input_data = self._processing_queue.get(timeout=1.0)
                
                # Normaliza e suaviza input
                input_smooth = np.convolve(input_data, np.ones(5)/5, mode='same')
                input_norm = (input_smooth - np.mean(input_smooth)) / (np.std(input_smooth) + 1e-6)
                
                # Calcula saliência
                salience = np.abs(input_norm - np.mean(input_norm))
                salience = np.convolve(salience, np.ones(5)/5, mode='same')
                
                # Atualiza atenção
                self.attention.update_attention(input_norm, salience)
                filtered_input = self.attention.filter_input(input_norm)
                
                # Detecta novas assembleias
                new_assemblies = self.detect_assemblies(filtered_input)
                
                # Atualiza assembleias existentes e adiciona novas
                updated_assemblies = {}
                for neurons, assembly in {**self.assemblies, **new_assemblies}.items():
                    updated = assembly.update(filtered_input)
                    if updated.stability > 0.2:  # Remove instáveis
                        updated_assemblies[neurons] = updated
                
                self.assemblies = updated_assemblies
                
                # Processa no espaço global
                processed_output = self.global_workspace.broadcast(
                    filtered_input, set(self.assemblies.values())
                )
                
                # Atualiza estado interno
                self.arousal = np.clip(np.mean(np.abs(processed_output)), 0, 1)
                self.stability = np.clip(1.0 - np.std(processed_output), 0, 1)
                
                # Atualiza neuromoduladores
                self.update_neuromodulators(self.arousal - 0.5)
                
                # Atualiza nível de consciência
                self.consciousness_level = self.determine_consciousness_level()
                
                # Armazena na memória de trabalho
                activation_level = float(np.mean(processed_output))
                self.working_memory.add_item(
                    f"state_{len(self.working_memory.items)}", 
                    activation_level
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
    
    def cleanup(self):
        """Limpa recursos do sistema"""
        self._running = False
        self._processing_thread.join()
        logger.info("Consciousness system shutdown complete")

# Exemplo de uso
def create_consciousness_system(input_size: int = 1000) -> ConsciousnessSystem:
    """Cria e inicializa sistema de consciência"""
    system = ConsciousnessSystem(input_size)
    logger.info(f"Created consciousness system with input size {input_size}")
    return system

if __name__ == "__main__":
    # Configuração de logging mais detalhada
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Exemplo de simulação
    system = create_consciousness_system(100)
    
    try:
        for i in range(100):
            # Cria padrão mais complexo
            t = np.linspace(0, 4*np.pi, 100)
            pattern = (np.sin(t) + 
                      0.5 * np.sin(2*t + np.pi/4) +  # Defasagem
                      0.25 * np.sin(4*t + np.pi/3))  # Mais defasagem
            
            # Adiciona ruído variável
            noise_level = 0.1 * (1 + np.sin(i/10))  # Ruído variável
            noise = np.random.randn(100) * noise_level
            input_data = pattern + noise
            
            # Processa input
            output, level = system.process_input(input_data)
            
            logger.info(f"\nStep {i}:")
            logger.info(f"Consciousness level: {level}")
            logger.info(f"Average output: {np.mean(output):.3f}")
            logger.info(f"Number of assemblies: {len(system.assemblies)}")
            logger.info(f"Arousal level: {system.arousal:.3f}")
            logger.info(f"Stability: {system.stability:.3f}")
            
            # Pausa mais curta
            time.sleep(0.05)
            
    finally:
        system.cleanup()