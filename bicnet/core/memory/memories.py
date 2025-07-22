import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
import logging
from datetime import datetime, timedelta
import pickle

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='memory_systems.log'
)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"

@dataclass
class MemoryTrace:
    """Representa um traço de memória"""
    content: np.ndarray
    creation_time: datetime
    strength: float = 1.0
    repetitions: int = 0
    emotional_value: float = 0.0
    context: Dict = None
    memory_type: MemoryType = MemoryType.SHORT_TERM

    def decay(self, rate: float):
        """Aplica decaimento temporal à força da memória"""
        self.strength *= (1 - rate)
        return self.strength

    def reinforce(self, amount: float):
        """Reforça a memória"""
        self.strength = min(1.0, self.strength + amount)
        self.repetitions += 1

class WorkingMemory:
    def __init__(self, capacity: int = 7):  # Miller's Law - 7±2 items
        self.capacity = capacity
        self.contents: Dict[str, MemoryTrace] = {}
        self.attention_focus: str = None
        
    def add_item(self, item_id: str, content: np.ndarray):
        """Adiciona item à memória de trabalho"""
        if len(self.contents) >= self.capacity:
            # Remove item mais antigo
            oldest_item = min(self.contents.items(), key=lambda x: x[1].creation_time)
            del self.contents[oldest_item[0]]
            
        self.contents[item_id] = MemoryTrace(
            content=content,
            creation_time=datetime.now(),
            memory_type=MemoryType.WORKING
        )
        self.attention_focus = item_id
        
    def refresh_items(self):
        """Atualiza força das memórias baseado em atenção"""
        current_time = datetime.now()
        for item_id, trace in list(self.contents.items()):
            time_diff = (current_time - trace.creation_time).total_seconds()
            
            # Maior decaimento para itens fora do foco de atenção
            decay_rate = 0.1 if item_id == self.attention_focus else 0.3
            
            if trace.decay(decay_rate * time_diff) < 0.2:
                del self.contents[item_id]

class LongTermMemory:
    def __init__(self):
        self.semantic: Dict[str, MemoryTrace] = {}
        self.episodic: Dict[str, MemoryTrace] = {}
        self.procedural: Dict[str, MemoryTrace] = {}
        
    def store(self, memory_id: str, trace: MemoryTrace):
        """Armazena memória no sistema apropriado"""
        if trace.memory_type == MemoryType.SEMANTIC:
            self.semantic[memory_id] = trace
        elif trace.memory_type == MemoryType.EPISODIC:
            self.episodic[memory_id] = trace
        elif trace.memory_type == MemoryType.PROCEDURAL:
            self.procedural[memory_id] = trace
            
    def consolidate(self, memory_id: str, trace: MemoryTrace):
        """Consolida memória de curto prazo em longo prazo"""
        # Fortalece com base em repetições e valor emocional
        consolidation_strength = 0.2 + (0.1 * trace.repetitions) + (0.2 * trace.emotional_value)
        trace.reinforce(consolidation_strength)
        
        # Determina tipo de memória baseado no contexto
        if trace.context and 'type' in trace.context:
            trace.memory_type = MemoryType(trace.context['type'])
        else:
            # Default para episódica se contexto não especificado
            trace.memory_type = MemoryType.EPISODIC
            
        self.store(memory_id, trace)

class MemorySystem:
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()
        self.consolidation_queue: List[Tuple[str, MemoryTrace]] = []
        self.hippocampus_activity = 0.0
        
    def process_new_information(self, info_id: str, content: np.ndarray, context: Dict = None):
        """Processa nova informação entrando no sistema"""
        # Adiciona à memória de trabalho
        self.working_memory.add_item(info_id, content)
        
        # Cria traço de memória
        trace = MemoryTrace(
            content=content,
            creation_time=datetime.now(),
            context=context or {},
            emotional_value=context.get('emotional_value', 0.0) if context else 0.0
        )
        
        # Adiciona à fila de consolidação
        self.consolidation_queue.append((info_id, trace))
        
    def consolidate_memories(self):
        """Consolida memórias da fila para longo prazo"""
        self.hippocampus_activity = min(1.0, self.hippocampus_activity + 0.2)
        
        for info_id, trace in self.consolidation_queue:
            if trace.strength > 0.4 or trace.emotional_value > 0.7:
                self.long_term_memory.consolidate(info_id, trace)
                logger.info(f"Consolidated memory {info_id} to long-term storage")
                
        self.consolidation_queue.clear()
        self.hippocampus_activity = max(0.0, self.hippocampus_activity - 0.1)
        
    def retrieve_memory(self, memory_id: str) -> Optional[MemoryTrace]:
        """Recupera memória de qualquer sistema"""
        # Tenta memória de trabalho primeiro
        if memory_id in self.working_memory.contents:
            trace = self.working_memory.contents[memory_id]
            self.working_memory.attention_focus = memory_id
            return trace
            
        # Procura em sistemas de longo prazo
        for memory_store in [
            self.long_term_memory.semantic,
            self.long_term_memory.episodic,
            self.long_term_memory.procedural
        ]:
            if memory_id in memory_store:
                trace = memory_store[memory_id]
                # Reconsolidação fortalece a memória
                trace.reinforce(0.1)
                return trace
                
        return None

class MemoryVisualizer:
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        
    def visualize_memory_state(self, timestamp: int):
        """Visualiza estado atual do sistema de memória"""
        plt.figure(figsize=(15, 10))
        
        # Working Memory
        plt.subplot(231)
        self._plot_working_memory()
        plt.title('Working Memory')
        
        # Long-term Memory Systems
        plt.subplot(232)
        self._plot_long_term_systems()
        plt.title('Long-term Memory Systems')
        
        # Consolidation Activity
        plt.subplot(233)
        self._plot_consolidation_activity()
        plt.title('Consolidation Activity')
        
        # Memory Strength Distribution
        plt.subplot(234)
        self._plot_memory_strengths()
        plt.title('Memory Strengths')
        
        # Hippocampus Activity
        plt.subplot(235)
        self._plot_hippocampus_activity()
        plt.title('Hippocampus Activity')
        
        plt.tight_layout()
        plt.savefig(f'memory_state_{timestamp}.png')
        plt.close()
        
    def _plot_working_memory(self):
        """Visualiza conteúdo da memória de trabalho"""
        items = list(self.memory_system.working_memory.contents.items())
        if items:
            strengths = [trace.strength for _, trace in items]
            labels = [id[:10] for id, _ in items]
            plt.bar(labels, strengths)
            plt.xticks(rotation=45)
            
    def _plot_long_term_systems(self):
        """Visualiza sistemas de memória de longo prazo"""
        systems = {
            'Semantic': len(self.memory_system.long_term_memory.semantic),
            'Episodic': len(self.memory_system.long_term_memory.episodic),
            'Procedural': len(self.memory_system.long_term_memory.procedural)
        }
        plt.bar(systems.keys(), systems.values())
        
    def _plot_consolidation_activity(self):
        """Visualiza atividade de consolidação"""
        queue_size = len(self.memory_system.consolidation_queue)
        plt.bar(['Queue Size'], [queue_size])
        
    def _plot_memory_strengths(self):
        """Visualiza distribuição de força das memórias"""
        all_strengths = []
        for memory_dict in [
            self.memory_system.working_memory.contents,
            self.memory_system.long_term_memory.semantic,
            self.memory_system.long_term_memory.episodic,
            self.memory_system.long_term_memory.procedural
        ]:
            all_strengths.extend([trace.strength for trace in memory_dict.values()])
            
        if all_strengths:
            plt.hist(all_strengths, bins=20)
            
    def _plot_hippocampus_activity(self):
        """Visualiza atividade do hipocampo"""
        plt.bar(['Hippocampus'], [self.memory_system.hippocampus_activity])
        plt.ylim(0, 1)

def simulate_memory_processes():
    """Simula processos de memória ao longo do tempo"""
    memory_system = MemorySystem()
    visualizer = MemoryVisualizer(memory_system)
    
    # Simula entrada de informações ao longo do tempo
    for i in range(20):
        # Gera informação sintética
        content = np.random.rand(10)
        context = {
            'emotional_value': np.random.random(),
            'type': np.random.choice([t.value for t in MemoryType])
        }
        
        # Processa nova informação
        memory_system.process_new_information(
            f'info_{i}',
            content,
            context
        )
        
        # Atualiza memória de trabalho
        memory_system.working_memory.refresh_items()
        
        # Periodicamente consolida memórias
        if i % 5 == 0:
            memory_system.consolidate_memories()
            
        # Visualiza estado
        visualizer.visualize_memory_state(i)
        
        logger.info(f"Completed simulation step {i}")

if __name__ == "__main__":
    simulate_memory_processes()