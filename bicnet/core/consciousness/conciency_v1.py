import numpy as np
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple
from enum import Enum
from pathlib import Path
import json
from datetime import datetime

# Configuração de logging
log_dir = Path("consciousness_logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"consciousness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ConsciousnessLevel(Enum):
    UNCONSCIOUS = 0
    SUBCONSCIOUS = 1
    CONSCIOUS = 2
    SELF_AWARE = 3

@dataclass
class EmotionalState:
    valence: float = 0.0  # Positivo/negativo
    arousal: float = 0.0  # Nível de ativação
    dominance: float = 0.0  # Controle/submissão
    
    def update(self, stimulus: np.ndarray):
        # Atualiza estado emocional baseado em estímulo
        self.valence = np.clip(self.valence + np.mean(stimulus) * 0.1, -1, 1)
        self.arousal = np.clip(self.arousal + np.std(stimulus) * 0.1, 0, 1)
        self.dominance = np.clip(self.dominance + np.max(stimulus) * 0.1, 0, 1)
        
        logger.debug(f"Emotional state updated - valence: {self.valence:.2f}, "
                    f"arousal: {self.arousal:.2f}, dominance: {self.dominance:.2f}")

@dataclass
class Memory:
    short_term: Dict[str, np.ndarray] = field(default_factory=dict)
    long_term: Dict[str, np.ndarray] = field(default_factory=dict)
    working: Dict[str, np.ndarray] = field(default_factory=dict)
    capacity: int = 1000
    
    def store(self, key: str, data: np.ndarray, memory_type: str = "short_term"):
        """Armazena informação na memória"""
        if memory_type == "short_term":
            self.short_term[key] = data
            if len(self.short_term) > self.capacity:
                oldest = min(self.short_term.keys())
                self.short_term.pop(oldest)
                
        elif memory_type == "long_term":
            self.long_term[key] = data
            
        elif memory_type == "working":
            self.working[key] = data
            if len(self.working) > 10:  # Limite menor para memória de trabalho
                oldest = min(self.working.keys())
                self.working.pop(oldest)
                
        logger.debug(f"Stored memory in {memory_type}: {key}")
        
    def consolidate(self):
        """Consolida memórias de curto prazo para longo prazo"""
        for key, data in self.short_term.items():
            if np.random.random() < 0.1:  # 10% chance de consolidação
                self.long_term[f"consolidated_{key}"] = data
                logger.info(f"Consolidated memory: {key}")

@dataclass
class AttentionalFocus:
    target: Optional[str] = None
    intensity: float = 0.0
    duration: float = 0.0
    
    def update(self, stimulus: np.ndarray):
        """Atualiza foco atencional"""
        # Calcula saliência do estímulo
        salience = np.max(np.abs(stimulus))
        
        # Atualiza intensidade com decaimento
        self.intensity = np.clip(
            0.8 * self.intensity + 0.2 * salience,
            0, 1
        )
        
        # Atualiza duração
        if salience > 0.5:
            self.duration += 1
        else:
            self.duration *= 0.9
            
        logger.debug(f"Attention updated - intensity: {self.intensity:.2f}, "
                    f"duration: {self.duration:.2f}")

@dataclass
class ConsciousnessState:
    level: ConsciousnessLevel = ConsciousnessLevel.UNCONSCIOUS
    attention: AttentionalFocus = field(default_factory=AttentionalFocus)
    emotion: EmotionalState = field(default_factory=EmotionalState)
    memory: Memory = field(default_factory=Memory)
    arousal: float = 0.5
    awareness: float = 0.0
    processing_queue: queue.Queue = field(default_factory=queue.Queue)
    
    def to_dict(self) -> Dict:
        """Converte estado para dicionário"""
        return {
            'level': self.level.name,
            'arousal': self.arousal,
            'awareness': self.awareness,
            'attention': {
                'intensity': self.attention.intensity,
                'duration': self.attention.duration
            },
            'emotion': {
                'valence': self.emotion.valence,
                'arousal': self.emotion.arousal,
                'dominance': self.emotion.dominance
            }
        }

class ConsciousnessSystem:
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.state = ConsciousnessState()
        self._running = True
        self.state_history: List[Dict] = []
        self.processing_interval = 0.1  # 100ms
        
        # Inicia thread de processamento
        self._processing_thread = threading.Thread(
            target=self._background_processing
        )
        self._processing_thread.start()
        
        logger.info("Consciousness system initialized")
        
    def process_input(self, data: np.ndarray) -> Tuple[ConsciousnessLevel, float]:
        """Processa input e retorna nível de consciência e awareness"""
        if not np.isfinite(data).all():
            logger.error("Invalid input data detected")
            raise ValueError("Invalid input data")
            
        # Adiciona input para processamento
        self.state.processing_queue.put(data)
        
        return self.state.level, self.state.awareness
        
    def _background_processing(self):
        """Processamento contínuo em background"""
        while self._running:
            try:
                # Processa dados na fila
                if not self.state.processing_queue.empty():
                    data = self.state.processing_queue.get(timeout=1.0)
                    self._update_consciousness(data)
                    
                # Consolida memórias periodicamente
                if np.random.random() < 0.01:  # 1% chance por ciclo
                    self.state.memory.consolidate()
                    
                # Registra estado
                self.state_history.append(self.state.to_dict())
                if len(self.state_history) > 1000:
                    self.state_history.pop(0)
                    
                time.sleep(self.processing_interval)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in background processing: {str(e)}")
                
    def _update_consciousness(self, data: np.ndarray):
        """Atualiza estado de consciência baseado no input"""
        try:
            # Atualiza subsistemas
            self.state.attention.update(data)
            self.state.emotion.update(data)
            
            # Calcula nível de awareness
            self.state.awareness = (
                self.state.attention.intensity * 0.4 +
                self.state.emotion.arousal * 0.3 +
                self.state.arousal * 0.3
            )
            
            # Determina nível de consciência
            if self.state.awareness < 0.2:
                new_level = ConsciousnessLevel.UNCONSCIOUS
            elif self.state.awareness < 0.4:
                new_level = ConsciousnessLevel.SUBCONSCIOUS
            elif self.state.awareness < 0.7:
                new_level = ConsciousnessLevel.CONSCIOUS
            else:
                new_level = ConsciousnessLevel.SELF_AWARE
                
            # Registra mudança de nível
            if new_level != self.state.level:
                logger.info(f"Consciousness level changed: {self.state.level.name} -> {new_level.name}")
                self.state.level = new_level
                
            # Armazena na memória de trabalho
            self.state.memory.store(
                f"state_{time.time()}",
                data,
                "working"
            )
            
            logger.debug(f"Consciousness updated - level: {self.state.level.name}, "
                        f"awareness: {self.state.awareness:.2f}")
                        
        except Exception as e:
            logger.error(f"Error updating consciousness: {str(e)}")
            
    def save_state(self, filename: str):
        """Salva estado atual e histórico"""
        state_data = {
            'current_state': self.state.to_dict(),
            'history': self.state_history
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2)
        logger.info(f"State saved to {filename}")
            
    def cleanup(self):
        """Limpa recursos e finaliza processamento"""
        self._running = False
        self._processing_thread.join()
        logger.info("Consciousness system cleaned up")

def test_consciousness_system():
    """Função de teste do sistema"""
    system = ConsciousnessSystem(input_size=100)
    logger.info("Starting consciousness system test")
    
    try:
        # Simula diferentes inputs
        for i in range(100):
            # Gera input com diferentes características
            if i < 30:
                # Input calmo
                data = np.random.normal(0, 0.1, 100)
            elif i < 60:
                # Input mais intenso
                data = np.random.normal(0, 0.5, 100)
            else:
                # Input muito ativo
                data = np.random.normal(0, 1.0, 100)
                
            level, awareness = system.process_input(data)
            
            if i % 10 == 0:
                logger.info(f"Test iteration {i}: Level={level.name}, "
                          f"Awareness={awareness:.2f}")
                
            time.sleep(0.1)
            
        # Salva estado final
        system.save_state("consciousness_test_results.json")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
    finally:
        system.cleanup()
        logger.info("Test completed")

if __name__ == "__main__":
    test_consciousness_system()