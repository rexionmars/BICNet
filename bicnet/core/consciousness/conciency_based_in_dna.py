from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any, Optional
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class Decision:
    """Estrutura de decisão"""
    options: List[str]
    values: Dict[str, float]
    context: Dict[str, Any]
    confidence: float
    emotional_influence: float

class DecisionMakingSystem:
    def __init__(self, consciousness_system: ConsciousnessSystem):
        self.consciousness = consciousness_system
        self.past_decisions: List[Decision] = []
        self.value_system = {
            'survival': 1.0,
            'curiosity': 0.7,
            'social': 0.6,
            'achievement': 0.5,
            'conservation': 0.4
        }
        self.goal_hierarchy = {
            'primary': ['survival', 'energy_conservation'],
            'secondary': ['learning', 'exploration'],
            'tertiary': ['optimization', 'social_interaction']
        }
        
    def evaluate_option(self, option: str, context: Dict[str, Any]) -> float:
        """Avalia uma opção baseada no sistema de valores e contexto"""
        value = 0.0
        
        # Influência do estado consciente
        consciousness_factor = self.consciousness.consciousness_level
        
        # Influência emocional
        emotional_factor = self.consciousness.emotional_state
        
        # Influência da memória
        memory_influence = 0.0
        for key, memory_value in self.consciousness.memory.long_term.items():
            if option in key:
                memory_influence += memory_value
        
        # Avaliação baseada nos valores do sistema
        for value_name, value_weight in self.value_system.items():
            if value_name in context:
                value += context[value_name] * value_weight
        
        # Combinação de todos os fatores
        final_value = (value * 0.4 + 
                      consciousness_factor * 0.3 +
                      emotional_factor * 0.2 +
                      memory_influence * 0.1)
        
        return final_value
    
    def make_decision(self, options: List[str], context: Dict[str, Any]) -> Decision:
        """Toma uma decisão baseada nas opções disponíveis e contexto"""
        values = {}
        for option in options:
            values[option] = self.evaluate_option(option, context)
        
        # Normaliza os valores
        total = sum(values.values())
        if total > 0:
            values = {k: v/total for k, v in values.items()}
        
        # Adiciona ruído baseado no nível de consciência
        consciousness_noise = 1.0 - self.consciousness.consciousness_level
        for option in values:
            values[option] += np.random.normal(0, consciousness_noise)
        
        # Escolhe a melhor opção
        best_option = max(values.items(), key=lambda x: x[1])
        confidence = best_option[1]
        
        decision = Decision(
            options=options,
            values=values,
            context=context,
            confidence=confidence,
            emotional_influence=self.consciousness.emotional_state
        )
        
        # Armazena a decisão na memória
        self.past_decisions.append(decision)
        self.consciousness.memory.short_term[f"decision_{len(self.past_decisions)}"] = confidence
        
        return decision
    
    def learn_from_outcome(self, decision: Decision, outcome: float):
        """Aprende com o resultado da decisão"""
        # Atualiza o sistema de valores
        for value_name in decision.context:
            if value_name in self.value_system:
                self.value_system[value_name] *= (1.0 + outcome * 0.1)
        
        # Consolida na memória de longo prazo
        decision_key = f"outcome_{len(self.past_decisions)}"
        self.consciousness.memory.long_term[decision_key] = outcome
        
        # Ajusta estado emocional baseado no resultado
        self.consciousness.emotional_state = 0.7 * self.consciousness.emotional_state + 0.3 * outcome

    def generate_response(self, input_data: np.ndarray) -> str:
        """Gera uma resposta baseada no input e estado interno"""
        # Processa o input através do sistema de consciência
        output, consciousness_level = self.consciousness.process_input(input_data)
        
        # Prepara o contexto para tomada de decisão
        context = {
            'input_strength': np.mean(np.abs(input_data)),
            'consciousness': consciousness_level.value,
            'emotional_state': self.consciousness.emotional_state,
            'energy': 1.0 - self.consciousness.internal_state['fatigue'],
            'stress': self.consciousness.internal_state['stress']
        }
        
        # Define possíveis respostas baseadas no estado
        if consciousness_level == ConsciousnessLevel.SELF_AWARE:
            options = ['explore', 'analyze', 'create', 'communicate']
        elif consciousness_level == ConsciousnessLevel.CONSCIOUS:
            options = ['respond', 'observe', 'learn', 'rest']
        else:
            options = ['wait', 'rest', 'minimal_response']
        
        # Toma uma decisão
        decision = self.make_decision(options, context)
        
        # Gera resposta baseada na decisão
        response = self.execute_decision(decision)
        
        return response
    
    def execute_decision(self, decision: Decision) -> str:
        """Executa a decisão tomada e gera uma resposta apropriada"""
        chosen_option = max(decision.values.items(), key=lambda x: x[1])[0]
        
        responses = {
            'explore': 'Investigating new possibilities and gathering information.',
            'analyze': 'Processing and analyzing current situation.',
            'create': 'Generating new ideas or solutions.',
            'communicate': 'Engaging in meaningful interaction.',
            'respond': 'Providing appropriate response to input.',
            'observe': 'Monitoring and learning from environment.',
            'learn': 'Incorporating new information into memory.',
            'rest': 'Conserving energy and processing memories.',
            'wait': 'Waiting for more clear input or context.',
            'minimal_response': 'Acknowledging input with minimal processing.'
        }
        
        return responses.get(chosen_option, 'Processing...')

class Gene(Enum):
    LEARNING_RATE = "learning_rate"
    EMOTIONAL_SENSITIVITY = "emotional_sensitivity"
    MEMORY_CAPACITY = "memory_capacity"
    CONSCIOUSNESS_THRESHOLD = "consciousness_threshold"
    NEUROTRANSMITTER_PRODUCTION = "neurotransmitter_production"
    SYNAPTIC_PLASTICITY = "synaptic_plasticity"
    DENDRITE_COMPLEXITY = "dendrite_complexity"
    NEURAL_RESILIENCE = "neural_resilience"
    ENERGY_EFFICIENCY = "energy_efficiency"
    PATTERN_RECOGNITION = "pattern_recognition"

@dataclass
class GeneticTrait:
    """Representa um traço genético específico"""
    gene: Gene
    value: float
    dominance: float
    mutation_rate: float
    expression_level: float
    
class DNAStructure:
    def __init__(self, chromosome_pairs: int = 23):
        self.chromosome_pairs = chromosome_pairs
        self.chromosomes = self._initialize_chromosomes()
        
    def _initialize_chromosomes(self) -> Dict[int, List[Dict[Gene, GeneticTrait]]]:
        """Inicializa os cromossomos com genes aleatórios"""
        chromosomes = {}
        
        for pair in range(self.chromosome_pairs):
            # Cada par tem dois cromossomos (materno e paterno)
            chromosomes[pair] = [
                self._create_chromosome(),
                self._create_chromosome()
            ]
            
        return chromosomes
    
    def _create_chromosome(self) -> Dict[Gene, GeneticTrait]:
        """Cria um único cromossomo com genes"""
        chromosome = {}
        
        for gene in Gene:
            # Cria traço genético com valores aleatórios
            trait = GeneticTrait(
                gene=gene,
                value=random.uniform(0, 1),
                dominance=random.uniform(0, 1),
                mutation_rate=random.uniform(0.001, 0.01),
                expression_level=random.uniform(0, 1)
            )
            chromosome[gene] = trait
            
        return chromosome
    
    def mutate(self, mutation_probability: float = 0.01):
        """Aplica mutações aleatórias ao DNA"""
        for pair in self.chromosomes.values():
            for chromosome in pair:
                for trait in chromosome.values():
                    if random.random() < trait.mutation_rate:
                        # Aplica mutação
                        trait.value += random.gauss(0, 0.1)
                        trait.value = max(0, min(1, trait.value))
                        
                        # Possível mutação na taxa de expressão
                        if random.random() < mutation_probability:
                            trait.expression_level += random.gauss(0, 0.1)
                            trait.expression_level = max(0, min(1, trait.expression_level))

class ConsciousnessDNA:
    def __init__(self, dna_structure: Optional[DNAStructure] = None):
        self.dna = dna_structure or DNAStructure()
        self.expressed_traits = self._express_genes()
        
    def _express_genes(self) -> Dict[Gene, float]:
        """Expressa os genes baseado na dominância e nível de expressão"""
        expressed_traits = {}
        
        for gene in Gene:
            # Coleta todos os alelos (versões) do gene
            alleles = []
            for pair in self.dna.chromosomes.values():
                for chromosome in pair:
                    if gene in chromosome:
                        trait = chromosome[gene]
                        alleles.append(trait)
            
            # Calcula a expressão final do gene
            if alleles:
                # Média ponderada baseada na dominância e nível de expressão
                total_influence = sum(a.dominance * a.expression_level for a in alleles)
                if total_influence > 0:
                    expressed_value = sum(
                        a.value * a.dominance * a.expression_level 
                        for a in alleles
                    ) / total_influence
                else:
                    expressed_value = sum(a.value for a in alleles) / len(alleles)
                
                expressed_traits[gene] = expressed_value
                
        return expressed_traits
    
    def create_consciousness_system(self, input_size: int) -> ConsciousnessSystem:
        """Cria um sistema de consciência baseado no DNA expresso"""
        # Configura parâmetros baseados nos genes
        config = {
            'learning_rate': self.expressed_traits[Gene.LEARNING_RATE] * 0.1,
            'consciousness_threshold': self.expressed_traits[Gene.CONSCIOUSNESS_THRESHOLD] * 0.5 + 0.3,
            'emotional_learning_rate': self.expressed_traits[Gene.EMOTIONAL_SENSITIVITY] * 0.1,
            'membrane_rest': -70.0 * (1 + self.expressed_traits[Gene.NEURAL_RESILIENCE] * 0.2),
            'threshold': -55.0 * (1 + self.expressed_traits[Gene.PATTERN_RECOGNITION] * 0.2),
            'refractory_period': int(5 * (1 + self.expressed_traits[Gene.ENERGY_EFFICIENCY] * 0.5))
        }
        
        # Cria o sistema de consciência
        consciousness = ConsciousnessSystem(
            input_size=input_size,
            **config
        )
        
        # Modifica sinapses baseado nos genes
        self._modify_synapses(consciousness)
        
        return consciousness
    
    def _modify_synapses(self, consciousness: ConsciousnessSystem):
        """Modifica as sinapses baseado nos genes"""
        plasticity_factor = self.expressed_traits[Gene.SYNAPTIC_PLASTICITY]
        neurotransmitter_factor = self.expressed_traits[Gene.NEUROTRANSMITTER_PRODUCTION]
        
        for synapse in consciousness.synapses.values():
            # Modifica plasticidade
            synapse.plasticity *= (1 + plasticity_factor)
            
            # Modifica produção de neurotransmissores
            synapse.neurotransmitter *= (1 + neurotransmitter_factor)

class GeneticConsciousness:
    def __init__(self, input_size: int):
        # Cria estrutura DNA inicial
        self.dna = ConsciousnessDNA()
        
        # Cria sistema de consciência baseado no DNA
        self.consciousness = self.dna.create_consciousness_system(input_size)
        
        # Sistema de decisão
        self.decision_system = DecisionMakingSystem(self.consciousness)
        
        # Registro de fitness
        self.fitness_history = []
        
    def evolve(self, fitness_score: float, mutation_rate: float = 0.01):
        """Evolui o sistema baseado no fitness"""
        self.fitness_history.append(fitness_score)
        
        # Aplica mutações se o fitness não estiver melhorando
        if len(self.fitness_history) > 10:
            recent_fitness = np.mean(self.fitness_history[-10:])
            if recent_fitness < np.mean(self.fitness_history[-20:-10]):
                self.dna.dna.mutate(mutation_rate)
                # Recria sistema com novo DNA
                self.consciousness = self.dna.create_consciousness_system(
                    len(self.consciousness.synapses)
                )
    
    def process_input(self, input_data: np.ndarray) -> str:
        """Processa input e gera resposta"""
        return self.decision_system.generate_response(input_data)
    
    def get_genetic_report(self) -> Dict[str, float]:
        """Retorna relatório dos traços genéticos expressos"""
        return {
            gene.name: value 
            for gene, value in self.dna.expressed_traits.items()
        }

# Exemplo de uso:
def create_evolving_consciousness(input_size: int = 100) -> GeneticConsciousness:
    """Cria uma consciência evolutiva"""
    return GeneticConsciousness(input_size)

def train_consciousness(consciousness: GeneticConsciousness, 
                       generations: int = 100,
                       inputs_per_generation: int = 1000):
    """Treina a consciência através de gerações"""
    for generation in range(generations):
        total_fitness = 0
        
        for _ in range(inputs_per_generation):
            # Gera input aleatório
            input_data = np.random.random(100)
            
            # Processa input
            response = consciousness.process_input(input_data)
            
            # Avalia resultado (exemplo simples)
            fitness = random.random()  # Substitua por sua métrica real
            total_fitness += fitness
            
        # Evolui baseado no fitness médio
        average_fitness = total_fitness / inputs_per_generation
        consciousness.evolve(average_fitness)
        
        if generation % 10 == 0:
            print(f"Geração {generation}: Fitness médio = {average_fitness:.3f}")
            print("Relatório genético:", consciousness.get_genetic_report())

if __name__ == "__main__":
    train_consciousness()