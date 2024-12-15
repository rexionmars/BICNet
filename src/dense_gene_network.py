import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dense_gene_network.log'
)
logger = logging.getLogger(__name__)

@dataclass
class GeneRegulationPath:
    """Representa uma via de regulação gênica"""
    name: str
    activator_genes: Set[str]
    target_genes: Set[str]
    strength: float = 1.0
    threshold: float = 0.5

class Gene:
    def __init__(self, name: str, category: str, subcategory: str):
        self.name = name
        self.category = category
        self.subcategory = subcategory
        self.expression_level = 0.0
        self.baseline_expression = 0.1
        self.is_active = False
        self.activation_threshold = 0.5
        self.regulators: Set[str] = set()
        self.targets: Set[str] = set()
        self.activity_history = []
        
    def activate(self, stimulus: float, regulatory_input: float = 0.0):
        """Ativa o gene baseado no estímulo e regulação"""
        total_input = stimulus + regulatory_input
        prev_level = self.expression_level
        self.expression_level = np.tanh(total_input + self.baseline_expression)
        self.is_active = self.expression_level > self.activation_threshold
        self.activity_history.append(self.expression_level)
        
        change = abs(self.expression_level - prev_level)
        logger.debug(f"Gene {self.name} - Level: {self.expression_level:.3f}, Change: {change:.3f}")
        return change

class DenseGeneNetwork:
    def __init__(self):
        self.genes: Dict[str, Gene] = {}
        self.regulation_paths: List[GeneRegulationPath] = []
        self.activation_history = []
        self._initialize_dense_network()
        
    def _initialize_dense_network(self):
        """Inicializa uma rede densa de genes do cérebro do rato"""
        # Genes de desenvolvimento neural
        self._add_development_genes()
        
        # Genes de neurotransmissores e receptores
        self._add_neurotransmitter_genes()
        
        # Genes de canais iônicos
        self._add_ion_channel_genes()
        
        # Genes de fatores de crescimento
        self._add_growth_factor_genes()
        
        # Genes de plasticidade sináptica
        self._add_plasticity_genes()
        
        # Genes reguladores do ciclo circadiano
        self._add_circadian_genes()
        
        # Estabelece conexões regulatórias
        self._establish_regulatory_networks()
        
        logger.info(f"Initialized dense network with {len(self.genes)} genes")

    def _add_development_genes(self):
        """Adiciona genes de desenvolvimento neural"""
        dev_genes = {
            'PAX6': 'Neurogenesis',
            'EMX2': 'Cortical_development',
            'FOXG1': 'Forebrain_development',
            'NKX2.1': 'Neural_patterning',
            'NGN1': 'Neural_differentiation',
            'NGN2': 'Neural_differentiation',
            'ASCL1': 'Neural_specification',
            'DLX1': 'Interneuron_development',
            'DLX2': 'Interneuron_development',
            'LHX6': 'Interneuron_migration',
            'SOX2': 'Neural_stem_cells',
            'NEUROD1': 'Neural_differentiation',
            'NEUROG2': 'Neurogenesis'
        }
        
        for name, subcat in dev_genes.items():
            self.genes[name] = Gene(name, 'Development', subcat)

    def _add_neurotransmitter_genes(self):
        """Adiciona genes relacionados a neurotransmissores"""
        nt_genes = {
            'SNAP25': 'Synaptic_release',
            'SYN1': 'Synaptic_vesicles',
            'GAD1': 'GABA_synthesis',
            'GAD2': 'GABA_synthesis',
            'SLC6A4': 'Serotonin_transport',
            'TH': 'Dopamine_synthesis',
            'DBH': 'Norepinephrine_synthesis',
            'TPH2': 'Serotonin_synthesis',
            'SLC1A1': 'Glutamate_transport',
            'VGLUT1': 'Glutamate_transport',
            'VGLUT2': 'Glutamate_transport',
            'CHAT': 'Acetylcholine_synthesis'
        }
        
        for name, subcat in nt_genes.items():
            self.genes[name] = Gene(name, 'Neurotransmitter', subcat)

    def _add_ion_channel_genes(self):
        """Adiciona genes de canais iônicos"""
        channel_genes = {
            'SCN1A': 'Sodium_channel',
            'SCN2A': 'Sodium_channel',
            'KCNA1': 'Potassium_channel',
            'KCNQ2': 'Potassium_channel',
            'CACNA1A': 'Calcium_channel',
            'CACNA1B': 'Calcium_channel',
            'HCN1': 'Hyperpolarization_channel',
            'GRIN1': 'NMDA_receptor',
            'GRIN2A': 'NMDA_receptor',
            'GRIA1': 'AMPA_receptor'
        }
        
        for name, subcat in channel_genes.items():
            self.genes[name] = Gene(name, 'Ion_Channel', subcat)

    def _add_growth_factor_genes(self):
        """Adiciona genes de fatores de crescimento"""
        growth_genes = {
            'BDNF': 'Neurotrophin',
            'NGF': 'Neurotrophin',
            'NT3': 'Neurotrophin',
            'GDNF': 'Glial_growth_factor',
            'IGF1': 'Insulin_like_growth_factor',
            'FGF2': 'Fibroblast_growth_factor',
            'EGF': 'Epidermal_growth_factor'
        }
        
        for name, subcat in growth_genes.items():
            self.genes[name] = Gene(name, 'Growth_Factor', subcat)

    def _add_plasticity_genes(self):
        """Adiciona genes relacionados à plasticidade sináptica"""
        plasticity_genes = {
            'ARC': 'Immediate_early',
            'C_FOS': 'Immediate_early',
            'NPAS4': 'Activity_dependent',
            'CREB1': 'Transcription_factor',
            'CAMK2A': 'Kinase',
            'HOMER1': 'Scaffold_protein',
            'PSD95': 'Scaffold_protein'
        }
        
        for name, subcat in plasticity_genes.items():
            self.genes[name] = Gene(name, 'Plasticity', subcat)

    def _add_circadian_genes(self):
        """Adiciona genes do ritmo circadiano"""
        circadian_genes = {
            'PER1': 'Core_clock',
            'PER2': 'Core_clock',
            'CLOCK': 'Core_clock',
            'BMAL1': 'Core_clock',
            'CRY1': 'Core_clock',
            'CRY2': 'Core_clock'
        }
        
        for name, subcat in circadian_genes.items():
            self.genes[name] = Gene(name, 'Circadian', subcat)

    def _establish_regulatory_networks(self):
        """Estabelece redes regulatórias entre genes"""
        # Desenvolvimento cortical
        self.regulation_paths.append(GeneRegulationPath(
            'Cortical_Development',
            {'PAX6', 'EMX2', 'FOXG1'},
            {'NGN1', 'NGN2', 'NEUROD1'}
        ))
        
        # Diferenciação de interneurônios
        self.regulation_paths.append(GeneRegulationPath(
            'Interneuron_Development',
            {'DLX1', 'DLX2', 'LHX6'},
            {'GAD1', 'GAD2'}
        ))
        
        # Sinaptogênese
        self.regulation_paths.append(GeneRegulationPath(
            'Synaptogenesis',
            {'BDNF', 'SNAP25', 'SYN1'},
            {'PSD95', 'HOMER1'}
        ))
        
        # Plasticidade atividade-dependente
        self.regulation_paths.append(GeneRegulationPath(
            'Activity_Dependent_Plasticity',
            {'NPAS4', 'CREB1'},
            {'ARC', 'C_FOS', 'BDNF'}
        ))
        
        # Regulação circadiana
        self.regulation_paths.append(GeneRegulationPath(
            'Circadian_Regulation',
            {'CLOCK', 'BMAL1'},
            {'PER1', 'PER2', 'CRY1', 'CRY2'}
        ))

        # Estabelece conexões entre genes
        for path in self.regulation_paths:
            for activator in path.activator_genes:
                for target in path.target_genes:
                    if activator in self.genes and target in self.genes:
                        self.genes[activator].targets.add(target)
                        self.genes[target].regulators.add(activator)

    def activate_pathway(self, pathway_name: str, stimulus: float):
        """Ativa uma via específica na rede"""
        changes = []
        
        # Encontra a via
        target_path = None
        for path in self.regulation_paths:
            if path.name == pathway_name:
                target_path = path
                break
        
        if target_path:
            # Ativa genes reguladores
            for gene_name in target_path.activator_genes:
                if gene_name in self.genes:
                    change = self.genes[gene_name].activate(stimulus)
                    changes.append(change)
            
            # Propaga ativação para genes alvo
            for gene_name in target_path.target_genes:
                if gene_name in self.genes:
                    regulatory_input = sum(self.genes[reg].expression_level 
                                        for reg in self.genes[gene_name].regulators)
                    change = self.genes[gene_name].activate(stimulus * 0.5, regulatory_input)
                    changes.append(change)
        
        self._record_activation_state()
        return np.mean(changes)

    def _record_activation_state(self):
        """Registra estado atual da rede"""
        current_state = {
            name: gene.expression_level for name, gene in self.genes.items()
        }
        self.activation_history.append(current_state)

class GeneNetworkVisualizer:
    def __init__(self, network: DenseGeneNetwork, output_dir: str = "gene_network_viz"):
        self.network = network
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_network_visualization(self, timestep: int):
        """Cria visualização da rede gênica"""
        plt.figure(figsize=(20, 10))
        
        # Rede de regulação
        plt.subplot(121)
        G = nx.DiGraph()
        
        # Adiciona nós
        for name, gene in self.network.genes.items():
            G.add_node(name, 
                      category=gene.category,
                      expression=gene.expression_level)
        
        # Adiciona arestas
        for name, gene in self.network.genes.items():
            for target in gene.targets:
                G.add_edge(name, target)
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Desenha nós
        nx.draw_networkx_nodes(G, pos,
                             node_color=[G.nodes[node]['expression'] for node in G.nodes()],
                             node_size=500,
                             cmap='viridis')
        
        # Desenha arestas
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Gene Regulatory Network')
        
        # Heatmap de expressão
        plt.subplot(122)
        data = np.array([gene.expression_level for gene in self.network.genes.values()]).reshape(-1, 1)
        sns.heatmap(data,
                   yticklabels=list(self.network.genes.keys()),
                   xticklabels=['Expression'],
                   cmap='viridis')
        plt.title('Gene Expression Levels')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'network_state_{timestep}.png', dpi=300, bbox_inches='tight')
        plt.close()

def simulate_brain_development():
    """Simula desenvolvimento cerebral com ativação sequencial de vias"""
    network = DenseGeneNetwork()
    visualizer = GeneNetworkVisualizer(network)
    
    # Sequência de desenvolvimento
    simulation_steps = [
        ('Cortical_Development', 0.8),
        ('Interneuron_Development', 0.7),
        ('Synaptogenesis', 0.6),
        ('Activity_Dependent_Plasticity', 0.5),
        ('Circadian_Regulation', 0.4)
    ]
    
    for timestep, (pathway, strength) in enumerate(simulation_steps):
        logger.info(f"Timestep {timestep}: Activating {pathway}")
        change = network.activate_pathway(pathway, strength)
        visualizer.create_network_visualization(timestep)
        logger.info(f"Average network change: {change:.3f}")

if __name__ == "__main__":
    simulate_brain_development()