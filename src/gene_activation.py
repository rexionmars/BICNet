import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Set
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='gene_activation.log'
)
logger = logging.getLogger(__name__)

@dataclass
class Gene:
    name: str
    expression_level: float = 0.0
    baseline_expression: float = 0.1
    is_active: bool = False
    activation_threshold: float = 0.5
    
    def activate(self, stimulus: float):
        """Ativa o gene baseado no estímulo recebido"""
        self.expression_level += stimulus
        self.is_active = self.expression_level > self.activation_threshold
        logger.debug(f"Gene {self.name} - Level: {self.expression_level:.3f}, Active: {self.is_active}")

class GeneNetwork:
    def __init__(self):
        # Genes do desenvolvimento
        self.development_genes = {
            'PAX6': Gene('PAX6'),
            'EMX2': Gene('EMX2'),
            'FOXG1': Gene('FOXG1'),
            'NKX2.1': Gene('NKX2.1')
        }
        
        # Genes de neurotransmissores
        self.neurotransmitter_genes = {
            'SNAP25': Gene('SNAP25'),
            'SYN1': Gene('SYN1'),
            'GAD1': Gene('GAD1'),
            'SLC6A4': Gene('SLC6A4')
        }
        
        # Genes motores
        self.motor_genes = {
            'FOXP2': Gene('FOXP2'),
            'BDNF': Gene('BDNF'),
            'NRXN1': Gene('NRXN1')
        }
        
        # Genes de resposta rápida
        self.early_response_genes = {
            'C_FOS': Gene('C_FOS'),
            'ARC': Gene('ARC'),
            'NPAS4': Gene('NPAS4')
        }
        
        # Todas as categorias de genes
        self.all_categories = {
            'Development': self.development_genes,
            'Neurotransmitter': self.neurotransmitter_genes,
            'Motor': self.motor_genes,
            'Early Response': self.early_response_genes
        }
        
        # Registro de ativações
        self.activation_history = []
        
        logger.info("Gene network initialized")

    def activate_pathway(self, pathway_name: str, stimulus: float):
        """Ativa uma via específica de genes"""
        if pathway_name == 'motor_control':
            self._activate_motor_pathway(stimulus)
        elif pathway_name == 'development':
            self._activate_development_pathway(stimulus)
        elif pathway_name == 'neurotransmitter':
            self._activate_neurotransmitter_pathway(stimulus)
            
        self._record_activation_state()
        logger.info(f"Pathway {pathway_name} activated with stimulus {stimulus}")

    def _activate_motor_pathway(self, stimulus: float):
        """Simula ativação da via motora"""
        # Ativa genes motores
        for gene in self.motor_genes.values():
            gene.activate(stimulus)
        
        # Ativa genes de resposta rápida relacionados
        self.early_response_genes['C_FOS'].activate(stimulus * 0.8)
        self.early_response_genes['ARC'].activate(stimulus * 0.6)
        
        # Ativa genes de neurotransmissores necessários
        self.neurotransmitter_genes['SNAP25'].activate(stimulus * 0.7)
        self.neurotransmitter_genes['SYN1'].activate(stimulus * 0.7)

    def _activate_development_pathway(self, stimulus: float):
        """Simula ativação da via de desenvolvimento"""
        for gene in self.development_genes.values():
            gene.activate(stimulus)

    def _activate_neurotransmitter_pathway(self, stimulus: float):
        """Simula ativação da via de neurotransmissores"""
        for gene in self.neurotransmitter_genes.values():
            gene.activate(stimulus)

    def _record_activation_state(self):
        """Registra o estado atual de ativação de todos os genes"""
        current_state = {}
        for category, genes in self.all_categories.items():
            current_state[category] = {
                name: gene.expression_level 
                for name, gene in genes.items()
            }
        self.activation_history.append(current_state)

class GeneVisualizer:
    def __init__(self, network: GeneNetwork):
        self.network = network
        
    def plot_current_state(self, timestep: int):
        """Plota o estado atual de ativação dos genes"""
        plt.figure(figsize=(15, 10))
        
        # Prepara dados para visualização
        categories = []
        gene_names = []
        expression_levels = []
        
        for category, genes in self.network.all_categories.items():
            for name, gene in genes.items():
                categories.append(category)
                gene_names.append(name)
                expression_levels.append(gene.expression_level)
        
        # Cria heatmap
        data = np.array(expression_levels).reshape(-1, 1)
        
        plt.subplot(121)
        sns.heatmap(data,
                   yticklabels=gene_names,
                   xticklabels=['Expression Level'],
                   cmap='viridis',
                   center=0.5)
        plt.title(f'Gene Expression Levels - Timestep {timestep}')
        
        # Grafo de ativação
        plt.subplot(122)
        G = nx.Graph()
        
        # Adiciona nós para cada gene
        for name, level in zip(gene_names, expression_levels):
            G.add_node(name, expression=level)
        
        # Adiciona conexões entre genes relacionados
        for gene1 in gene_names:
            for gene2 in gene_names:
                if gene1 != gene2 and np.random.random() < 0.2:  # 20% chance de conexão
                    G.add_edge(gene1, gene2)
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos,
                node_color=[G.nodes[node]['expression'] for node in G.nodes()],
                cmap='viridis',
                node_size=1000,
                with_labels=True,
                font_size=8)
        
        plt.tight_layout()
        plt.savefig(f'gene_activation_{timestep}.png')
        plt.close()

def simulate_gene_activation():
    """Simula sequência de ativações gênicas"""
    network = GeneNetwork()
    visualizer = GeneVisualizer(network)
    
    # Simula vários passos de ativação
    for timestep in range(10):
        logger.info(f"Starting timestep {timestep}")
        
        # Simula diferentes estímulos
        if timestep < 3:
            # Fase inicial: desenvolvimento
            network.activate_pathway('development', 0.8)
        elif timestep < 6:
            # Fase intermediária: estabelecimento de neurotransmissores
            network.activate_pathway('neurotransmitter', 0.6)
        else:
            # Fase final: controle motor
            network.activate_pathway('motor_control', 0.7)
        
        # Visualiza estado atual
        visualizer.plot_current_state(timestep)
        
        logger.info(f"Completed timestep {timestep}")

if __name__ == "__main__":
    simulate_gene_activation()