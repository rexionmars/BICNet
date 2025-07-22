# neural_visualization_exporter.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import networkx as nx
from scipy import signal
import datetime
from complex_neural import ComplexNeuralAssembly, InteractionType, NeuromodulatorState
from enhanced_learning import EnhancedLearningAssembly, LearningRule

class NeuralVisualizationExporter:
    """Gera e salva visualizações de alta qualidade das redes neurais para posts de blog"""
    
    def __init__(self, output_dir="./blog_images"):
        """Inicializa o exportador com diretório de saída especificado"""
        self.output_dir = output_dir
        
        # Cria o diretório de saída se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Cria subdiretórios para diferentes tipos de visualizações
        self.standard_dir = os.path.join(output_dir, "standard_assembly")
        self.enhanced_dir = os.path.join(output_dir, "enhanced_assembly")
        self.demonstrations_dir = os.path.join(output_dir, "demonstrations")
        
        os.makedirs(self.standard_dir, exist_ok=True)
        os.makedirs(self.enhanced_dir, exist_ok=True)
        os.makedirs(self.demonstrations_dir, exist_ok=True)
        
        # Configuração de alta qualidade para as figuras
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
        # Tamanho de figura para posts de blog
        self.blog_figsize = (12, 10)
    
    def generate_standard_assembly_visualizations(self, assembly, timestep):
        """Gera visualizações para a assembleia neural padrão"""
        # Visualização completa
        self._generate_full_visualization(assembly, timestep, self.standard_dir, "standard")
        
        # Visualizações específicas
        self._generate_connectivity_visualization(assembly, timestep, self.standard_dir)
        self._generate_activity_visualization(assembly, timestep, self.standard_dir)
        self._generate_calcium_dynamics_visualization(assembly, timestep, self.standard_dir)
        self._generate_neuromodulator_visualization(assembly, timestep, self.standard_dir)
    
    def generate_enhanced_assembly_visualizations(self, assembly, timestep):
        """Gera visualizações para a assembleia neural aprimorada"""
        # Visualização completa
        self._generate_full_visualization(assembly, timestep, self.enhanced_dir, "enhanced")
        
        # Visualizações específicas do aprendizado aprimorado
        self._generate_learning_rules_visualization(assembly, timestep, self.enhanced_dir)
        self._generate_reinforcement_learning_visualization(assembly, timestep, self.enhanced_dir)
        self._generate_neural_assemblies_visualization(assembly, timestep, self.enhanced_dir)
        self._generate_bcm_visualization(assembly, timestep, self.enhanced_dir)
        self._generate_metaplasticity_visualization(assembly, timestep, self.enhanced_dir)
    
    def generate_demonstration_visualizations(self, demo_name, demo_state, assembly, timestep):
        """Gera visualizações para uma demonstração específica"""
        demo_dir = os.path.join(self.demonstrations_dir, demo_name.lower().replace(' ', '_'))
        os.makedirs(demo_dir, exist_ok=True)
        
        # Visualização baseada no tipo de demonstração
        if demo_name == "Pattern Recognition":
            self._generate_pattern_recognition_visualization(demo_state, assembly, timestep, demo_dir)
        elif demo_name == "Neuroplasticity Visualization":
            self._generate_neuroplasticity_visualization(demo_state, assembly, timestep, demo_dir)
        elif demo_name == "Oscillatory Dynamics":
            self._generate_oscillatory_dynamics_visualization(demo_state, assembly, timestep, demo_dir)
        elif demo_name == "Neuromodulation Effects":
            self._generate_neuromodulation_effects_visualization(demo_state, assembly, timestep, demo_dir)
        # Outras demonstrações podem ser adicionadas conforme necessário
    
    def _generate_full_visualization(self, assembly, timestep, output_dir, prefix):
        """Gera visualização completa da assembleia neural"""
        fig = plt.figure(figsize=self.blog_figsize)
        
        # Matriz de conectividade
        ax1 = fig.add_subplot(231)
        matrix = np.zeros((assembly.size, assembly.size))
        for (i, j), conn in assembly.connections.items():
            matrix[i, j] = conn.weight if conn.type == InteractionType.EXCITATORY else -conn.weight
        im = ax1.imshow(matrix, cmap='RdBu_r')
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Connectivity Matrix')
        
        # Grafo de conexões
        ax2 = fig.add_subplot(232)
        G = nx.DiGraph()
        
        # Adiciona apenas conexões fortes para clareza
        for (i, j), conn in assembly.connections.items():
            if conn.weight > 0.5:  # Limiar para clareza visual
                G.add_edge(i, j, weight=conn.weight, 
                          color='g' if conn.type == InteractionType.EXCITATORY else 'r')
        
        if G.number_of_edges() > 0:
            # Limita o número de nós para clareza
            if G.number_of_nodes() > 30:
                # Seleciona os nós com maior grau
                degree = dict(G.degree())
                top_nodes = sorted(degree.keys(), key=lambda x: degree[x], reverse=True)[:30]
                G = G.subgraph(top_nodes)
            
            pos = nx.spring_layout(G, seed=42)
            edge_colors = [G[u][v]['color'] for u, v in G.edges()]
            nx.draw(G, pos, edge_color=edge_colors, with_labels=True, node_size=200, ax=ax2)
        
        ax2.set_title('Connection Graph')
        
        # Níveis de cálcio
        ax3 = fig.add_subplot(233)
        ax3.plot(assembly.calcium_levels)
        ax3.set_title('Calcium Levels')
        ax3.set_xlabel('Neuron ID')
        ax3.set_ylabel('Calcium Level')
        
        # Distribuição de pesos
        ax4 = fig.add_subplot(234)
        weights = [conn.weight for conn in assembly.connections.values()]
        ax4.hist(weights, bins=20)
        ax4.set_title('Weight Distribution')
        ax4.set_xlabel('Weight')
        ax4.set_ylabel('Count')
        
        # Neuromoduladores
        ax5 = fig.add_subplot(235)
        levels = [
            assembly.neuromodulators.dopamine,
            assembly.neuromodulators.serotonin,
            assembly.neuromodulators.acetylcholine,
            assembly.neuromodulators.norepinephrine
        ]
        ax5.bar(['DA', '5-HT', 'ACh', 'NE'], levels)
        ax5.set_title('Neuromodulator Levels')
        ax5.set_ylabel('Level')
        
        # Atividade recente
        ax6 = fig.add_subplot(236)
        history_len = min(100, len(assembly.activation_history))
        if history_len > 0:
            recent = assembly.activation_history[-history_len:]
            activity = [len(act) for act in recent]
            ax6.plot(range(len(activity)), activity)
            ax6.set_title('Recent Activity')
            ax6.set_xlabel('Time Step')
            ax6.set_ylabel('Active Neurons')
        
        # Adiciona informações de timestep e tipo de assembleia
        fig.suptitle(f'{prefix.capitalize()} Neural Assembly at Timestep {timestep}', fontsize=14)
        
        # Salva a figura
        filename = f"{prefix}_assembly_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Ajusta para o título principal
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_connectivity_visualization(self, assembly, timestep, output_dir):
        """Gera visualização detalhada da conectividade neural"""
        fig = plt.figure(figsize=(10, 8))
        
        # Matriz de conectividade em alta resolução
        ax1 = fig.add_subplot(221)
        matrix = np.zeros((assembly.size, assembly.size))
        for (i, j), conn in assembly.connections.items():
            matrix[i, j] = conn.weight if conn.type == InteractionType.EXCITATORY else -conn.weight
        im = ax1.imshow(matrix, cmap='RdBu_r')
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Connectivity Matrix')
        ax1.set_xlabel('Post-synaptic Neuron')
        ax1.set_ylabel('Pre-synaptic Neuron')
        
        # Distribuição de tipos de conexões
        ax2 = fig.add_subplot(222)
        excitatory = sum(1 for conn in assembly.connections.values() 
                         if conn.type == InteractionType.EXCITATORY)
        inhibitory = sum(1 for conn in assembly.connections.values() 
                         if conn.type == InteractionType.INHIBITORY)
        modulatory = sum(1 for conn in assembly.connections.values() 
                         if conn.type == InteractionType.MODULATORY)
        associative = sum(1 for conn in assembly.connections.values() 
                         if conn.type == InteractionType.ASSOCIATIVE)
        
        ax2.bar(['Excitatory', 'Inhibitory', 'Modulatory', 'Associative'], 
                [excitatory, inhibitory, modulatory, associative])
        ax2.set_title('Connection Types')
        ax2.set_ylabel('Count')
        
        # Histograma de pesos por tipo
        ax3 = fig.add_subplot(223)
        weights_exc = [conn.weight for conn in assembly.connections.values() 
                      if conn.type == InteractionType.EXCITATORY]
        weights_inh = [conn.weight for conn in assembly.connections.values() 
                      if conn.type == InteractionType.INHIBITORY]
        
        if weights_exc:
            ax3.hist(weights_exc, bins=20, alpha=0.5, label='Excitatory')
        if weights_inh:
            ax3.hist(weights_inh, bins=20, alpha=0.5, label='Inhibitory')
        
        ax3.set_title('Weight Distribution by Type')
        ax3.set_xlabel('Weight')
        ax3.set_ylabel('Count')
        ax3.legend()
        
        # Grafo de conexões destacando tipos
        ax4 = fig.add_subplot(224)
        G = nx.DiGraph()
        
        # Adiciona apenas conexões fortes
        for (i, j), conn in assembly.connections.items():
            if conn.weight > 0.5:  # Limiar para clareza visual
                G.add_edge(i, j, weight=conn.weight, 
                          color='g' if conn.type == InteractionType.EXCITATORY else 'r')
        
        if G.number_of_edges() > 0:
            # Limita número de nós
            if G.number_of_nodes() > 20:
                degree = dict(G.degree())
                top_nodes = sorted(degree.keys(), key=lambda x: degree[x], reverse=True)[:20]
                G = G.subgraph(top_nodes)
            
            pos = nx.spring_layout(G, seed=42)
            edge_colors = [G[u][v]['color'] for u, v in G.edges()]
            
            # Tamanho do nó baseado no grau
            node_size = [G.degree(n) * 50 for n in G.nodes()]
            
            nx.draw(G, pos, edge_color=edge_colors, node_size=node_size,
                   with_labels=True, ax=ax4)
        
        ax4.set_title('Connection Graph (Strongest Connections)')
        
        # Título principal
        fig.suptitle(f'Neural Connectivity Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"connectivity_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_activity_visualization(self, assembly, timestep, output_dir):
        """Gera visualização detalhada da atividade neural"""
        fig = plt.figure(figsize=(10, 8))
        
        # Atividade recente em detalhe
        ax1 = fig.add_subplot(221)
        history_len = min(100, len(assembly.activation_history))
        if history_len > 0:
            recent = assembly.activation_history[-history_len:]
            activity = [len(act) for act in recent]
            ax1.plot(range(len(activity)), activity)
            
            # Adiciona média móvel
            window_size = min(20, len(activity))
            if window_size > 1:
                activity_smooth = np.convolve(activity, np.ones(window_size)/window_size, mode='valid')
                valid_steps = range(window_size-1, len(activity))
                ax1.plot(valid_steps, activity_smooth, 'r-', linewidth=2, label='Moving Average')
                ax1.legend()
            
            ax1.set_title('Recent Activity')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Active Neurons')
            ax1.grid(True, alpha=0.3)
        
        # Mapa de ativação por neurônio
        ax2 = fig.add_subplot(222)
        if history_len > 0:
            neuron_activity = np.zeros(assembly.size)
            for active_set in assembly.activation_history[-30:]:
                for neuron in active_set:
                    neuron_activity[neuron] += 1
            
            # Normaliza por número de passos
            neuron_activity = neuron_activity / 30
            
            ax2.bar(range(assembly.size), neuron_activity)
            ax2.set_title('Neuron Activation Frequency (Last 30 steps)')
            ax2.set_xlabel('Neuron ID')
            ax2.set_ylabel('Activation Probability')
        
        # Padrão de ativação atual
        ax3 = fig.add_subplot(223)
        if assembly.activation_history:
            current_activation = np.zeros(assembly.size)
            for neuron in assembly.activation_history[-1]:
                current_activation[neuron] = 1
            
            ax3.bar(range(assembly.size), current_activation)
            ax3.set_title('Current Activation Pattern')
            ax3.set_xlabel('Neuron ID')
            ax3.set_ylabel('Active (1) / Inactive (0)')
        
        # Análise de oscilações (se houver atividade suficiente)
        ax4 = fig.add_subplot(224)
        if history_len >= 64:  # Mínimo para análise de Fourier
            activity_array = np.array([len(act) for act in assembly.activation_history[-512:]])
            
            # Remove tendência e aplica janela
            activity_detrended = activity_array - np.mean(activity_array)
            windowed = activity_detrended * signal.windows.hann(len(activity_detrended))
            
            # Calcula FFT
            fft = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(len(windowed), d=0.1)  # Assumindo 10 passos = 1 segundo
            fft_mag = np.abs(fft)
            
            ax4.plot(freqs, fft_mag)
            ax4.set_title('Frequency Analysis')
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Magnitude')
            ax4.set_xlim(0, min(10, max(freqs)))  # Limita a exibição
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Not enough data for\nfrequency analysis", 
                    ha='center', va='center')
        
        # Título principal
        fig.suptitle(f'Neural Activity Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"activity_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_calcium_dynamics_visualization(self, assembly, timestep, output_dir):
        """Gera visualização detalhada da dinâmica de cálcio"""
        fig = plt.figure(figsize=(10, 8))
        
        # Níveis de cálcio atuais
        ax1 = fig.add_subplot(221)
        ax1.plot(assembly.calcium_levels)
        ax1.set_title('Current Calcium Levels')
        ax1.set_xlabel('Neuron ID')
        ax1.set_ylabel('Calcium Level')
        ax1.grid(True, alpha=0.3)
        
        # Relação entre cálcio e ativação recente
        ax2 = fig.add_subplot(222)
        if assembly.activation_history:
            neuron_activity = np.zeros(assembly.size)
            for active_set in assembly.activation_history[-30:]:
                for neuron in active_set:
                    neuron_activity[neuron] += 1
            
            # Scatter plot: atividade vs. cálcio
            ax2.scatter(neuron_activity, assembly.calcium_levels, alpha=0.5)
            ax2.set_title('Calcium vs. Recent Activity')
            ax2.set_xlabel('Activity Count (Last 30 steps)')
            ax2.set_ylabel('Calcium Level')
            ax2.grid(True, alpha=0.3)
        
        # Síntese de proteínas
        ax3 = fig.add_subplot(223)
        ax3.plot(assembly.protein_synthesis)
        ax3.set_title('Protein Synthesis Levels')
        ax3.set_xlabel('Neuron ID')
        ax3.set_ylabel('Synthesis Level')
        ax3.grid(True, alpha=0.3)
        
        # Visualização do limiar de síntese de proteínas
        ax4 = fig.add_subplot(224)
        
        # Plotamos níveis de cálcio e o limiar
        ax4.plot(assembly.calcium_levels, label='Calcium Level')
        ax4.axhline(y=0.5, color='r', linestyle='--', label='Synthesis Threshold')
        
        # Destacar neurônios acima do limiar
        above_threshold = assembly.calcium_levels > 0.5
        if np.any(above_threshold):
            ax4.scatter(np.where(above_threshold)[0], 
                       assembly.calcium_levels[above_threshold], 
                       color='r', s=50, zorder=3)
        
        ax4.set_title('Calcium Threshold for Protein Synthesis')
        ax4.set_xlabel('Neuron ID')
        ax4.set_ylabel('Calcium Level')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Título principal
        fig.suptitle(f'Calcium Dynamics Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"calcium_dynamics_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_neuromodulator_visualization(self, assembly, timestep, output_dir):
        """Gera visualização detalhada dos neuromoduladores"""
        fig = plt.figure(figsize=(10, 8))
        
        # Níveis atuais de neuromoduladores
        ax1 = fig.add_subplot(221)
        levels = [
            assembly.neuromodulators.dopamine,
            assembly.neuromodulators.serotonin,
            assembly.neuromodulators.acetylcholine,
            assembly.neuromodulators.norepinephrine
        ]
        bars = ax1.bar(['Dopamine', 'Serotonin', 'Acetylcholine', 'Norepinephrine'], levels)
        
        # Colorindo baseado no desvio do normal (1.0)
        for i, bar in enumerate(bars):
            if levels[i] > 1.1:
                bar.set_color('green')
            elif levels[i] < 0.9:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)  # Linha de base
        ax1.set_title('Current Neuromodulator Levels')
        ax1.set_ylabel('Level')
        ax1.set_ylim(0, 2.0)
        
        # Efeitos dos neuromoduladores
        ax2 = fig.add_subplot(222)
        
        # Efeitos teóricos de cada neuromodulador em diferentes processos
        effects = {
            'Learning Rate': assembly.neuromodulators.dopamine,
            'Signal-to-Noise': assembly.neuromodulators.acetylcholine,
            'Mood Stability': assembly.neuromodulators.serotonin,
            'Arousal': assembly.neuromodulators.norepinephrine
        }
        
        ax2.bar(effects.keys(), effects.values())
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)  # Linha de base
        ax2.set_title('Neuromodulatory Effects')
        ax2.set_ylabel('Relative Strength')
        ax2.set_ylim(0, 2.0)
        
        # Simulação de regra STDP modulada por dopamina
        ax3 = fig.add_subplot(223)
        
        # Simula a curva STDP para diferentes níveis de dopamina
        time_diff = np.linspace(-50, 50, 100)
        
        # Função de STDP base
        def stdp_base(t):
            if t > 0:  # Potenciação
                return 0.1 * np.exp(-t / 20.0)
            else:  # Depressão
                return -0.1 * np.exp(t / 20.0)
        
        # Calcula curvas STDP para diferentes níveis de dopamina
        stdp_normal = np.array([stdp_base(t) for t in time_diff])
        stdp_high_da = stdp_normal * assembly.neuromodulators.dopamine
        stdp_low_da = stdp_normal * 0.5  # Dopamina baixa simulada
        
        ax3.plot(time_diff, stdp_normal, 'k--', label='Baseline')
        ax3.plot(time_diff, stdp_high_da, 'g-', label=f'Current DA={assembly.neuromodulators.dopamine:.2f}')
        ax3.plot(time_diff, stdp_low_da, 'r-', label='Low DA=0.50')
        
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_title('Dopamine Modulation of STDP')
        ax3.set_xlabel('Spike Timing Difference (ms)')
        ax3.set_ylabel('Weight Change')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Estados comuns de neuromoduladores
        ax4 = fig.add_subplot(224, projection='polar')  # Aqui está a mudança: projectionn='polar'
        
        # Define alguns estados de neuromoduladores comuns
        states = {
            'Focused': [1.2, 1.0, 1.5, 1.3],  # DA, 5-HT, ACh, NE
            'Relaxed': [1.0, 1.3, 0.8, 0.7],
            'Stressed': [0.7, 0.6, 1.0, 1.8],
            'Rewarded': [1.8, 1.2, 1.0, 1.1],
            'Current': levels
        }
        
        # Cria um gráfico de radar
        categories = ['Dopamine', 'Serotonin', 'Acetylcholine', 'Norepinephrine']
        N = len(categories)
        
        # Ângulos para o gráfico de radar
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Fecha o círculo
        
        # Inicializa gráfico de radar
        ax4.set_theta_offset(np.pi / 2)
        ax4.set_theta_direction(-1)
        ax4.set_rlabel_position(0)
        
        # Desenha linhas de nível de eixo
        plt.yticks([0.5, 1.0, 1.5], ["0.5", "1.0", "1.5"], color="grey", size=7)
        plt.ylim(0, 2)
        
        # Plota cada estado
        for state_name, state_values in states.items():
            # Fecha o loop para o gráfico de radar
            values = state_values.copy()
            values += values[:1]
            
            ax4.plot(angles, values, linewidth=1, linestyle='solid', label=state_name)
            ax4.fill(angles, values, alpha=0.1)
        
        # Adiciona categorias
        plt.xticks(angles[:-1], categories, size=8)
        ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax4.set_title('Neuromodulator States Comparison')
        
        # Título principal
        fig.suptitle(f'Neuromodulator Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"neuromodulators_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_learning_rules_visualization(self, assembly, timestep, output_dir):
        """Gera visualização das regras de aprendizado para a assembleia neural aprimorada"""
        # Verifica se a assembleia é do tipo correto
        if not isinstance(assembly, EnhancedLearningAssembly):
            return None
        
        fig = plt.figure(figsize=(12, 10))
        
        # Status das regras de aprendizado ativas
        ax1 = fig.add_subplot(321)
        rules = list(assembly.active_learning_rules.keys())
        active = [int(assembly.active_learning_rules[rule]) for rule in rules]
        rule_names = [rule.value for rule in rules]
        
        bars = ax1.bar(rule_names, active)
        for i, bar in enumerate(bars):
            if active[i] == 1:
                bar.set_color('green')
            else:
                bar.set_color('lightgray')
        
        ax1.set_title('Active Learning Rules')
        ax1.set_ylabel('Active (1) / Inactive (0)')
        ax1.set_ylim(0, 1.2)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Visualização de STDP
        ax2 = fig.add_subplot(322)
        
        # Simula a curva STDP
        time_diff = np.linspace(-50, 50, 100)
        
        # Função de STDP base
        def stdp_base(t):
            if t > 0:  # Potenciação
                return 0.1 * np.exp(-t / 20.0)
            else:  # Depressão
                return -0.1 * np.exp(t / 20.0)
        
        # Calcula curva STDP
        stdp_values = np.array([stdp_base(t) for t in time_diff])
        
        ax2.plot(time_diff, stdp_values)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_title('STDP Learning Function')
        ax2.set_xlabel('Spike Timing Difference (ms)')
        ax2.set_ylabel('Weight Change')
        ax2.grid(True, alpha=0.3)
        
        # Visualização de BCM
        ax3 = fig.add_subplot(323)
        
        # Simula a regra BCM
        post_activity = np.linspace(0, 2, 100)
        thresholds = [0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            # Calcula mudança de peso para diferentes níveis de atividade pós-sináptica
            bcm_dw = post_activity * (post_activity - threshold)
            ax3.plot(post_activity, bcm_dw, label=f'Threshold={threshold}')
        
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_title('BCM Learning Rule')
        ax3.set_xlabel('Post-synaptic Activity')
        ax3.set_ylabel('Weight Change')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Visualização de aprendizado por reforço
        ax4 = fig.add_subplot(324)
        
        # Simula traços de elegibilidade decaindo
        time_steps = np.arange(0, 50)
        decay_rates = [0.8, 0.9, 0.95, 0.98]
        
        for decay in decay_rates:
            # Calcula decaimento do traço
            trace = decay ** time_steps
            ax4.plot(time_steps, trace, label=f'Decay={decay}')
        
        ax4.set_title('Eligibility Trace Decay')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Trace Strength')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Visualização de aprendizado competitivo
        ax5 = fig.add_subplot(325)
        
        # Simula ativação em uma rede competitiva
        neurons = np.arange(20)
        activations = np.zeros(20)
        
        # Definindo alguns neurônios como vencedores
        winners = [5, 12, 18]
        for w in winners:
            # Pico de ativação no vencedor
            activations[w] = 1.0
            # Ativação diminui com a distância do vencedor
            for i in range(20):
                if i != w:
                    dist = abs(i - w)
                    activations[i] = max(activations[i], 0.9 ** dist)
        
        ax5.bar(neurons, activations)
        
        # Destacando neurônios vencedores
        for w in winners:
            ax5.bar(w, activations[w], color='red')
        
        ax5.set_title('Competitive Learning: Winner Neurons')
        ax5.set_xlabel('Neuron ID')
        ax5.set_ylabel('Activation Level')
        
        # Visualização de Hebbian/Oja
        ax6 = fig.add_subplot(326)
        
        # Criando dados para visualizar regra de Oja vs Hebbian clássica
        pre_activation = np.linspace(0, 1, 100)
        post_activation = 0.8  # fixo para simplicidade
        weight = 0.5  # fixo para simplicidade
        
        # Regra Hebbiana: dw = η * pre * post
        hebbian_dw = 0.1 * pre_activation * post_activation
        
        # Regra de Oja: dw = η * post * (pre - post * weight)
        oja_dw = 0.1 * post_activation * (pre_activation - post_activation * weight)
        
        ax6.plot(pre_activation, hebbian_dw, label='Hebbian')
        ax6.plot(pre_activation, oja_dw, label='Oja')
        ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax6.set_title('Hebbian vs Oja Learning Rule')
        ax6.set_xlabel('Pre-synaptic Activity')
        ax6.set_ylabel('Weight Change')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Título principal
        fig.suptitle(f'Advanced Learning Rules (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"learning_rules_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_reinforcement_learning_visualization(self, assembly, timestep, output_dir):
        """Gera visualização do aprendizado por reforço para a assembleia neural aprimorada"""
        # Verifica se a assembleia é do tipo correto
        if not isinstance(assembly, EnhancedLearningAssembly):
            return None
        
        fig = plt.figure(figsize=(12, 8))
        
        # Visualização de traços de elegibilidade
        ax1 = fig.add_subplot(221)
        
        # Cria matriz para visualizar traços
        eligibility_matrix = np.zeros((assembly.size, assembly.size))
        for (i, j), trace in assembly.eligibility_traces.items():
            eligibility_matrix[i, j] = trace
        
        im = ax1.imshow(eligibility_matrix, cmap='hot')
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Eligibility Traces')
        ax1.set_xlabel('Post-synaptic Neuron')
        ax1.set_ylabel('Pre-synaptic Neuron')
        
        # Visualização do histórico de recompensas
        ax2 = fig.add_subplot(222)
        
        if assembly.reward_history:
            ax2.plot(assembly.reward_history)
            
            # Adiciona média móvel
            if len(assembly.reward_history) > 10:
                window_size = min(20, len(assembly.reward_history))
                reward_smooth = np.convolve(assembly.reward_history, 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
                valid_steps = range(window_size-1, len(assembly.reward_history))
                ax2.plot(valid_steps, reward_smooth, 'r-', linewidth=2, label='Moving Average')
                ax2.legend()
            
            ax2.set_title('Reward History')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Reward')
            ax2.grid(True, alpha=0.3)
        
        # Valores top traços de elegibilidade
        ax3 = fig.add_subplot(223)
        
        # Obtém os top 10 traços de elegibilidade
        if assembly.eligibility_traces:
            top_traces = sorted(assembly.eligibility_traces.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
            
            connection_ids = [f"({pre},{post})" for (pre, post), _ in top_traces]
            trace_values = [trace for _, trace in top_traces]
            
            ax3.barh(connection_ids, trace_values)
            ax3.set_title('Top 10 Eligibility Traces')
            ax3.set_xlabel('Trace Value')
        else:
            ax3.text(0.5, 0.5, "No eligibility traces available", 
                    ha='center', va='center')
        
        # Demonstração de TD-Learning
        ax4 = fig.add_subplot(224)
        
        # Simulação de TD-Learning para visualização
        time_steps = np.arange(50)
        reward = np.zeros(50)
        # Recompensa em t=10
        reward[10] = 1.0
        
        value = np.zeros(50)
        td_error = np.zeros(50)
        
        # Simulando TD-Learning com gamma=0.9
        gamma = 0.9
        for t in range(1, 50):
            value[t] = reward[t] + gamma * value[t-1]
            td_error[t] = reward[t] + gamma * value[t] - value[t-1]
        
        # Plotando resultados
        ax4.plot(time_steps, reward, 'g-', label='Reward')
        ax4.plot(time_steps, value, 'b-', label='Value Estimate')
        ax4.plot(time_steps, td_error, 'r-', label='TD Error')
        
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_title('Temporal Difference Learning')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Título principal
        fig.suptitle(f'Reinforcement Learning Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"reinforcement_learning_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_neural_assemblies_visualization(self, assembly, timestep, output_dir):
        """Gera visualização de assembleias neurais"""
        # Verifica se a assembleia é do tipo correto
        if not isinstance(assembly, EnhancedLearningAssembly):
            return None
        
        fig = plt.figure(figsize=(12, 8))
        
        # Detecta assembleias neurais
        neural_assemblies = assembly.detect_neural_assemblies()
        
        # Mapa de assembleias neurais
        ax1 = fig.add_subplot(221)
        
        if neural_assemblies:
            # Cria uma matriz para visualizar as assembleias
            assembly_matrix = np.zeros(assembly.size)
            for i, neurons in enumerate(neural_assemblies):
                for neuron in neurons:
                    if neuron < assembly.size:  # Garantir limites
                        assembly_matrix[neuron] = i + 1
            
            # Plotando como barras coloridas
            cmap = plt.cm.get_cmap('tab10', len(neural_assemblies) + 1)
            im = ax1.bar(range(assembly.size), assembly_matrix, color=cmap(assembly_matrix))
            ax1.set_title(f'Neural Assemblies ({len(neural_assemblies)} detected)')
            ax1.set_xlabel('Neuron ID')
            ax1.set_ylabel('Assembly ID')
        else:
            ax1.text(0.5, 0.5, "No neural assemblies detected", 
                    ha='center', va='center')
        
        # Visualização de grafo das assembleias
        ax2 = fig.add_subplot(222)
        
        if neural_assemblies:
            G = nx.DiGraph()
            
            # Adiciona nós para cada assembleia
            colors = {}
            for i, neurons in enumerate(neural_assemblies):
                for neuron in neurons:
                    if neuron < assembly.size:  # Garantir limites
                        G.add_node(neuron)
                        colors[neuron] = i + 1
            
            # Adiciona arestas entre neurônios na mesma assembleia
            for (i, j), conn in assembly.connections.items():
                if i in colors and j in colors:
                    if conn.weight > 0.3:  # Mostra apenas conexões fortes
                        G.add_edge(i, j, weight=conn.weight)
            
            if G.number_of_nodes() > 0:
                # Layout do grafo
                pos = nx.spring_layout(G, seed=42)
                
                # Cores dos nós baseadas na assembleia
                node_colors = [plt.cm.tab10(colors.get(n, 0)) for n in G.nodes()]
                
                # Desenha o grafo
                nx.draw(G, pos, node_color=node_colors, with_labels=True, 
                       node_size=200, edge_color='gray', alpha=0.7, ax=ax2)
                
                ax2.set_title('Neural Assemblies Connectivity')
            else:
                ax2.text(0.5, 0.5, "Not enough strong connections\nto visualize", 
                        ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, "No neural assemblies detected", 
                    ha='center', va='center')
        
        # Distribuição de tamanho das assembleias
        ax3 = fig.add_subplot(223)
        
        if neural_assemblies:
            # Calcula tamanhos das assembleias
            assembly_sizes = [len(neurons) for neurons in neural_assemblies]
            
            # Plotar histograma dos tamanhos
            ax3.bar(range(1, len(assembly_sizes) + 1), assembly_sizes)
            ax3.set_title('Assembly Size Distribution')
            ax3.set_xlabel('Assembly ID')
            ax3.set_ylabel('Number of Neurons')
        else:
            ax3.text(0.5, 0.5, "No neural assemblies detected", 
                    ha='center', va='center')
        
        # Ativação de assembleias ao longo do tempo
        ax4 = fig.add_subplot(224)
        
        if neural_assemblies and len(assembly.activation_history) > 10:
            # Analisamos ativação das assembleias ao longo do tempo
            history_len = min(100, len(assembly.activation_history))
            recent_history = assembly.activation_history[-history_len:]
            
            # Para cada assembleia, calculamos quanto dela está ativa em cada passo
            assembly_activity = []
            for neurons in neural_assemblies:
                neurons_set = set(neurons)
                activity = []
                for step_active in recent_history:
                    # Interseção entre neurônios ativos e neurônios na assembleia
                    common = neurons_set.intersection(step_active)
                    # Porcentagem da assembleia que está ativa
                    if neurons:  # Evita divisão por zero
                        percent_active = len(common) / len(neurons)
                        activity.append(percent_active)
                    else:
                        activity.append(0)
                assembly_activity.append(activity)
            
            # Plotagem
            for i, activity in enumerate(assembly_activity):
                ax4.plot(activity, label=f'Assembly {i+1}')
            
            ax4.set_title('Assembly Activation Over Time')
            ax4.set_xlabel('Recent Time Steps')
            ax4.set_ylabel('Portion Active')
            ax4.set_ylim(0, 1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Not enough data to analyze\nassembly activation", 
                    ha='center', va='center')
        
        # Título principal
        fig.suptitle(f'Neural Assemblies Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"neural_assemblies_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_bcm_visualization(self, assembly, timestep, output_dir):
        """Gera visualização da regra BCM (Bienenstock-Cooper-Munro)"""
        # Verifica se a assembleia é do tipo correto
        if not isinstance(assembly, EnhancedLearningAssembly):
            return None
        
        fig = plt.figure(figsize=(12, 8))
        
        # Visualização atual dos limiares de BCM
        ax1 = fig.add_subplot(221)
        
        # Plotando médias de atividade (limiar BCM)
        ax1.plot(assembly.activity_averages)
        ax1.set_title('BCM Activity Averages (Thresholds)')
        ax1.set_xlabel('Neuron ID')
        ax1.set_ylabel('Activity Average')
        ax1.grid(True, alpha=0.3)
        
        # Visualização da dinâmica da regra BCM
        ax2 = fig.add_subplot(222)
        
        # Simulando função BCM para diferentes médias de atividade
        post_activity = np.linspace(0, 1.5, 100)
        thresholds = [0.2, 0.4, 0.6, 0.8]
        
        for threshold in thresholds:
            # BCM: dw = pre * post * (post - threshold)
            # Assumindo pre = 1 para simplicidade
            dw = post_activity * (post_activity - threshold)
            ax2.plot(post_activity, dw, label=f'Threshold={threshold}')
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_title('BCM Weight Modification Function')
        ax2.set_xlabel('Post-synaptic Activity')
        ax2.set_ylabel('Weight Change (dw)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Visualização da relação entre atividade e limiar
        ax3 = fig.add_subplot(223)
        
        # Simula como o limiar muda com a média de atividade
        activity_history = np.linspace(0, 1, 100)
        thresholds = np.zeros_like(activity_history)
        
        # Parâmetros de simulação
        tau = assembly.learning_parameters.bcm_time_constant
        current_threshold = 0.5
        
        # Simulando atualização do limiar
        for i, activity in enumerate(activity_history):
            # Atualização do limiar (média móvel de atividade^2)
            current_threshold = (1 - 1/tau) * current_threshold + (1/tau) * activity**2
            thresholds[i] = current_threshold
        
        ax3.plot(activity_history, thresholds)
        ax3.set_title('BCM Threshold Adaptation')
        ax3.set_xlabel('Activity Level')
        ax3.set_ylabel('Threshold')
        ax3.grid(True, alpha=0.3)
        
        # Análise de estabilidade
        ax4 = fig.add_subplot(224)
        
        # Simulando comportamento a longo prazo da regra BCM
        initial_weights = np.linspace(0, 1, 100)
        time_steps = 1000
        final_weights = np.zeros_like(initial_weights)
        
        # Parâmetros da simulação
        pre_activity = 0.5  # Atividade pré-sináptica constante
        threshold = 0.3     # Limiar fixo para simplicidade
        learning_rate = 0.001
        
        # Simulando evolução de pesos para diferentes valores iniciais
        for i, w in enumerate(initial_weights):
            weight = w
            for _ in range(time_steps):
                # Cálculo da atividade pós-sináptica (simplificado)
                post_activity = pre_activity * weight
                
                # Atualização de peso BCM
                dw = learning_rate * pre_activity * post_activity * (post_activity - threshold)
                weight += dw
                weight = np.clip(weight, 0, 1)
            
            final_weights[i] = weight
        
        # Plotando pesos iniciais vs. finais
        ax4.plot(initial_weights, final_weights)
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Linha de identidade
        ax4.set_title('BCM Weight Convergence')
        ax4.set_xlabel('Initial Weight')
        ax4.set_ylabel('Final Weight (After Convergence)')
        ax4.grid(True, alpha=0.3)
        
        # Título principal
        fig.suptitle(f'BCM Learning Rule Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"bcm_learning_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_metaplasticity_visualization(self, assembly, timestep, output_dir):
        """Gera visualização de metaplasticidade"""
        # Verifica se a assembleia é do tipo correto
        if not isinstance(assembly, EnhancedLearningAssembly):
            return None
        
        fig = plt.figure(figsize=(12, 8))
        
        # Visualização dos limiares de plasticidade
        ax1 = fig.add_subplot(221)
        
        ax1.plot(assembly.plasticity_thresholds)
        ax1.set_title('Metaplasticity Thresholds')
        ax1.set_xlabel('Neuron ID')
        ax1.set_ylabel('Threshold')
        ax1.grid(True, alpha=0.3)
        
        # Distribuição de limiares
        ax2 = fig.add_subplot(222)
        
        ax2.hist(assembly.plasticity_thresholds, bins=20)
        ax2.set_title('Metaplasticity Threshold Distribution')
        ax2.set_xlabel('Threshold Value')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # Simulação de efeitos da metaplasticidade na STDP
        ax3 = fig.add_subplot(223)
        
        # Simula curvas STDP para diferentes limiares de plasticidade
        time_diff = np.linspace(-50, 50, 100)
        
        # Função de modulação da STDP baseada em limiar
        def modulated_stdp(t, threshold):
            base_value = 0.1 * np.exp(-abs(t) / 20.0) * (1 if t > 0 else -1)
            # Aplica modulação baseada no limiar
            modulation = 2 * (1 - threshold)  # Mais plástico quando limiar é baixo
            return base_value * modulation
        
        # Plota curvas STDP para diferentes limiares
        thresholds = [0.2, 0.5, 0.8]
        for threshold in thresholds:
            stdp_values = [modulated_stdp(t, threshold) for t in time_diff]
            ax3.plot(time_diff, stdp_values, label=f'Threshold={threshold}')
        
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_title('Metaplasticity Effect on STDP')
        ax3.set_xlabel('Spike Timing Difference (ms)')
        ax3.set_ylabel('Weight Change')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Dinâmica de metaplasticidade
        ax4 = fig.add_subplot(224)
        
        # Simula como o limiar de plasticidade muda com a atividade
        activity_levels = np.linspace(0, 1, 100)
        thresholds = np.zeros((3, 100))
        
        # Taxas de metaplasticidade
        rates = [0.001, 0.01, 0.1]
        
        # Simula evolução do limiar para diferentes taxas
        for r, rate in enumerate(rates):
            threshold = 0.5  # Limiar inicial
            for i, activity in enumerate(activity_levels):
                # Atualização do limiar
                delta = rate * (activity - threshold)
                threshold += delta
                threshold = np.clip(threshold, 0.1, 0.9)
                thresholds[r, i] = threshold
        
        # Plota evolução do limiar
        for r, rate in enumerate(rates):
            ax4.plot(activity_levels, thresholds[r, :], label=f'Rate={rate}')
        
        ax4.set_title('Metaplasticity Threshold Dynamics')
        ax4.set_xlabel('Activity Level')
        ax4.set_ylabel('Plasticity Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Título principal
        fig.suptitle(f'Metaplasticity Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"metaplasticity_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_pattern_recognition_visualization(self, demo_state, assembly, timestep, output_dir):
        """Gera visualização da demonstração de reconhecimento de padrões"""
        fig = plt.figure(figsize=(12, 10))
        
        # Padrões disponíveis
        ax1 = fig.add_subplot(321)
        
        patterns = demo_state.get("patterns", {})
        if patterns:
            # Plot cada padrão
            for name, pattern in patterns.items():
                ax1.plot(pattern, label=f'Pattern {name}')
            
            ax1.set_title('Available Input Patterns')
            ax1.set_xlabel('Neuron ID')
            ax1.set_ylabel('Input Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Padrão atual com ruído
        ax2 = fig.add_subplot(322)
        
        current_pattern = demo_state.get("current_pattern")
        if current_pattern and current_pattern in patterns:
            # Obtém o padrão e adiciona ruído simulado
            pattern = patterns[current_pattern].copy()
            noise_level = demo_state.get("noise_level", 0.1)
            noisy_pattern = pattern + np.random.normal(0, noise_level, len(pattern))
            noisy_pattern = np.clip(noisy_pattern, 0, 1)
            
            ax2.plot(pattern, 'b-', label='Clean Pattern')
            ax2.plot(noisy_pattern, 'r-', alpha=0.7, label='With Noise')
            
            ax2.set_title(f'Current Input: Pattern {current_pattern}')
            ax2.set_xlabel('Neuron ID')
            ax2.set_ylabel('Input Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Resposta atual da rede
        ax3 = fig.add_subplot(323)
        
        if assembly.activation_history:
            # Visualiza padrão de ativação atual
            active_neurons = assembly.activation_history[-1]
            activation = np.zeros(assembly.size)
            for neuron in active_neurons:
                activation[neuron] = 1
            
            ax3.bar(range(assembly.size), activation)
            ax3.set_title('Current Network Response')
            ax3.set_xlabel('Neuron ID')
            ax3.set_ylabel('Active (1) / Inactive (0)')
        
        # Histórico de respostas
        ax4 = fig.add_subplot(324)
        
        pattern_history = demo_state.get("pattern_history", [])
        response_history = demo_state.get("response_history", [])
        
        if pattern_history and response_history:
            # Limita ao histórico recente
            max_history = 100
            if len(pattern_history) > max_history:
                pattern_history = pattern_history[-max_history:]
                response_history = response_history[-max_history:]
            
            # Para cada padrão, plota a força de resposta
            pattern_names = list(patterns.keys())
            
            for p_idx, pattern_name in enumerate(pattern_names):
                # Extrai resposta para este padrão
                responses = [resp.get(pattern_name, 0) for resp in response_history]
                ax4.plot(responses, label=f'Pattern {pattern_name}')
            
            # Marca quando cada padrão foi mostrado
            pattern_markers = {}
            for p_idx, pattern_name in enumerate(pattern_names):
                pattern_markers[pattern_name] = [i for i, p in enumerate(pattern_history) if p == pattern_name]
            
            # Adiciona marcadores no gráfico
            marker_colors = ['r', 'g', 'b']
            for p_idx, (pattern_name, markers) in enumerate(pattern_markers.items()):
                if markers:
                    color = marker_colors[p_idx % len(marker_colors)]
                    ax4.scatter(markers, [0.05] * len(markers), marker='|', color=color, s=100, 
                              label=f'Pattern {pattern_name} shown')
            
            ax4.set_title('Response Strength Over Time')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Response Strength')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Matriz de similaridade
        ax5 = fig.add_subplot(325)
        
        if patterns:
            # Cria uma matriz de similaridade entre padrões
            pattern_names = list(patterns.keys())
            n_patterns = len(pattern_names)
            similarity_matrix = np.zeros((n_patterns, n_patterns))
            
            for i, p1_name in enumerate(pattern_names):
                p1 = patterns[p1_name]
                for j, p2_name in enumerate(pattern_names):
                    p2 = patterns[p2_name]
                    # Calcula similaridade como correlação
                    similarity_matrix[i, j] = np.corrcoef(p1, p2)[0, 1]
            
            im = ax5.imshow(similarity_matrix, cmap='viridis', vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax5)
            
            # Adiciona labels
            ax5.set_xticks(range(n_patterns))
            ax5.set_yticks(range(n_patterns))
            ax5.set_xticklabels(pattern_names)
            ax5.set_yticklabels(pattern_names)
            
            ax5.set_title('Pattern Similarity Matrix')
        
        # Evolução da seletividade
        ax6 = fig.add_subplot(326)
        
        if len(assembly.activation_history) > 30 and patterns:
            # Para cada padrão, calculamos a seletividade da rede
            pattern_names = list(patterns.keys())
            selectivity = np.zeros(len(pattern_names))
            
            # Para cada padrão, calculamos a resposta média por região
            for p_idx, pattern_name in enumerate(pattern_names):
                pattern = patterns[pattern_name]
                active_region = np.where(pattern > 0.5)[0]
                
                # Calcula ativação média na região deste padrão
                if active_region.size > 0:
                    neuron_activation = np.zeros(assembly.size)
                    
                    # Conta ativações recentes para cada neurônio
                    for active_set in assembly.activation_history[-30:]:
                        for neuron in active_set:
                            neuron_activation[neuron] += 1
                    
                    # Normaliza
                    neuron_activation /= 30
                    
                    # Média de ativação na região ativa do padrão
                    selectivity[p_idx] = np.mean(neuron_activation[active_region])
            
            # Plota seletividade para cada padrão
            ax6.bar(pattern_names, selectivity)
            ax6.set_title('Pattern Selectivity')
            ax6.set_xlabel('Pattern')
            ax6.set_ylabel('Average Response')
            ax6.grid(True, alpha=0.3)
        
        # Título principal
        fig.suptitle(f'Pattern Recognition Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"pattern_recognition_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_neuroplasticity_visualization(self, demo_state, assembly, timestep, output_dir):
        """Gera visualização da demonstração de neuroplasticidade"""
        fig = plt.figure(figsize=(12, 10))
        
        # Mudanças de peso ao longo do tempo
        ax1 = fig.add_subplot(321)
        
        weight_history = demo_state.get("weight_history", [])
        selected_connections = demo_state.get("selected_connections", [])
        
        if weight_history and selected_connections:
            # Para cada conexão selecionada, plotamos mudanças de peso
            time_steps = range(len(weight_history))
            
            for conn_key in selected_connections:
                if conn_key in assembly.connections:
                    # Extrai histórico de peso para esta conexão
                    weights = [entry.get(conn_key, 0) for entry in weight_history]
                    
                    # Determina o tipo da conexão para coloração
                    conn_type = assembly.connections[conn_key].type
                    color = 'g' if conn_type == InteractionType.EXCITATORY else 'r'
                    
                    # Plota evolução do peso
                    ax1.plot(time_steps, weights, color=color, 
                           label=f"({conn_key[0]},{conn_key[1]})")
            
            ax1.set_title('Synaptic Weight Changes Over Time')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Weight Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Mapa de mudanças de peso
        ax2 = fig.add_subplot(322)
        
        if len(weight_history) > 1:
            # Compara primeiro e último estado para ver mudanças
            first_weights = weight_history[0] if weight_history else {}
            last_weights = weight_history[-1] if weight_history else {}
            
            # Cria matriz de mudanças
            change_matrix = np.zeros((assembly.size, assembly.size))
            
            # Para cada conexão na rede
            for (i, j), conn in assembly.connections.items():
                # Calcula a mudança de peso
                initial = first_weights.get((i, j), conn.weight)
                final = last_weights.get((i, j), conn.weight)
                change = final - initial
                
                # Armazena na matriz
                change_matrix[i, j] = change
            
            # Visualiza a matriz com colormap para aumento/diminuição
            im = ax2.imshow(change_matrix, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
            fig.colorbar(im, ax=ax2)
            
            ax2.set_title('Synaptic Weight Changes')
            ax2.set_xlabel('Post-synaptic Neuron')
            ax2.set_ylabel('Pre-synaptic Neuron')
        
        # Visualização de STDP
        ax3 = fig.add_subplot(323)
        
        # Simula função STDP
        time_diff = np.linspace(-50, 50, 100)
        
        # Função STDP
        def stdp_function(t):
            if t > 0:  # Potenciação
                return 0.1 * np.exp(-t / 20.0)
            else:  # Depressão
                return -0.1 * np.exp(t / 20.0)
        
        # Calcula valores STDP
        stdp_values = np.array([stdp_function(t) for t in time_diff])
        
        # Plota função STDP
        ax3.plot(time_diff, stdp_values)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        ax3.set_title('STDP Function')
        ax3.set_xlabel('Time Difference (post - pre) ms')
        ax3.set_ylabel('Weight Change')
        ax3.grid(True, alpha=0.3)
        
        # Distribuição de eventos STDP
        ax4 = fig.add_subplot(324)
        
        # Coleta todos os eventos STDP de todas as conexões
        all_stdp_events = []
        for conn in assembly.connections.values():
            all_stdp_events.extend(conn.stdp_window)
        
        if all_stdp_events:
            ax4.hist(all_stdp_events, bins=30, range=(-50, 50))
            ax4.set_title('STDP Event Distribution')
            ax4.set_xlabel('Time Difference (ms)')
            ax4.set_ylabel('Count')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No STDP events recorded", 
                   ha='center', va='center')
        
        # Estímulo atual
        ax5 = fig.add_subplot(325)
        
        stimulus_pattern = demo_state.get("stimulus_pattern")
        stimulus_strength = demo_state.get("stimulus_strength", 0.5)
        
        if stimulus_pattern is not None:
            # Aplica força do estímulo
            scaled_stimulus = stimulus_pattern * stimulus_strength
            
            ax5.plot(scaled_stimulus)
            ax5.set_title(f'Current Stimulus (Strength: {stimulus_strength:.2f})')
            ax5.set_xlabel('Neuron ID')
            ax5.set_ylabel('Input Strength')
            ax5.grid(True, alpha=0.3)
        
        # Relação entre STDP e mudanças de peso
        ax6 = fig.add_subplot(326)
        
        if weight_history and len(weight_history) > 1:
            # Compara mudanças de peso com contagem de eventos STDP
            first_weights = weight_history[0] if weight_history else {}
            last_weights = weight_history[-1] if weight_history else {}
            
            # Dados para o scatter plot
            weight_changes = []
            stdp_counts = []
            
            # Para cada conexão, calculamos mudança de peso e contamos eventos STDP
            for (i, j), conn in assembly.connections.items():
                # Mudança de peso
                initial = first_weights.get((i, j), conn.weight)
                final = last_weights.get((i, j), conn.weight)
                change = final - initial
                
                # Conta de eventos STDP
                stdp_count = len(conn.stdp_window)
                
                # Adiciona aos dados se houver eventos
                if stdp_count > 0 or abs(change) > 0.001:
                    weight_changes.append(change)
                    stdp_counts.append(stdp_count)
            
            if weight_changes and stdp_counts:
                # Scatter plot
                ax6.scatter(stdp_counts, weight_changes, alpha=0.5)
                
                # Tenta ajustar uma linha de tendência
                if len(weight_changes) > 2:
                    try:
                        z = np.polyfit(stdp_counts, weight_changes, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(min(stdp_counts), max(stdp_counts), 100)
                        ax6.plot(x_line, p(x_line), "r--")
                        
                        # Adiciona equação da linha
                        ax6.text(0.05, 0.95, f'y = {z[0]:.4f}x + {z[1]:.4f}', 
                               transform=ax6.transAxes, fontsize=9,
                               verticalalignment='top')
                    except:
                        pass
                
                ax6.set_title('Weight Change vs. STDP Events')
                ax6.set_xlabel('Number of STDP Events')
                ax6.set_ylabel('Weight Change')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, "Not enough data for analysis", 
                       ha='center', va='center')
        
        # Título principal
        fig.suptitle(f'Neuroplasticity Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"neuroplasticity_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_oscillatory_dynamics_visualization(self, demo_state, assembly, timestep, output_dir):
        """Gera visualização da demonstração de dinâmicas oscilatórias"""
        fig = plt.figure(figsize=(12, 10))
        
        # Série temporal de atividade neural
        ax1 = fig.add_subplot(321)
        
        activity_history = demo_state.get("activity_history", [])
        if activity_history:
            # Limitamos ao histórico recente
            max_history = 200
            if len(activity_history) > max_history:
                recent_activity = activity_history[-max_history:]
            else:
                recent_activity = activity_history
            
            time_steps = range(len(recent_activity))
            ax1.plot(time_steps, recent_activity)
            
            # Adiciona média móvel
            if len(recent_activity) > 10:
                window_size = min(20, len(recent_activity))
                activity_smooth = np.convolve(recent_activity, 
                                           np.ones(window_size)/window_size, 
                                           mode='valid')
                valid_steps = range(window_size-1, len(recent_activity))
                ax1.plot(valid_steps, activity_smooth, 'r-', linewidth=2, label='Moving Average')
                ax1.legend()
            
            ax1.set_title(f'Neural Activity - {demo_state.get("oscillation_mode", "Unknown")} Mode')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Active Neurons')
            ax1.grid(True, alpha=0.3)
        
        # Análise de frequência
        ax2 = fig.add_subplot(322)
        
        if len(activity_history) >= 64:  # Mínimo para análise de Fourier
            activity_array = np.array(activity_history[-512:])
            
            # Remove tendência e aplica janela
            activity_detrended = activity_array - np.mean(activity_array)
            windowed = activity_detrended * signal.windows.hann(len(activity_detrended))
            
            # Calcula FFT
            fft = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(len(windowed), d=0.1)  # Assumindo 10 passos = 1 segundo
            fft_mag = np.abs(fft)
            
            ax2.plot(freqs, fft_mag)
            
            # Destaca a faixa de frequência do modo atual
            oscillation_mode = demo_state.get("oscillation_mode", "")
            freq_ranges = {
                "Gamma": (30, 80),
                "Beta": (13, 30),
                "Alpha": (8, 12),
                "Theta": (4, 8)
            }
            
            if oscillation_mode in freq_ranges:
                low, high = freq_ranges[oscillation_mode]
                ax2.axvspan(low, high, alpha=0.2, color='green')
                ax2.text((low+high)/2, 0.9*max(fft_mag), 
                       f"{oscillation_mode} Band", 
                       ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7))
            
            ax2.set_title('Frequency Spectrum')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude')
            ax2.set_xlim(0, min(50, max(freqs)))  # Limita a exibição a 50Hz
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "Not enough data for frequency analysis", 
                   ha='center', va='center')
        
        # Espectrograma
        ax3 = fig.add_subplot(323)
        
        if len(activity_history) >= 128:
            activity_array = np.array(activity_history)
            
            # Calcula espectrograma
            f, t, Sxx = signal.spectrogram(activity_array, fs=10, nperseg=64, noverlap=32)
            
            # Visualiza o espectrograma
            im = ax3.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
            fig.colorbar(im, ax=ax3)
            ax3.set_title('Spectrogram')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Frequency (Hz)')
            ax3.set_ylim(0, 10)  # Limita a visualização a 10Hz
        else:
            ax3.text(0.5, 0.5, "Not enough data for spectrogram", 
                   ha='center', va='center')
        
        # Autocorrelação
        ax4 = fig.add_subplot(324)
        
        if len(activity_history) >= 50:
            activity_array = np.array(activity_history[-100:])
            
            # Normaliza a atividade
            activity_norm = activity_array - np.mean(activity_array)
            
            # Calcula autocorrelação
            autocorr = np.correlate(activity_norm, activity_norm, mode='full')
            autocorr = autocorr[len(activity_norm)-1:]
            autocorr = autocorr / autocorr[0]  # Normaliza
            
            ax4.plot(autocorr)
            ax4.set_title('Autocorrelation')
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('Correlation')
            ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Not enough data for autocorrelation", 
                   ha='center', va='center')
        
        # Padrão de entrada atual
        ax5 = fig.add_subplot(325)
        
        excitation = demo_state.get("excitation", 0.5)
        inhibition = demo_state.get("inhibition", 0.5)
        
        # Cria uma aproximação do padrão de entrada
        pattern = np.zeros(assembly.size)
        
        # Região excitatória
        exc_start = int(assembly.size * 0.2)
        exc_end = int(assembly.size * 0.4)
        pattern[exc_start:exc_end] = excitation
        
        # Região inibitória
        inh_start = int(assembly.size * 0.6)
        inh_end = int(assembly.size * 0.8)
        pattern[inh_start:inh_end] = -inhibition  # Negativo para indicar inibição
        
        ax5.bar(range(assembly.size), pattern)
        ax5.set_title('Input Pattern Configuration')
        ax5.set_xlabel('Neuron ID')
        ax5.set_ylabel('Input Level')
        ax5.set_ylim(-1, 1)
        ax5.grid(True, alpha=0.3)
        
        # Mapa de recorrência
        ax6 = fig.add_subplot(326)
        
        if len(activity_history) >= 50:
            activity_array = np.array(activity_history[-100:])
            n = len(activity_array)
            recurrence = np.zeros((n, n))
            
            # Calcula matriz de recorrência
            threshold = np.std(activity_array) * 0.2
            for i in range(n):
                for j in range(n):
                    recurrence[i, j] = 1 if abs(activity_array[i] - activity_array[j]) < threshold else 0
            
            ax6.imshow(recurrence, cmap='binary', aspect='auto')
            ax6.set_title('Recurrence Plot')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Time')
        else:
            ax6.text(0.5, 0.5, "Not enough data for recurrence plot", 
                   ha='center', va='center')
        
        # Título principal
        oscillation_mode = demo_state.get("oscillation_mode", "Unknown")
        fig.suptitle(f'Oscillatory Dynamics Analysis - {oscillation_mode} Mode (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"oscillatory_dynamics_{oscillation_mode.lower()}_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename
    
    def _generate_neuromodulation_effects_visualization(self, demo_state, assembly, timestep, output_dir):
        """Gera visualização da demonstração de efeitos de neuromodulação"""
        fig = plt.figure(figsize=(12, 10))
        
        # Histórico de níveis de neuromoduladores
        ax1 = fig.add_subplot(321)
        
        modulator_history = demo_state.get("modulator_history", [])
        
        if modulator_history:
            # Limita ao histórico recente
            max_history = 200
            if len(modulator_history) > max_history:
                modulator_history = modulator_history[-max_history:]
            
            time_steps = range(len(modulator_history))
            
            # Plota o histórico de cada neuromodulador
            for modulator in ["dopamine", "serotonin", "acetylcholine", "norepinephrine"]:
                values = [entry.get(modulator, 1.0) for entry in modulator_history]
                ax1.plot(time_steps, values, label=modulator.capitalize())
            
            ax1.set_title('Neuromodulator Levels Over Time')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Level')
            ax1.legend()
            ax1.set_ylim(0, 2.0)
            ax1.grid(True, alpha=0.3)
        
        # Histórico de resposta neural
        ax2 = fig.add_subplot(322)
        
        response_history = demo_state.get("response_history", [])
        
        if response_history:
            # Limita ao histórico recente
            max_history = 200
            if len(response_history) > max_history:
                response_history = response_history[-max_history:]
            
            time_steps = range(len(response_history))
            
            ax2.plot(time_steps, response_history)
            
            # Adiciona média móvel
            if len(response_history) > 10:
                window_size = min(20, len(response_history))
                response_smooth = np.convolve(response_history, 
                                            np.ones(window_size)/window_size, 
                                            mode='valid')
                valid_steps = range(window_size-1, len(response_history))
                ax2.plot(valid_steps, response_smooth, 'r-', linewidth=2, label='Moving Average')
                ax2.legend()
            
            ax2.set_title('Neural Response')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Active Neurons')
            ax2.grid(True, alpha=0.3)
        
        # Taxa de aprendizado efetiva
        ax3 = fig.add_subplot(323)
        
        learning_rate_history = demo_state.get("learning_rate_history", [])
        
        if learning_rate_history:
            # Limita ao histórico recente
            max_history = 200
            if len(learning_rate_history) > max_history:
                learning_rate_history = learning_rate_history[-max_history:]
            
            time_steps = range(len(learning_rate_history))
            
            ax3.plot(time_steps, learning_rate_history)
            ax3.set_title('Effective Learning Rate')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Rate')
            ax3.grid(True, alpha=0.3)
        
        # Efeitos neuromoduladores
        ax4 = fig.add_subplot(324)
        
        # Efeitos teóricos dos neuromoduladores
        effects = {
            'Learning Rate': assembly.neuromodulators.dopamine,
            'Signal-to-Noise': assembly.neuromodulators.acetylcholine,
            'Mood Stability': assembly.neuromodulators.serotonin,
            'Arousal': assembly.neuromodulators.norepinephrine
        }
        
        bars = ax4.bar(effects.keys(), effects.values())
        
        # Colorindo baseado no desvio do normal
        for i, (effect, value) in enumerate(effects.items()):
            if value > 1.1:
                bars[i].set_color('green')
            elif value < 0.9:
                bars[i].set_color('red')
            else:
                bars[i].set_color('blue')
        
        ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)  # Linha de base
        ax4.set_title('Neuromodulatory Effects')
        ax4.set_ylabel('Relative Level')
        ax4.set_ylim(0, 2.0)
        
        # Correlação neuromoduladores vs. resposta
        ax5 = fig.add_subplot(325)
        
        if modulator_history and response_history and len(modulator_history) == len(response_history) and len(modulator_history) > 10:
            # Calcula correlações entre neuromoduladores e resposta neural
            modulators = ['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine']
            correlations = []
            
            for modulator in modulators:
                mod_values = [entry.get(modulator, 1.0) for entry in modulator_history]
                correlation = np.corrcoef(mod_values, response_history)[0, 1]
                correlations.append(correlation)
            
            # Plota barras de correlação
            bars = ax5.bar(modulators, correlations)
            
            # Cor baseada no sinal da correlação
            for i, corr in enumerate(correlations):
                bars[i].set_color('green' if corr > 0 else 'red')
            
            ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax5.set_title('Correlation with Neural Activity')
            ax5.set_ylabel('Correlation')
            ax5.set_ylim(-1, 1)
        else:
            ax5.text(0.5, 0.5, "Not enough data for correlation analysis", 
                   ha='center', va='center')
        
        # Estímulo atual
        ax6 = fig.add_subplot(326)
        
        stimulus_type = demo_state.get("stimulus_type", "neutral")
        
        # Cria uma visualização do estímulo
        stimulus = np.zeros(assembly.size)
        
        if stimulus_type == "reward":
            # Estímulo associado a recompensa (região anterior)
            start = int(assembly.size * 0.2)
            end = int(assembly.size * 0.4)
            stimulus[start:end] = 1.0
            title = "Reward Stimulus"
        elif stimulus_type == "aversive":
            # Estímulo aversivo (região posterior)
            start = int(assembly.size * 0.6)
            end = int(assembly.size * 0.8)
            stimulus[start:end] = 1.0
            title = "Aversive Stimulus"
        else:  # neutral
            # Estímulo neutro (região central)
            start = int(assembly.size * 0.4)
            end = int(assembly.size * 0.6)
            stimulus[start:end] = 1.0
            title = "Neutral Stimulus"
        
        ax6.plot(stimulus)
        ax6.set_title(f'Current Input: {title}')
        ax6.set_xlabel('Neuron ID')
        ax6.set_ylabel('Input Strength')
        ax6.grid(True, alpha=0.3)
        
        # Título principal
        fig.suptitle(f'Neuromodulation Effects Analysis (Timestep {timestep})', fontsize=14)
        
        # Salva a figura
        filename = f"neuromodulation_effects_t{timestep}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        
        return filename