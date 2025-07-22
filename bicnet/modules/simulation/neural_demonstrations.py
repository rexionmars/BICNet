# neural_demonstrations.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from scipy import signal, stats
import PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QTabWidget, QLabel, QPushButton, QComboBox, QSlider,
                            QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importa as classes da rede neural
from complex_neural import ComplexNeuralAssembly, InteractionType, NeuromodulatorState
from enhanced_learning import EnhancedLearningAssembly, LearningRule

class BiologicalNeuralDemos(QMainWindow):
    """Interface para demonstrar os princípios da rede neural biológica"""
    
    def __init__(self):
        super().__init__()
        
        # Configuração da janela
        self.setWindowTitle("Biological Neural Network Demonstrations")
        self.setGeometry(100, 100, 1280, 900)
        
        # Estado da demonstração
        self.running = False
        self.update_interval = 50  # ms
        self.neural_size = 100
        self.timestep = 0
        
        # Demonstrações disponíveis
        self.demos = {
            "Pattern Recognition": self.setup_pattern_recognition,
            "Neuroplasticity Visualization": self.setup_neuroplasticity,
            "Oscillatory Dynamics": self.setup_oscillatory_dynamics,
            "Neuromodulation Effects": self.setup_neuromodulation,
            "Memory Formation": self.setup_memory_formation,
            "Sensory Integration": self.setup_sensory_integration,
            "Attractor Dynamics": self.setup_attractor_dynamics,
            "Decision Making": self.setup_decision_making
        }
        self.current_demo = "Pattern Recognition"
        
        # Assembleias neurais
        self.standard_assembly = ComplexNeuralAssembly(self.neural_size)
        self.enhanced_assembly = EnhancedLearningAssembly(self.neural_size)
        
        # Configuração de aprendizado avançado
        self.enhanced_assembly.set_learning_rule(LearningRule.REINFORCEMENT, True)
        self.enhanced_assembly.set_learning_rule(LearningRule.BCM, True)
        self.enhanced_assembly.set_learning_rule(LearningRule.HEBBIAN, True)
        
        # Assembleia atual
        self.assembly = self.enhanced_assembly
        
        # Estados das demonstrações
        self.demo_states = {}
        
        # Timer para animações
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_demo)
        
        # Configura a interface
        self.setup_ui()
        
        # Inicializa a demonstração inicial
        self.initialize_demo(self.current_demo)
    
    def setup_ui(self):
        """Configura a interface do usuário"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Layout superior - controles globais
        global_controls = QHBoxLayout()
        
        # Seleção de demonstração
        demo_selection = QHBoxLayout()
        demo_selection.addWidget(QLabel("Demonstration:"))
        self.demo_combo = QComboBox()
        self.demo_combo.addItems(list(self.demos.keys()))
        self.demo_combo.setCurrentText(self.current_demo)
        self.demo_combo.currentTextChanged.connect(self.change_demo)
        demo_selection.addWidget(self.demo_combo)
        global_controls.addLayout(demo_selection)
        
        # Tipo de assembleia
        assembly_selection = QHBoxLayout()
        assembly_selection.addWidget(QLabel("Assembly Type:"))
        self.assembly_combo = QComboBox()
        self.assembly_combo.addItems(["Standard", "Enhanced"])
        self.assembly_combo.setCurrentText("Enhanced")
        self.assembly_combo.currentTextChanged.connect(self.change_assembly_type)
        assembly_selection.addWidget(self.assembly_combo)
        global_controls.addLayout(assembly_selection)
        
        # Controles de simulação
        sim_controls = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_simulation)
        sim_controls.addWidget(self.start_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_demo)
        sim_controls.addWidget(self.reset_button)
        
        self.save_button = QPushButton("Save Data")
        self.save_button.clicked.connect(self.save_demo_data)
        sim_controls.addWidget(self.save_button)
        
        global_controls.addLayout(sim_controls)
        
        # Velocidade de simulação
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(2)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.valueChanged.connect(self.set_simulation_speed)
        speed_layout.addWidget(self.speed_slider)
        global_controls.addLayout(speed_layout)
        
        main_layout.addLayout(global_controls)
        
        # Tabs para diferentes visualizações
        self.tabs = QTabWidget()
        
        # Tab principal para a demonstração
        self.main_tab = QWidget()
        main_tab_layout = QVBoxLayout(self.main_tab)
        
        # Descrição da demonstração
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("font-size: 14px; margin: 10px;")
        main_tab_layout.addWidget(self.description_label)
        
        # Visualização principal
        self.main_figure = Figure(figsize=(10, 6), dpi=100)
        self.main_canvas = FigureCanvas(self.main_figure)
        main_tab_layout.addWidget(self.main_canvas)
        
        # Controles específicos da demonstração
        self.demo_controls_widget = QWidget()
        self.demo_controls_layout = QHBoxLayout(self.demo_controls_widget)
        main_tab_layout.addWidget(self.demo_controls_widget)
        
        self.tabs.addTab(self.main_tab, "Main Visualization")
        
        # Tab de atividade neural
        self.neural_tab = QWidget()
        neural_layout = QVBoxLayout(self.neural_tab)
        self.neural_figure = Figure(figsize=(10, 6), dpi=100)
        self.neural_canvas = FigureCanvas(self.neural_figure)
        neural_layout.addWidget(self.neural_canvas)
        self.tabs.addTab(self.neural_tab, "Neural Activity")
        
        # Tab de conectividade
        self.connectivity_tab = QWidget()
        connectivity_layout = QVBoxLayout(self.connectivity_tab)
        self.connectivity_figure = Figure(figsize=(10, 6), dpi=100)
        self.connectivity_canvas = FigureCanvas(self.connectivity_figure)
        connectivity_layout.addWidget(self.connectivity_canvas)
        self.tabs.addTab(self.connectivity_tab, "Connectivity")
        
        # Tab de parâmetros
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.params_tab = QWidget()
            params_layout = QVBoxLayout(self.params_tab)
            
            # Learning rules
            rules_group = QGroupBox("Learning Rules")
            rules_layout = QVBoxLayout()
            
            self.rule_checkboxes = {}
            for rule in LearningRule:
                checkbox = QCheckBox(rule.value.capitalize())
                checkbox.setChecked(self.enhanced_assembly.active_learning_rules[rule])
                checkbox.stateChanged.connect(lambda state, r=rule: self.toggle_learning_rule(r, state))
                self.rule_checkboxes[rule] = checkbox
                rules_layout.addWidget(checkbox)
            
            rules_group.setLayout(rules_layout)
            params_layout.addWidget(rules_group)
            
            # Learning parameters
            params_group = QGroupBox("Parameters")
            params_grid = QVBoxLayout()
            
            # Learning rate
            lr_layout = QHBoxLayout()
            lr_layout.addWidget(QLabel("Learning Rate:"))
            self.lr_spin = QDoubleSpinBox()
            self.lr_spin.setRange(0.001, 0.1)
            self.lr_spin.setSingleStep(0.001)
            self.lr_spin.setValue(self.enhanced_assembly.learning_parameters.learning_rate)
            self.lr_spin.valueChanged.connect(lambda v: setattr(self.enhanced_assembly.learning_parameters, 'learning_rate', v))
            lr_layout.addWidget(self.lr_spin)
            params_grid.addLayout(lr_layout)
            
            # Reward discount
            rd_layout = QHBoxLayout()
            rd_layout.addWidget(QLabel("Reward Discount:"))
            self.rd_spin = QDoubleSpinBox()
            self.rd_spin.setRange(0.5, 0.99)
            self.rd_spin.setSingleStep(0.01)
            self.rd_spin.setValue(self.enhanced_assembly.learning_parameters.reward_discount)
            self.rd_spin.valueChanged.connect(lambda v: setattr(self.enhanced_assembly.learning_parameters, 'reward_discount', v))
            rd_layout.addWidget(self.rd_spin)
            params_grid.addLayout(rd_layout)
            
            params_group.setLayout(params_grid)
            params_layout.addWidget(params_group)
            
            self.tabs.addTab(self.params_tab, "Learning Parameters")
        
        main_layout.addWidget(self.tabs)
        
        # Barra de status
        self.statusBar().showMessage("Ready")
    
    def toggle_learning_rule(self, rule, state):
        """Ativa ou desativa uma regra de aprendizado"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.set_learning_rule(rule, state == Qt.Checked)
    
    def change_assembly_type(self, assembly_type):
        """Muda o tipo de assembleia neural"""
        if assembly_type == "Standard":
            self.assembly = self.standard_assembly
        else:
            self.assembly = self.enhanced_assembly
        
        # Reinicia a demonstração atual
        self.reset_demo()
    
    def set_simulation_speed(self, value):
        """Define a velocidade da simulação"""
        self.update_interval = 500 // value  # 50-500ms
        if self.running:
            self.timer.stop()
            self.timer.start(self.update_interval)
    
    def toggle_simulation(self):
        """Inicia ou pausa a simulação"""
        self.running = not self.running
        if self.running:
            self.start_button.setText("Pause")
            self.timer.start(self.update_interval)
            self.statusBar().showMessage("Simulation running")
        else:
            self.start_button.setText("Start")
            self.timer.stop()
            self.statusBar().showMessage("Simulation paused")
    
    def change_demo(self, demo_name):
        """Muda a demonstração atual"""
        self.current_demo = demo_name
        self.initialize_demo(demo_name)
    
    def clear_demo_controls(self):
        """Limpa os controles específicos da demonstração atual"""
        # Remove todos os widgets do layout de controles da demonstração
        while self.demo_controls_layout.count():
            item = self.demo_controls_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def initialize_demo(self, demo_name):
        """Inicializa a demonstração selecionada"""
        # Para a simulação se estiver rodando
        if self.running:
            self.toggle_simulation()
        
        # Limpa os controles específicos da demonstração anterior
        self.clear_demo_controls()
        
        # Reseta o contador de tempo
        self.timestep = 0
        
        # Chama a função de configuração específica da demonstração
        setup_function = self.demos.get(demo_name)
        if setup_function:
            setup_function()
        
        # Atualiza as visualizações
        self.update_main_visualization()
        self.update_neural_visualization()
        self.update_connectivity_visualization()
        
        self.statusBar().showMessage(f"Demonstration {demo_name} initialized")
    
    def reset_demo(self):
        """Reinicia a demonstração atual"""
        # Reinicializa a demonstração
        self.initialize_demo(self.current_demo)
    
    def save_demo_data(self):
        """Salva os dados da demonstração atual"""
        # Obtém o estado atual da demonstração
        data = self.demo_states.get(self.current_demo, {})
        
        # Adiciona informações da rede neural
        data['weights'] = {k: v.weight for k, v in self.assembly.connections.items()}
        
        if isinstance(self.assembly, EnhancedLearningAssembly):
            data['reward_history'] = self.assembly.reward_history
        
        # Salva em arquivo
        filename, _ = QFileDialog.getSaveFileName(self, "Save Demo Data", "", "NPZ Files (*.npz)")
        if filename:
            np.savez(filename, **data)
            self.statusBar().showMessage(f"Data saved to {filename}")
    
    def update_demo(self):
        """Atualiza a demonstração atual a cada passo de tempo"""
        # Incrementa o contador de tempo
        self.timestep += 1
        
        # Chama a função de atualização correspondente à demonstração atual
        update_function = getattr(self, f"update_{self.current_demo.lower().replace(' ', '_')}", None)
        if update_function:
            update_function()
        
        # Atualiza as visualizações (não todas a cada passo para melhor desempenho)
        self.update_main_visualization()
        
        if self.timestep % 5 == 0:
            self.update_neural_visualization()
        
        if self.timestep % 10 == 0:
            self.update_connectivity_visualization()
    
    def update_main_visualization(self):
        """Atualiza a visualização principal da demonstração atual"""
        # Esta função será sobrescrita pelas demonstrações específicas
        pass
    
    def update_neural_visualization(self):
        """Atualiza a visualização de atividade neural"""
        self.neural_figure.clear()
        
        # Layout da figura
        gs = gridspec.GridSpec(2, 2)
        
        # Atividade recente
        ax1 = self.neural_figure.add_subplot(gs[0, 0])
        history_len = min(100, len(self.assembly.activation_history))
        if history_len > 0:
            recent = self.assembly.activation_history[-history_len:]
            activity = [len(act) for act in recent]
            ax1.plot(range(len(activity)), activity)
            ax1.set_title('Neural Activity')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Active Neurons')
        
        # Níveis de cálcio
        ax2 = self.neural_figure.add_subplot(gs[0, 1])
        ax2.plot(self.assembly.calcium_levels)
        ax2.set_title('Calcium Levels')
        
        # Distribuição de ativações
        ax3 = self.neural_figure.add_subplot(gs[1, 0])
        if history_len > 0:
            neuron_activity = np.zeros(self.assembly.size)
            for active_set in self.assembly.activation_history[-50:]:
                for neuron in active_set:
                    neuron_activity[neuron] += 1
            ax3.bar(range(self.assembly.size), neuron_activity / 50)
            ax3.set_title('Neuron Activation Frequency')
            ax3.set_xlabel('Neuron ID')
            ax3.set_ylabel('Activation Probability')
        
        # Níveis de neuromoduladores
        ax4 = self.neural_figure.add_subplot(gs[1, 1])
        levels = [
            self.assembly.neuromodulators.dopamine,
            self.assembly.neuromodulators.serotonin,
            self.assembly.neuromodulators.acetylcholine,
            self.assembly.neuromodulators.norepinephrine
        ]
        ax4.bar(['DA', '5-HT', 'ACh', 'NE'], levels)
        ax4.set_title('Neuromodulator Levels')
        
        self.neural_figure.tight_layout()
        self.neural_canvas.draw()
    
    def update_connectivity_visualization(self):
        """Atualiza a visualização de conectividade"""
        self.connectivity_figure.clear()
        
        # Layout da figura
        gs = gridspec.GridSpec(2, 2)
        
        # Matriz de conectividade
        ax1 = self.connectivity_figure.add_subplot(gs[0, 0])
        matrix = np.zeros((self.assembly.size, self.assembly.size))
        for (i, j), conn in self.assembly.connections.items():
            matrix[i, j] = conn.weight if conn.type == InteractionType.EXCITATORY else -conn.weight
        im = ax1.imshow(matrix, cmap='RdBu_r')
        self.connectivity_figure.colorbar(im, ax=ax1)
        ax1.set_title('Connectivity Matrix')
        
        # Distribuição de pesos
        ax2 = self.connectivity_figure.add_subplot(gs[0, 1])
        weights = [conn.weight for conn in self.assembly.connections.values()]
        ax2.hist(weights, bins=20)
        ax2.set_title('Weight Distribution')
        
        # Visualização de grafo
        ax3 = self.connectivity_figure.add_subplot(gs[1, 0])
        G = nx.DiGraph()
        
        # Adiciona apenas conexões fortes para clareza
        for (i, j), conn in self.assembly.connections.items():
            if conn.weight > 0.6:
                G.add_edge(i, j, weight=conn.weight, 
                          color='g' if conn.type == InteractionType.EXCITATORY else 'r')
        
        if G.number_of_edges() > 0:
            # Limita o número de nós para clareza
            if G.number_of_nodes() > 20:
                # Seleciona os nós com maior grau
                degree = dict(G.degree())
                top_nodes = sorted(degree.keys(), key=lambda x: degree[x], reverse=True)[:20]
                G = G.subgraph(top_nodes)
            
            pos = nx.spring_layout(G, seed=42)
            edge_colors = [G[u][v]['color'] for u, v in G.edges()]
            nx.draw(G, pos, edge_color=edge_colors, with_labels=True, node_size=300, ax=ax3)
        
        ax3.set_title('Strong Connections Graph')
        
        # Visualização especializada para a versão aprimorada
        ax4 = self.connectivity_figure.add_subplot(gs[1, 1])
        if isinstance(self.assembly, EnhancedLearningAssembly):
            if self.assembly.neural_assemblies:
                # Visualização de assembleias neurais
                assembly_matrix = np.zeros(self.assembly.size)
                for i, neurons in enumerate(self.assembly.neural_assemblies):
                    for neuron in neurons:
                        assembly_matrix[neuron] = i + 1
                
                im = ax4.bar(range(self.assembly.size), assembly_matrix)
                ax4.set_title('Neural Assemblies')
                ax4.set_xlabel('Neuron ID')
                ax4.set_ylabel('Assembly ID')
            else:
                # Visualização de traços de elegibilidade
                eligibility_matrix = np.zeros((self.assembly.size, self.assembly.size))
                for (i, j), trace in self.assembly.eligibility_traces.items():
                    eligibility_matrix[i, j] = trace
                im = ax4.imshow(eligibility_matrix, cmap='plasma')
                self.connectivity_figure.colorbar(im, ax=ax4)
                ax4.set_title('Eligibility Traces')
        else:
            # Para a versão padrão, visualiza a síntese de proteínas
            ax4.plot(self.assembly.protein_synthesis)
            ax4.set_title('Protein Synthesis')
        
        self.connectivity_figure.tight_layout()
        self.connectivity_canvas.draw()
    
    # === DEMONSTRAÇÕES ESPECÍFICAS ===
    
    def setup_pattern_recognition(self):
        """Configuração da demonstração de reconhecimento de padrões"""
        # Descrição
        self.description_label.setText(
            "Pattern Recognition Demonstration\n\n"
            "This demonstration shows how the neural network can learn to recognize and differentiate "
            "between different input patterns. Over time, the network develops specialized neural assemblies "
            "that respond selectively to specific patterns, mimicking how our brain recognizes objects."
        )
        
        # Estado da demonstração
        if "Pattern Recognition" not in self.demo_states:
            self.demo_states["Pattern Recognition"] = {
                "patterns": {
                    "A": np.zeros(self.assembly.size),
                    "B": np.zeros(self.assembly.size),
                    "C": np.zeros(self.assembly.size)
                },
                "current_pattern": "A",
                "pattern_history": [],
                "response_history": []
            }
        
        state = self.demo_states["Pattern Recognition"]
        
        # Define os padrões
        a_start = int(self.assembly.size * 0.1)
        a_end = int(self.assembly.size * 0.3)
        state["patterns"]["A"][a_start:a_end] = 1
        
        b_start = int(self.assembly.size * 0.4)
        b_end = int(self.assembly.size * 0.6)
        state["patterns"]["B"][b_start:b_end] = 1
        
        c_start = int(self.assembly.size * 0.7)
        c_end = int(self.assembly.size * 0.9)
        state["patterns"]["C"][c_start:c_end] = 1
        
        # Adiciona controles
        pattern_group = QGroupBox("Input Pattern")
        pattern_layout = QVBoxLayout()
        
        # Botões de seleção de padrão
        pattern_buttons = QHBoxLayout()
        for pattern_name in state["patterns"].keys():
            button = QPushButton(f"Pattern {pattern_name}")
            button.clicked.connect(lambda checked, p=pattern_name: self.select_pattern(p))
            pattern_buttons.addWidget(button)
        
        pattern_layout.addLayout(pattern_buttons)
        
        # Controle de ruído
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise Level:"))
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(20)
        noise_layout.addWidget(self.noise_slider)
        pattern_layout.addLayout(noise_layout)
        
        pattern_group.setLayout(pattern_layout)
        self.demo_controls_layout.addWidget(pattern_group)
        
        # Configurações de recompensa
        if isinstance(self.assembly, EnhancedLearningAssembly):
            reward_group = QGroupBox("Reward Settings")
            reward_layout = QVBoxLayout()
            
            # Recompensa para cada padrão
            for pattern_name in state["patterns"].keys():
                reward_pattern = QHBoxLayout()
                reward_pattern.addWidget(QLabel(f"Reward for Pattern {pattern_name}:"))
                slider = QSlider(Qt.Horizontal)
                slider.setRange(-100, 100)
                slider.setValue(50 if pattern_name == "A" else 0)
                reward_pattern.addWidget(slider)
                reward_pattern.addWidget(QLabel("0.5" if pattern_name == "A" else "0.0"))
                
                # Armazena o slider para uso posterior
                state[f"reward_slider_{pattern_name}"] = slider
                
                reward_layout.addLayout(reward_pattern)
            
            reward_group.setLayout(reward_layout)
            self.demo_controls_layout.addWidget(reward_group)
    
    def select_pattern(self, pattern_name):
        """Seleciona um padrão de entrada para a demonstração de reconhecimento de padrões"""
        state = self.demo_states["Pattern Recognition"]
        state["current_pattern"] = pattern_name
    
    def update_pattern_recognition(self):
        """Atualiza a demonstração de reconhecimento de padrões"""
        state = self.demo_states["Pattern Recognition"]
        
        # Obtém o padrão atual
        current_pattern = state["patterns"][state["current_pattern"]].copy()
        
        # Adiciona ruído
        noise_level = self.noise_slider.value() / 500  # 0-0.2
        current_pattern += np.random.normal(0, noise_level, self.assembly.size)
        current_pattern = np.clip(current_pattern, 0, 1)
        
        # Obtém recompensa para o padrão atual
        reward = 0.0
        if isinstance(self.assembly, EnhancedLearningAssembly):
            reward_slider = state.get(f"reward_slider_{state['current_pattern']}")
            if reward_slider:
                reward = reward_slider.value() / 100  # -1.0 to 1.0
        
        # Atualiza a rede neural
        if isinstance(self.assembly, EnhancedLearningAssembly):
            activation = self.assembly.update(current_pattern, self.timestep, reward)
        else:
            activation = self.assembly.update(current_pattern, self.timestep)
        
        # Registra o histórico
        state["pattern_history"].append(state["current_pattern"])
        
        # Calculamos a resposta para cada região correspondente aos padrões
        responses = {}
        for pattern_name, pattern in state["patterns"].items():
            # Identifica a região ativa para este padrão
            active_region = np.where(pattern > 0.5)[0]
            # Calcula a ativação média nesta região
            if len(active_region) > 0:
                responses[pattern_name] = np.mean(activation[active_region])
            else:
                responses[pattern_name] = 0
        
        state["response_history"].append(responses)
        
        # Para manter o histórico limitado
        if len(state["pattern_history"]) > 500:
            state["pattern_history"] = state["pattern_history"][-500:]
            state["response_history"] = state["response_history"][-500:]
    
    def update_main_visualization(self):
        """Atualiza a visualização principal com base na demonstração atual"""
        # Seleciona a função de visualização apropriada
        viz_function = getattr(self, f"visualize_{self.current_demo.lower().replace(' ', '_')}", None)
        if viz_function:
            viz_function()
        else:
            # Visualização padrão se não houver específica
            self.main_figure.clear()
            ax = self.main_figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Visualization for {self.current_demo}\nTimestep: {self.timestep}", 
                   ha='center', va='center', fontsize=14)
            self.main_canvas.draw()
    
    def visualize_pattern_recognition(self):
        """Visualiza a demonstração de reconhecimento de padrões"""
        state = self.demo_states["Pattern Recognition"]
        
        self.main_figure.clear()
        gs = gridspec.GridSpec(2, 2)
        
        # Padrão atual
        ax1 = self.main_figure.add_subplot(gs[0, 0])
        current_pattern = state["patterns"][state["current_pattern"]].copy()
        noise_level = self.noise_slider.value() / 500
        current_pattern += np.random.normal(0, noise_level, self.assembly.size)
        current_pattern = np.clip(current_pattern, 0, 1)
        ax1.plot(current_pattern)
        ax1.set_title(f'Current Pattern: {state["current_pattern"]}')
        ax1.set_ylim(0, 1.2)
        
        # Resposta da rede
        ax2 = self.main_figure.add_subplot(gs[0, 1])
        if self.assembly.activation_history:
            last_activation = np.zeros(self.assembly.size)
            for neuron in self.assembly.activation_history[-1]:
                last_activation[neuron] = 1
            ax2.plot(last_activation)
            ax2.set_title('Network Response')
            ax2.set_ylim(0, 1.2)
        
        # Histórico de seletividade
        ax3 = self.main_figure.add_subplot(gs[1, :])
        if state["response_history"]:
            # Prepara os dados para o gráfico
            history_length = len(state["response_history"])
            x = range(history_length)
            
            # Para cada padrão, plotamos a resposta ao longo do tempo
            for pattern_name in state["patterns"].keys():
                y = [resp[pattern_name] for resp in state["response_history"]]
                ax3.plot(x, y, label=f'Pattern {pattern_name}')
            
            # Marca os momentos de apresentação de
            # Marca os momentos de apresentação de cada padrão
            patterns_shown = state["pattern_history"]
            
            # Cria marcadores coloridos para cada tipo de padrão
            pattern_colors = {'A': 'r', 'B': 'g', 'C': 'b'}
            for p, color in pattern_colors.items():
                # Encontra os índices onde este padrão foi mostrado
                indices = [i for i, pattern in enumerate(patterns_shown) if pattern == p]
                # Marca no eixo x
                if indices:
                    ax3.scatter(indices, [0.05] * len(indices), 
                               marker='|', color=color, s=100, 
                               label=f'Pattern {p} shown')
            
            ax3.set_title('Pattern Selectivity Development')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Response Strength')
            ax3.legend(loc='upper left')
            ax3.set_ylim(0, 1.0)
            ax3.grid(True, alpha=0.3)
        
        self.main_figure.tight_layout()
        self.main_canvas.draw()

    # === OUTRAS DEMONSTRAÇÕES ===
    
    def setup_neuroplasticity(self):
        """Configuração da demonstração de neuroplasticidade"""
        # Descrição
        self.description_label.setText(
            "Neuroplasticity Visualization\n\n"
            "This demonstration visualizes how neural connections change over time in response to stimuli. "
            "It illustrates various forms of plasticity including STDP, homeostatic plasticity, and metaplasticity. "
            "You can observe how learning processes modify the neural network's structure, similar to how "
            "your brain physically rewires itself during learning."
        )
        
        # Estado da demonstração
        if "Neuroplasticity" not in self.demo_states:
            self.demo_states["Neuroplasticity"] = {
                "weight_history": [],
                "selected_connections": [],
                "stimulus_strength": 0.5,
                "stimulus_pattern": np.zeros(self.assembly.size)
            }
        
        state = self.demo_states["Neuroplasticity"]
        
        # Seleciona algumas conexões para rastrear
        if not state["selected_connections"]:
            connections = list(self.assembly.connections.keys())
            if connections:
                # Seleciona 5 conexões aleatórias
                n_connections = min(5, len(connections))
                state["selected_connections"] = np.random.choice(
                    connections, n_connections, replace=False
                ).tolist()
        
        # Define um padrão de estímulo
        start = int(self.assembly.size * 0.3)
        end = int(self.assembly.size * 0.7)
        state["stimulus_pattern"] = np.zeros(self.assembly.size)
        state["stimulus_pattern"][start:end] = 1
        
        # Adiciona controles
        stimulus_group = QGroupBox("Stimulus Control")
        stimulus_layout = QVBoxLayout()
        
        # Controle de força do estímulo
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Stimulus Strength:"))
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(int(state["stimulus_strength"] * 100))
        self.strength_slider.valueChanged.connect(self.set_stimulus_strength)
        strength_layout.addWidget(self.strength_slider)
        stimulus_layout.addLayout(strength_layout)
        
        # Botões para padrões de estímulo
        pattern_buttons = QHBoxLayout()
        
        focused_button = QPushButton("Focused Stimulus")
        focused_button.clicked.connect(self.set_focused_stimulus)
        pattern_buttons.addWidget(focused_button)
        
        distributed_button = QPushButton("Distributed Stimulus")
        distributed_button.clicked.connect(self.set_distributed_stimulus)
        pattern_buttons.addWidget(distributed_button)
        
        random_button = QPushButton("Random Stimulus")
        random_button.clicked.connect(self.set_random_stimulus)
        pattern_buttons.addWidget(random_button)
        
        stimulus_layout.addLayout(pattern_buttons)
        
        stimulus_group.setLayout(stimulus_layout)
        self.demo_controls_layout.addWidget(stimulus_group)
        
        # Controles de visualização
        vis_group = QGroupBox("Visualization Options")
        vis_layout = QVBoxLayout()
        
        # Tipo de visualização
        vis_type = QHBoxLayout()
        vis_type.addWidget(QLabel("Visualization Type:"))
        self.vis_combo = QComboBox()
        self.vis_combo.addItems(["Weight Changes", "Connection Map", "Plasticity Mechanisms"])
        self.vis_combo.setCurrentText("Weight Changes")
        vis_type.addWidget(self.vis_combo)
        vis_layout.addLayout(vis_type)
        
        vis_group.setLayout(vis_layout)
        self.demo_controls_layout.addWidget(vis_group)
    
    def set_stimulus_strength(self):
        """Define a força do estímulo para a demonstração de neuroplasticidade"""
        state = self.demo_states["Neuroplasticity"]
        state["stimulus_strength"] = self.strength_slider.value() / 100
    
    def set_focused_stimulus(self):
        """Define um estímulo focado para a demonstração de neuroplasticidade"""
        state = self.demo_states["Neuroplasticity"]
        state["stimulus_pattern"] = np.zeros(self.assembly.size)
        start = int(self.assembly.size * 0.4)
        end = int(self.assembly.size * 0.6)
        state["stimulus_pattern"][start:end] = 1
    
    def set_distributed_stimulus(self):
        """Define um estímulo distribuído para a demonstração de neuroplasticidade"""
        state = self.demo_states["Neuroplasticity"]
        state["stimulus_pattern"] = np.zeros(self.assembly.size)
        
        # Cria três regiões ativas
        regions = [
            (int(self.assembly.size * 0.1), int(self.assembly.size * 0.2)),
            (int(self.assembly.size * 0.4), int(self.assembly.size * 0.5)),
            (int(self.assembly.size * 0.7), int(self.assembly.size * 0.8))
        ]
        
        for start, end in regions:
            state["stimulus_pattern"][start:end] = 1
    
    def set_random_stimulus(self):
        """Define um estímulo aleatório para a demonstração de neuroplasticidade"""
        state = self.demo_states["Neuroplasticity"]
        state["stimulus_pattern"] = np.random.binomial(1, 0.3, self.assembly.size)
    
    def update_neuroplasticity(self):
        """Atualiza a demonstração de neuroplasticidade"""
        state = self.demo_states["Neuroplasticity"]
        
        # Obtém o padrão de estímulo atual
        stimulus = state["stimulus_pattern"].copy() * state["stimulus_strength"]
        
        # Atualiza a rede neural
        self.assembly.update(stimulus, self.timestep)
        
        # Registra o histórico de pesos
        weights = {}
        for conn_key in state["selected_connections"]:
            if conn_key in self.assembly.connections:
                weights[conn_key] = self.assembly.connections[conn_key].weight
        
        state["weight_history"].append(weights)
        
        # Limita o tamanho do histórico
        if len(state["weight_history"]) > 200:
            state["weight_history"] = state["weight_history"][-200:]
    
    def visualize_neuroplasticity(self):
        """Visualiza a demonstração de neuroplasticidade"""
        state = self.demo_states["Neuroplasticity"]
        
        self.main_figure.clear()
        
        # Tipo de visualização
        vis_type = self.vis_combo.currentText() if hasattr(self, 'vis_combo') else "Weight Changes"
        
        if vis_type == "Weight Changes":
            # Visualiza as mudanças de peso ao longo do tempo
            ax = self.main_figure.add_subplot(111)
            
            if state["weight_history"]:
                # Prepara os dados para o gráfico
                time_steps = range(len(state["weight_history"]))
                
                # Para cada conexão selecionada, plotamos seu peso ao longo do tempo
                for conn_key in state["selected_connections"]:
                    weights = [entry.get(conn_key, 0) for entry in state["weight_history"]]
                    if conn_key in self.assembly.connections:
                        conn_type = self.assembly.connections[conn_key].type
                        color = 'g' if conn_type == InteractionType.EXCITATORY else 'r'
                        label = f"({conn_key[0]},{conn_key[1]})"
                        ax.plot(time_steps, weights, color=color, label=label)
                
                ax.set_title('Synaptic Weight Changes Over Time')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Weight')
                ax.legend(loc='upper left')
                ax.set_ylim(0, 1.0)
                ax.grid(True, alpha=0.3)
            
        elif vis_type == "Connection Map":
            # Visualiza um mapa das conexões e suas mudanças
            ax = self.main_figure.add_subplot(111)
            
            # Cria matriz de conectividade
            matrix = np.zeros((self.assembly.size, self.assembly.size))
            
            # Calcula as mudanças de peso se houver histórico
            if len(state["weight_history"]) > 1:
                first = state["weight_history"][0]
                last = state["weight_history"][-1]
                
                for (i, j), conn in self.assembly.connections.items():
                    # Calcula a mudança de peso
                    initial = first.get((i, j), conn.weight)
                    final = last.get((i, j), conn.weight)
                    change = final - initial
                    
                    # Cores: vermelho para diminuição, verde para aumento
                    matrix[i, j] = change
            
            # Visualiza a matriz
            im = ax.imshow(matrix, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
            self.main_figure.colorbar(im, ax=ax, label='Weight Change')
            ax.set_title('Synaptic Weight Changes')
            ax.set_xlabel('Post-synaptic Neuron')
            ax.set_ylabel('Pre-synaptic Neuron')
            
        elif vis_type == "Plasticity Mechanisms":
            # Visualiza os diferentes mecanismos de plasticidade
            gs = gridspec.GridSpec(2, 2)
            
            # STDP
            ax1 = self.main_figure.add_subplot(gs[0, 0])
            # Simula a curva STDP
            time_diff = np.linspace(-50, 50, 100)
            stdp_curve = np.zeros_like(time_diff)
            
            for i, t in enumerate(time_diff):
                if t > 0:  # Potenciação
                    stdp_curve[i] = 0.1 * np.exp(-t / 20.0)
                else:  # Depressão
                    stdp_curve[i] = -0.1 * np.exp(t / 20.0)
            
            ax1.plot(time_diff, stdp_curve)
            ax1.set_title('STDP Function')
            ax1.set_xlabel('Time Difference (post - pre)')
            ax1.set_ylabel('Weight Change')
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # Homeostatic Plasticity
            ax2 = self.main_figure.add_subplot(gs[0, 1])
            if len(self.assembly.activation_history) > 0:
                # Calcula a atividade média
                activity = [len(act) for act in self.assembly.activation_history]
                mean_activity = np.mean(activity[-100:]) if len(activity) > 100 else np.mean(activity)
                target = self.assembly.size * 0.1  # 10% target activity
                
                # Visualiza barras para atividade atual, alvo e diferença
                bars = ['Current', 'Target', 'Difference']
                values = [mean_activity, target, mean_activity - target]
                colors = ['blue', 'green', 'red' if values[2] > 0 else 'green']
                
                ax2.bar(bars, values, color=colors)
                ax2.set_title('Homeostatic Plasticity')
                ax2.set_ylabel('Activity Level')
                
                # Adiciona texto explicando o efeito
                effect = "Downscaling" if values[2] > 0 else "Upscaling"
                ax2.text(0.5, -0.1, f"Effect: {effect}", 
                        ha='center', transform=ax2.transAxes)
            
            # Metaplasticidade (só para EnhancedLearningAssembly)
            ax3 = self.main_figure.add_subplot(gs[1, 0])
            if isinstance(self.assembly, EnhancedLearningAssembly):
                ax3.plot(self.assembly.plasticity_thresholds)
                ax3.set_title('Metaplasticity Thresholds')
                ax3.set_xlabel('Neuron ID')
                ax3.set_ylabel('Threshold')
            else:
                ax3.text(0.5, 0.5, "Metaplasticity\nOnly available with\nEnhanced Assembly", 
                        ha='center', va='center')
            
            # Neuromodulação
            ax4 = self.main_figure.add_subplot(gs[1, 1])
            
            # Simula o efeito dos neuromoduladores na plasticidade
            neuromod_labels = ['Baseline', 'High Dopamine', 'High ACh', 'Low Dopamine']
            modulation_factors = [1.0, 1.5, 1.2, 0.5]
            
            # Multiplicamos a curva STDP pelos fatores de modulação
            for i, (label, factor) in enumerate(zip(neuromod_labels, modulation_factors)):
                ax4.plot(time_diff, stdp_curve * factor, label=label)
            
            ax4.set_title('Neuromodulation of STDP')
            ax4.set_xlabel('Time Difference')
            ax4.set_ylabel('Modulated Weight Change')
            ax4.legend(loc='lower right', fontsize='small')
            ax4.grid(True, alpha=0.3)
        
        self.main_figure.tight_layout()
        self.main_canvas.draw()
    
    def setup_oscillatory_dynamics(self):
        """Configuração da demonstração de dinâmicas oscilatórias"""
        # Descrição
        self.description_label.setText(
            "Oscillatory Dynamics\n\n"
            "This demonstration visualizes how neural populations can generate oscillatory activity "
            "through excitatory-inhibitory interactions. You can observe different oscillation patterns "
            "like gamma, beta, and theta rhythms that emerge from the network dynamics, similar to "
            "brain waves observed in EEG recordings."
        )
        
        # Estado da demonstração
        if "Oscillatory Dynamics" not in self.demo_states:
            self.demo_states["Oscillatory Dynamics"] = {
                "oscillation_mode": "Gamma",
                "activity_history": [],
                "frequency_history": [],
                "visualization_mode": "Time Domain"
            }
        
        state = self.demo_states["Oscillatory Dynamics"]
        
        # Adiciona controles
        oscillation_group = QGroupBox("Oscillation Settings")
        oscillation_layout = QVBoxLayout()
        
        # Seleção do modo de oscilação
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Oscillation Mode:"))
        self.oscillation_combo = QComboBox()
        self.oscillation_combo.addItems(["Gamma (30-80 Hz)", "Beta (13-30 Hz)", "Alpha (8-12 Hz)", "Theta (4-8 Hz)"])
        self.oscillation_combo.setCurrentText("Gamma (30-80 Hz)")
        self.oscillation_combo.currentTextChanged.connect(self.set_oscillation_mode)
        mode_layout.addWidget(self.oscillation_combo)
        oscillation_layout.addLayout(mode_layout)
        
        # Controle de excitação
        excitation_layout = QHBoxLayout()
        excitation_layout.addWidget(QLabel("Excitation Level:"))
        self.excitation_slider = QSlider(Qt.Horizontal)
        self.excitation_slider.setRange(10, 100)
        self.excitation_slider.setValue(50)
        excitation_layout.addWidget(self.excitation_slider)
        oscillation_layout.addLayout(excitation_layout)
        
        # Controle de inibição
        inhibition_layout = QHBoxLayout()
        inhibition_layout.addWidget(QLabel("Inhibition Level:"))
        self.inhibition_slider = QSlider(Qt.Horizontal)
        self.inhibition_slider.setRange(10, 100)
        self.inhibition_slider.setValue(50)
        inhibition_layout.addWidget(self.inhibition_slider)
        oscillation_layout.addLayout(inhibition_layout)
        
        oscillation_group.setLayout(oscillation_layout)
        self.demo_controls_layout.addWidget(oscillation_group)
        
        # Controles de visualização
        vis_group = QGroupBox("Visualization Options")
        vis_layout = QVBoxLayout()
        
        # Tipo de visualização
        vis_layout.addWidget(QLabel("Visualization Type:"))
        self.vis_radio_time = QRadioButton("Time Domain")
        self.vis_radio_time.setChecked(True)
        self.vis_radio_time.toggled.connect(lambda: self.set_oscillation_visualization("Time Domain"))
        vis_layout.addWidget(self.vis_radio_time)
        
        self.vis_radio_freq = QRadioButton("Frequency Domain")
        self.vis_radio_freq.toggled.connect(lambda: self.set_oscillation_visualization("Frequency Domain"))
        vis_layout.addWidget(self.vis_radio_freq)
        
        self.vis_radio_phase = QRadioButton("Phase Space")
        self.vis_radio_phase.toggled.connect(lambda: self.set_oscillation_visualization("Phase Space"))
        vis_layout.addWidget(self.vis_radio_phase)
        
        vis_group.setLayout(vis_layout)
        self.demo_controls_layout.addWidget(vis_group)
    
    def set_oscillation_mode(self, mode_text):
        """Define o modo de oscilação para a demonstração de dinâmicas oscilatórias"""
        state = self.demo_states["Oscillatory Dynamics"]
        state["oscillation_mode"] = mode_text.split(" ")[0]  # Extrai só o nome do modo
    
    def set_oscillation_visualization(self, vis_mode):
        """Define o modo de visualização para a demonstração de dinâmicas oscilatórias"""
        state = self.demo_states["Oscillatory Dynamics"]
        state["visualization_mode"] = vis_mode
    
    def update_oscillatory_dynamics(self):
        """Atualiza a demonstração de dinâmicas oscilatórias"""
        state = self.demo_states["Oscillatory Dynamics"]
        
        # Configura parâmetros de oscilação baseado no modo
        excitation = self.excitation_slider.value() / 100
        inhibition = self.inhibition_slider.value() / 100
        
        # Definimos os diferentes padrões de entrada para os modos
        oscillation_mode = state["oscillation_mode"]
        
        # Padrão base
        input_pattern = np.zeros(self.assembly.size)
        
        if oscillation_mode == "Gamma":
            # Gamma: oscilações rápidas e localizadas
            frequency = 0.2  # mais rápido
            duration = 0.3   # mais curto
        elif oscillation_mode == "Beta":
            # Beta: oscilações de amplitude média
            frequency = 0.1
            duration = 0.5
        elif oscillation_mode == "Alpha":
            # Alpha: oscilações mais lentas
            frequency = 0.05
            duration = 0.7
        else:  # Theta
            # Theta: oscilações lentas e amplas
            frequency = 0.025  # mais lento
            duration = 0.9    # mais longo
        
        # Gera um padrão oscilatório
        t = self.timestep * frequency
        amplitude = (np.sin(t) + 1) / 2  # Escala para [0, 1]
        
        # Região de neurônios excitatórios
        exc_start = int(self.assembly.size * 0.2)
        exc_end = int(self.assembly.size * 0.4)
        input_pattern[exc_start:exc_end] = amplitude * excitation
        
        # Região de neurônios inibitórios (ativamos com atraso de fase)
        inh_start = int(self.assembly.size * 0.6)
        inh_end = int(self.assembly.size * 0.8)
        input_pattern[inh_start:inh_end] = ((1-amplitude) * inhibition) * duration
        
        # Atualiza a rede neural
        activation = self.assembly.update(input_pattern, self.timestep)
        
        # Registra o histórico de atividade
        activity_level = len(self.assembly.activation_history[-1]) if self.assembly.activation_history else 0
        state["activity_history"].append(activity_level)
        
        # Limita o tamanho do histórico
        max_history = 1000
        if len(state["activity_history"]) > max_history:
            state["activity_history"] = state["activity_history"][-max_history:]
        
        # Periodicamente, calcula o espectro de frequência
        if self.timestep % 10 == 0 and len(state["activity_history"]) > 50:
            # Usa FFT para obter o espectro
            activity_array = np.array(state["activity_history"][-512:])  # Usa potência de 2 para FFT
            if len(activity_array) >= 64:  # Mínimo de pontos
                # Remove tendência e aplica janela
                activity_detrended = activity_array - np.mean(activity_array)
                windowed = activity_detrended * signal.windows.hann(len(activity_detrended))
                
                # Calcula FFT
                fft = np.fft.rfft(windowed)
                freqs = np.fft.rfftfreq(len(windowed), d=0.1)  # Assumindo 10 passos = 1 segundo
                fft_mag = np.abs(fft)
                
                # Guarda o espectro
                state["frequency_history"] = list(zip(freqs, fft_mag))
    
    def visualize_oscillatory_dynamics(self):
        """Visualiza a demonstração de dinâmicas oscilatórias"""
        state = self.demo_states["Oscillatory Dynamics"]
        
        self.main_figure.clear()
        
        # Tipo de visualização
        vis_mode = state["visualization_mode"]
        
        if vis_mode == "Time Domain":
            # Séries temporais de atividade
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            
            # Gráfico principal de atividade
            ax1 = self.main_figure.add_subplot(gs[0])
            
            if state["activity_history"]:
                # Plotamos o histórico de atividade
                activity = state["activity_history"]
                time_steps = range(len(activity))
                ax1.plot(time_steps, activity)
                
                # Adiciona uma linha para a atividade média
                if len(activity) > 10:
                    window_size = min(50, len(activity))
                    activity_smooth = np.convolve(activity, np.ones(window_size)/window_size, mode='valid')
                    valid_steps = range(window_size-1, len(activity))
                    ax1.plot(valid_steps, activity_smooth, 'r-', linewidth=2)
                
                ax1.set_title(f'{state["oscillation_mode"]} Oscillation Pattern')
                ax1.set_ylabel('Active Neurons')
                ax1.grid(True, alpha=0.3)
                
                # Limita a visualização aos últimos 500 passos
                if len(time_steps) > 500:
                    ax1.set_xlim(time_steps[-500], time_steps[-1])
            
            # Padrão de entrada atual
            ax2 = self.main_figure.add_subplot(gs[1])
            excitation = self.excitation_slider.value() / 100
            inhibition = self.inhibition_slider.value() / 100
            
            # Cria um padrão aproximado do que está sendo usado
            pattern = np.zeros(self.assembly.size)
            exc_start = int(self.assembly.size * 0.2)
            exc_end = int(self.assembly.size * 0.4)
            pattern[exc_start:exc_end] = excitation
            
            inh_start = int(self.assembly.size * 0.6)
            inh_end = int(self.assembly.size * 0.8)
            pattern[inh_start:inh_end] = inhibition * -1  # Negativo para indicar inibição
            
            ax2.bar(range(self.assembly.size), pattern)
            ax2.set_title('Input Pattern')
            ax2.set_xlabel('Neuron ID')
            ax2.set_ylabel('Input Level')
            ax2.set_ylim(-1, 1)
            
        elif vis_mode == "Frequency Domain":
            # Análise de frequência
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            
            # Espectro de potência
            ax1 = self.main_figure.add_subplot(gs[0])
            
            if state["frequency_history"]:
                # Plotamos o espectro de frequência
                freqs, magnitudes = zip(*state["frequency_history"])
                ax1.plot(freqs, magnitudes)
                
                # Destaca a faixa de frequência do modo atual
                freq_ranges = {
                    "Gamma": (30, 80),
                    "Beta": (13, 30),
                    "Alpha": (8, 12),
                    "Theta": (4, 8)
                }
                
                if state["oscillation_mode"] in freq_ranges:
                    low, high = freq_ranges[state["oscillation_mode"]]
                    ax1.axvspan(low, high, alpha=0.2, color='green')
                    ax1.text((low+high)/2, 0.9*max(magnitudes), 
                            f"{state['oscillation_mode']} Band", 
                            ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7))
                
                ax1.set_title('Frequency Spectrum')
                ax1.set_xlabel('Frequency (Hz)')
                ax1.set_ylabel('Magnitude')
                ax1.set_xlim(0, min(100, max(freqs)))
                ax1.grid(True, alpha=0.3)
            
            # Espectrograma (evolução do espectro ao longo do tempo)
            ax2 = self.main_figure.add_subplot(gs[1])
            
            if len(state["activity_history"]) > 128:
                # Calcula o espectrograma
                activity_array = np.array(state["activity_history"])
                f, t, Sxx = signal.spectrogram(activity_array, fs=10, nperseg=64, noverlap=32)
                
                # Visualiza o espectrograma
                im = ax2.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
                self.main_figure.colorbar(im, ax=ax2)
                ax2.set_title('Spectrogram')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Frequency (Hz)')
                ax2.set_ylim(0, 10)  # Limita a visualização às frequências mais relevantes
            
        elif vis_mode == "Phase Space":
            # Visualização do espaço de fase
            gs = gridspec.GridSpec(2, 2)
            
            # Trajetória no espaço de fase
            ax1 = self.main_figure.add_subplot(gs[0, 0])
            
            if len(state["activity_history"]) > 2:
                # Cria uma visualização de espaço de fase usando atividade e sua derivada
                activity = np.array(state["activity_history"])
                activity_diff = np.diff(activity, prepend=activity[0])
                
                ax1.plot(activity[:-10], activity_diff[:-10], 'k-', alpha=0.3)  # trajetória histórica
                ax1.plot(activity[-10:], activity_diff[-10:], 'r-', linewidth=2)  # trajetória recente
                
                # Marca o estado atual
                ax1.plot(activity[-1], activity_diff[-1], 'ro', markersize=8)
                
                ax1.set_title('Phase Space')
                ax1.set_xlabel('Activity')
                ax1.set_ylabel('Activity Derivative')
                ax1.grid(True, alpha=0.3)
            
            # Visualização 3D do espaço de fase
            ax2 = self.main_figure.add_subplot(gs[0, 1], projection='3d')
            
            if len(state["activity_history"]) > 4:
                # Usa embedding de atraso para criar visualização 3D
                activity = np.array(state["activity_history"])
                x = activity[:-2]
                y = activity[1:-1]
                z = activity[2:]
                
                # Limita o número de pontos para clareza
                max_points = 500
                if len(x) > max_points:
                    step = len(x) // max_points
                    step = len(x) // max_points
                    x = x[::step]
                    y = y[::step]
                    z = z[::step]
                
                ax2.plot(x, y, z, 'b-', alpha=0.5)
                
                # Marca os últimos pontos
                if len(x) > 20:
                    ax2.plot(x[-20:], y[-20:], z[-20:], 'r-', linewidth=2)
                
                ax2.set_title('3D Phase Space')
                ax2.set_xlabel('Activity(t)')
                ax2.set_ylabel('Activity(t+1)')
                ax2.set_zlabel('Activity(t+2)')
                
                # Define limites para manter a visualização estável
                axis_max = max(np.max(activity), 10)
                ax2.set_xlim(0, axis_max)
                ax2.set_ylim(0, axis_max)
                ax2.set_zlim(0, axis_max)
            
            # Mapa de recorrência
            ax3 = self.main_figure.add_subplot(gs[1, 0])
            
            if len(state["activity_history"]) > 50:
                # Cria um mapa de recorrência
                activity = np.array(state["activity_history"][-200:])
                n = len(activity)
                recurrence = np.zeros((n, n))
                
                # Calcula a matriz de recorrência
                threshold = 5.0  # Ajuste conforme necessário
                for i in range(n):
                    for j in range(n):
                        recurrence[i, j] = 1 if abs(activity[i] - activity[j]) < threshold else 0
                
                ax3.imshow(recurrence, cmap='binary', aspect='auto')
                ax3.set_title('Recurrence Plot')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Time')
            
            # Autocorrelação
            ax4 = self.main_figure.add_subplot(gs[1, 1])
            
            if len(state["activity_history"]) > 50:
                # Calcula a autocorrelação
                activity = np.array(state["activity_history"][-100:])
                activity_norm = activity - np.mean(activity)
                autocorr = np.correlate(activity_norm, activity_norm, mode='full')
                autocorr = autocorr[len(activity_norm)-1:]
                autocorr = autocorr / autocorr[0]  # Normaliza
                
                ax4.plot(autocorr)
                ax4.set_title('Autocorrelation')
                ax4.set_xlabel('Lag')
                ax4.set_ylabel('Correlation')
                ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax4.grid(True, alpha=0.3)
        
        self.main_figure.tight_layout()
        self.main_canvas.draw()
    
    def setup_neuromodulation(self):
        """Configuração da demonstração de efeitos de neuromodulação"""
        # Descrição
        self.description_label.setText(
            "Neuromodulation Effects\n\n"
            "This demonstration visualizes how neuromodulators (dopamine, serotonin, acetylcholine, "
            "and norepinephrine) influence neural activity and learning. You can observe how different "
            "neuromodulatory states affect the network's response to stimuli, plasticity, and "
            "information processing, similar to how mood and alertness affect brain function."
        )
        
        # Estado da demonstração
        if "Neuromodulation" not in self.demo_states:
            self.demo_states["Neuromodulation"] = {
                "current_modulator": "dopamine",
                "modulator_history": [],
                "response_history": [],
                "learning_rate_history": []
            }
        
        state = self.demo_states["Neuromodulation"]
        
        # Adiciona controles
        neuromod_group = QGroupBox("Neuromodulator Controls")
        neuromod_layout = QVBoxLayout()
        
        # Controles para cada neuromodulador
        for modulator in ["dopamine", "serotonin", "acetylcholine", "norepinephrine"]:
            mod_layout = QHBoxLayout()
            mod_layout.addWidget(QLabel(f"{modulator.capitalize()}:"))
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 200)
            slider.setValue(100)  # Valor normal = 1.0
            slider.valueChanged.connect(lambda val, m=modulator: self.set_neuromodulator(m, val/100))
            mod_layout.addWidget(slider)
            
            # Armazena o slider para uso posterior
            state[f"{modulator}_slider"] = slider
            
            value_label = QLabel("1.00")
            mod_layout.addWidget(value_label)
            state[f"{modulator}_label"] = value_label
            
            neuromod_layout.addLayout(mod_layout)
        
        # Botões para cenários predefinidos
        scenario_layout = QHBoxLayout()
        scenario_layout.addWidget(QLabel("Preset Scenarios:"))
        
        reward_button = QPushButton("Reward")
        reward_button.clicked.connect(self.set_reward_scenario)
        scenario_layout.addWidget(reward_button)
        
        stress_button = QPushButton("Stress")
        stress_button.clicked.connect(self.set_stress_scenario)
        scenario_layout.addWidget(stress_button)
        
        focus_button = QPushButton("Focus")
        focus_button.clicked.connect(self.set_focus_scenario)
        scenario_layout.addWidget(focus_button)
        
        sleep_button = QPushButton("Sleep")
        sleep_button.clicked.connect(self.set_sleep_scenario)
        scenario_layout.addWidget(sleep_button)
        
        neuromod_layout.addLayout(scenario_layout)
        
        neuromod_group.setLayout(neuromod_layout)
        self.demo_controls_layout.addWidget(neuromod_group)
        
        # Controles de estímulo
        stimulus_group = QGroupBox("Stimulus Control")
        stimulus_layout = QVBoxLayout()
        
        # Tipo de estímulo
        stimulus_layout.addWidget(QLabel("Stimulus Type:"))
        
        self.stim_radio_reward = QRadioButton("Reward Cue")
        self.stim_radio_reward.setChecked(True)
        self.stim_radio_reward.toggled.connect(lambda: self.set_stimulus_type("reward"))
        stimulus_layout.addWidget(self.stim_radio_reward)
        
        self.stim_radio_aversion = QRadioButton("Aversive Cue")
        self.stim_radio_aversion.toggled.connect(lambda: self.set_stimulus_type("aversive"))
        stimulus_layout.addWidget(self.stim_radio_aversion)
        
        self.stim_radio_neutral = QRadioButton("Neutral Cue")
        self.stim_radio_neutral.toggled.connect(lambda: self.set_stimulus_type("neutral"))
        stimulus_layout.addWidget(self.stim_radio_neutral)
        
        # Armazena o tipo de estímulo
        state["stimulus_type"] = "reward"
        
        stimulus_group.setLayout(stimulus_layout)
        self.demo_controls_layout.addWidget(stimulus_group)
    
    def set_neuromodulator(self, modulator, value):
        """Define o nível de um neuromodulador"""
        if modulator == "dopamine":
            self.assembly.neuromodulators.dopamine = value
        elif modulator == "serotonin":
            self.assembly.neuromodulators.serotonin = value
        elif modulator == "acetylcholine":
            self.assembly.neuromodulators.acetylcholine = value
        elif modulator == "norepinephrine":
            self.assembly.neuromodulators.norepinephrine = value
        
        # Atualiza o label
        state = self.demo_states["Neuromodulation"]
        state[f"{modulator}_label"].setText(f"{value:.2f}")
    
    def set_stimulus_type(self, stim_type):
        """Define o tipo de estímulo"""
        state = self.demo_states["Neuromodulation"]
        state["stimulus_type"] = stim_type
    
    def set_reward_scenario(self):
        """Configura um cenário de recompensa"""
        self.set_neuromodulator("dopamine", 1.8)  # Dopamina alta
        self.set_neuromodulator("serotonin", 1.2)  # Serotonina levemente alta
        self.set_neuromodulator("acetylcholine", 1.0)  # Acetilcolina normal
        self.set_neuromodulator("norepinephrine", 1.2)  # Norepinefrina levemente alta
        
        # Atualiza os sliders
        state = self.demo_states["Neuromodulation"]
        state["dopamine_slider"].setValue(180)
        state["serotonin_slider"].setValue(120)
        state["acetylcholine_slider"].setValue(100)
        state["norepinephrine_slider"].setValue(120)
    
    def set_stress_scenario(self):
        """Configura um cenário de estresse"""
        self.set_neuromodulator("dopamine", 0.7)  # Dopamina baixa
        self.set_neuromodulator("serotonin", 0.5)  # Serotonina baixa
        self.set_neuromodulator("acetylcholine", 1.0)  # Acetilcolina normal
        self.set_neuromodulator("norepinephrine", 1.8)  # Norepinefrina alta
        
        # Atualiza os sliders
        state = self.demo_states["Neuromodulation"]
        state["dopamine_slider"].setValue(70)
        state["serotonin_slider"].setValue(50)
        state["acetylcholine_slider"].setValue(100)
        state["norepinephrine_slider"].setValue(180)
    
    def set_focus_scenario(self):
        """Configura um cenário de foco/atenção"""
        self.set_neuromodulator("dopamine", 1.2)  # Dopamina levemente alta
        self.set_neuromodulator("serotonin", 1.0)  # Serotonina normal
        self.set_neuromodulator("acetylcholine", 1.6)  # Acetilcolina alta
        self.set_neuromodulator("norepinephrine", 1.4)  # Norepinefrina moderadamente alta
        
        # Atualiza os sliders
        state = self.demo_states["Neuromodulation"]
        state["dopamine_slider"].setValue(120)
        state["serotonin_slider"].setValue(100)
        state["acetylcholine_slider"].setValue(160)
        state["norepinephrine_slider"].setValue(140)
    
    def set_sleep_scenario(self):
        """Configura um cenário de sono"""
        self.set_neuromodulator("dopamine", 0.5)  # Dopamina baixa
        self.set_neuromodulator("serotonin", 1.5)  # Serotonina alta
        self.set_neuromodulator("acetylcholine", 0.3)  # Acetilcolina muito baixa
        self.set_neuromodulator("norepinephrine", 0.2)  # Norepinefrina muito baixa
        
        # Atualiza os sliders
        state = self.demo_states["Neuromodulation"]
        state["dopamine_slider"].setValue(50)
        state["serotonin_slider"].setValue(150)
        state["acetylcholine_slider"].setValue(30)
        state["norepinephrine_slider"].setValue(20)
    
    def update_neuromodulation(self):
        """Atualiza a demonstração de efeitos de neuromodulação"""
        state = self.demo_states["Neuromodulation"]
        
        # Prepara o estímulo com base no tipo selecionado
        stimulus = np.zeros(self.assembly.size)
        
        if state["stimulus_type"] == "reward":
            # Estímulo associado a recompensa (região anterior)
            start = int(self.assembly.size * 0.2)
            end = int(self.assembly.size * 0.4)
            stimulus[start:end] = 1.0
        elif state["stimulus_type"] == "aversive":
            # Estímulo aversivo (região posterior)
            start = int(self.assembly.size * 0.6)
            end = int(self.assembly.size * 0.8)
            stimulus[start:end] = 1.0
        else:  # neutral
            # Estímulo neutro (região central)
            start = int(self.assembly.size * 0.4)
            end = int(self.assembly.size * 0.6)
            stimulus[start:end] = 1.0
        
        # Adiciona ruído
        stimulus += np.random.normal(0, 0.1, self.assembly.size)
        stimulus = np.clip(stimulus, 0, 1)
        
        # Atualiza a rede neural
        activation = self.assembly.update(stimulus, self.timestep)
        
        # Registra o histórico
        modulator_levels = {
            "dopamine": self.assembly.neuromodulators.dopamine,
            "serotonin": self.assembly.neuromodulators.serotonin,
            "acetylcholine": self.assembly.neuromodulators.acetylcholine,
            "norepinephrine": self.assembly.neuromodulators.norepinephrine
        }
        
        state["modulator_history"].append(modulator_levels)
        
        # Registra a resposta (atividade neural)
        response = len(self.assembly.activation_history[-1]) if self.assembly.activation_history else 0
        state["response_history"].append(response)
        
        # Registra a taxa de aprendizado efetiva (apenas para EnhancedLearningAssembly)
        if isinstance(self.assembly, EnhancedLearningAssembly):
            effective_rate = self.assembly.learning_parameters.learning_rate * self.assembly.neuromodulators.dopamine
            state["learning_rate_history"].append(effective_rate)
        else:
            state["learning_rate_history"].append(0.01)  # Valor padrão
        
        # Limita o tamanho do histórico
        max_history = 200
        if len(state["modulator_history"]) > max_history:
            state["modulator_history"] = state["modulator_history"][-max_history:]
            state["response_history"] = state["response_history"][-max_history:]
            state["learning_rate_history"] = state["learning_rate_history"][-max_history:]
    
    def visualize_neuromodulation(self):
        """Visualiza a demonstração de efeitos de neuromodulação"""
        state = self.demo_states["Neuromodulation"]
        
        self.main_figure.clear()
        gs = gridspec.GridSpec(3, 2)
        
        # Histórico de níveis de neuromoduladores
        ax1 = self.main_figure.add_subplot(gs[0, :])
        
        if state["modulator_history"]:
            time_steps = range(len(state["modulator_history"]))
            
            # Plota cada neuromodulador
            for modulator in ["dopamine", "serotonin", "acetylcholine", "norepinephrine"]:
                values = [entry[modulator] for entry in state["modulator_history"]]
                ax1.plot(time_steps, values, label=modulator.capitalize())
            
            ax1.set_title('Neuromodulator Levels Over Time')
            ax1.set_ylabel('Level')
            ax1.legend(loc='upper left')
            ax1.set_ylim(0, 2.0)
            ax1.grid(True, alpha=0.3)
            
            # Limita a visualização aos últimos 200 passos
            if len(time_steps) > 200:
                ax1.set_xlim(time_steps[-200], time_steps[-1])
        
        # Resposta neural
        ax2 = self.main_figure.add_subplot(gs[1, 0])
        
        if state["response_history"]:
            time_steps = range(len(state["response_history"]))
            ax2.plot(time_steps, state["response_history"])
            
            # Adiciona linha de média móvel
            if len(state["response_history"]) > 10:
                window_size = min(20, len(state["response_history"]))
                response_smooth = np.convolve(state["response_history"], 
                                             np.ones(window_size)/window_size, 
                                             mode='valid')
                valid_steps = range(window_size-1, len(state["response_history"]))
                ax2.plot(valid_steps, response_smooth, 'r-', linewidth=2)
            
            ax2.set_title('Neural Response')
            ax2.set_ylabel('Active Neurons')
            ax2.grid(True, alpha=0.3)
            
            # Limita a visualização aos últimos 200 passos
            if len(time_steps) > 200:
                ax2.set_xlim(time_steps[-200], time_steps[-1])
        
        # Taxa de aprendizado efetiva
        ax3 = self.main_figure.add_subplot(gs[1, 1])
        
        if state["learning_rate_history"]:
            time_steps = range(len(state["learning_rate_history"]))
            ax3.plot(time_steps, state["learning_rate_history"])
            
            ax3.set_title('Effective Learning Rate')
            ax3.set_ylabel('Rate')
            ax3.grid(True, alpha=0.3)
            
            # Limita a visualização aos últimos 200 passos
            if len(time_steps) > 200:
                ax3.set_xlim(time_steps[-200], time_steps[-1])
        
        # Efeitos neuromoduladores na rede
        ax4 = self.main_figure.add_subplot(gs[2, 0])
        
        # Visualiza o efeito atual dos neuromoduladores
        mod_effects = {
            "Excitability": self.assembly.neuromodulators.dopamine,
            "Stability": self.assembly.neuromodulators.serotonin,
            "Precision": self.assembly.neuromodulators.acetylcholine,
            "Responsivity": self.assembly.neuromodulators.norepinephrine
        }
        
        effects = list(mod_effects.keys())
        values = [mod_effects[effect] for effect in effects]
        
        ax4.bar(effects, values)
        ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)  # Linha de base
        ax4.set_title('Network Properties')
        ax4.set_ylabel('Relative Level')
        ax4.set_ylim(0, 2.0)
        
        # Correlação entre moduladores e resposta
        ax5 = self.main_figure.add_subplot(gs[2, 1])
        
        if len(state["modulator_history"]) > 10 and len(state["response_history"]) > 10:
            # Calcula correlações
            dopamine = [entry["dopamine"] for entry in state["modulator_history"]]
            serotonin = [entry["serotonin"] for entry in state["modulator_history"]]
            acetylcholine = [entry["acetylcholine"] for entry in state["modulator_history"]]
            norepinephrine = [entry["norepinephrine"] for entry in state["modulator_history"]]
            
            response = state["response_history"]
            
            # Limita ao mesmo tamanho
            min_length = min(len(dopamine), len(response))
            
            # Calcula correlações
            corr_da = stats.pearsonr(dopamine[-min_length:], response[-min_length:])[0]
            corr_5ht = stats.pearsonr(serotonin[-min_length:], response[-min_length:])[0]
            corr_ach = stats.pearsonr(acetylcholine[-min_length:], response[-min_length:])[0]
            corr_ne = stats.pearsonr(norepinephrine[-min_length:], response[-min_length:])[0]
            
            correlations = [corr_da, corr_5ht, corr_ach, corr_ne]
            labels = ['DA', '5-HT', 'ACh', 'NE']
            
            # Plotamos as correlações
            bars = ax5.bar(labels, correlations)
            
            # Adiciona cores com base no sinal da correlação
            for i, bar in enumerate(bars):
                bar.set_color('green' if correlations[i] > 0 else 'red')
            
            ax5.set_title('Correlation with Neural Activity')
            ax5.set_ylabel('Correlation')
            ax5.set_ylim(-1, 1)
            ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        self.main_figure.tight_layout()
        self.main_canvas.draw()
    
    # === Demonstrações adicionais podem ser implementadas aqui ===
    
    def setup_memory_formation(self):
        """Configuração da demonstração de formação de memória"""
        # Descrição
        self.description_label.setText(
            "Memory Formation\n\n"
            "This demonstration shows how neural networks form and consolidate memories. "
            "You can observe the processes of encoding, consolidation, and recall - similar to "
            "how your brain forms short-term and long-term memories through synaptic changes "
            "and structural reorganization."
        )
        
        # Implementação básica
        self.statusBar().showMessage("Memory Formation demonstration initialized")
    
    def setup_sensory_integration(self):
        """Configuração da demonstração de integração sensorial"""
        # Descrição
        self.description_label.setText(
            "Sensory Integration\n\n"
            "This demonstration visualizes how different sensory inputs are integrated "
            "in the brain. You can observe how multimodal information is combined to form "
            "coherent percepts, similar to how your brain merges visual, auditory, and "
            "tactile information."
        )
        
        # Implementação básica
        self.statusBar().showMessage("Sensory Integration demonstration initialized")
    
    def setup_attractor_dynamics(self):
        """Configuração da demonstração de dinâmicas de atratores"""
        # Descrição
        self.description_label.setText(
            "Attractor Dynamics\n\n"
            "This demonstration shows how neural networks can form stable attractor states. "
            "You can observe how network activity converges to specific patterns regardless "
            "of initial conditions, similar to how memories and behavioral patterns become "
            "stable in the brain."
        )
        
        # Implementação básica
        self.statusBar().showMessage("Attractor Dynamics demonstration initialized")
    
    def setup_decision_making(self):
        """Configuração da demonstração de tomada de decisão"""
        # Descrição
        self.description_label.setText(
            "Decision Making\n\n"
            "This demonstration visualizes neural mechanisms of decision making. "
            "You can observe evidence accumulation, threshold crossing, and commitment "
            "to choices, similar to how your brain weighs options and makes decisions."
        )
        
        # Implementação básica
        self.statusBar().showMessage("Decision Making demonstration initialized")


# Função principal para executar a aplicação
def run_neural_demos():
    """Executa a aplicação de demonstrações neurais"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = BiologicalNeuralDemos()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    run_neural_demos()