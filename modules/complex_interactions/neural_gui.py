import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QSlider, QLabel, 
                            QComboBox, QCheckBox, QGroupBox, QSpinBox, 
                            QDoubleSpinBox, QTabWidget, QRadioButton,
                            QButtonGroup, QFileDialog, QMessageBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# Importa as classes das outras partes do sistema
from complex_neural import ComplexNeuralAssembly, ComplexInteractionVisualizer, InteractionType
from enhanced_learning import EnhancedLearningAssembly, LearningRule

class EnhancedVisualizationTab(QWidget):
    """Tab para visualizações avançadas baseadas em regras de aprendizado específicas"""
    def __init__(self, assembly):
        super().__init__()
        self.assembly = assembly
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Área de gráficos
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
        
        # Controles de visualização
        controls_layout = QHBoxLayout()
        
        # Dropdown para selecionar visualização
        viz_layout = QHBoxLayout()
        viz_layout.addWidget(QLabel("Visualization:"))
        self.viz_combo = QComboBox()
        self.viz_combo.addItems([
            "Eligibility Traces", 
            "Learning Rules Activity",
            "Reward History",
            "Neural Assemblies",
            "Metaplasticity Thresholds",
            "BCM Activity Averages"
        ])
        self.viz_combo.currentIndexChanged.connect(self.update_visualization)
        viz_layout.addWidget(self.viz_combo)
        controls_layout.addLayout(viz_layout)
        
        main_layout.addLayout(controls_layout)
    
    def update_visualization(self):
        """Atualiza a visualização com base na seleção do usuário"""
        self.figure.clear()
        
        viz_type = self.viz_combo.currentText()
        
        if viz_type == "Eligibility Traces":
            # Visualiza traços de elegibilidade
            ax = self.figure.add_subplot(111)
            if hasattr(self.assembly, 'eligibility_traces'):
                eligibility_matrix = np.zeros((self.assembly.size, self.assembly.size))
                for (i, j), trace in self.assembly.eligibility_traces.items():
                    eligibility_matrix[i, j] = trace
                im = ax.imshow(eligibility_matrix, cmap='plasma')
                self.figure.colorbar(im, ax=ax)
                ax.set_title('Eligibility Traces')
            else:
                ax.text(0.5, 0.5, "Eligibility traces not available\nUse Enhanced Assembly with Reinforcement Learning", 
                       ha='center', va='center')
                
        elif viz_type == "Learning Rules Activity":
            # Visualiza quais regras de aprendizado estão ativas
            ax = self.figure.add_subplot(111)
            if hasattr(self.assembly, 'active_learning_rules'):
                rules = list(self.assembly.active_learning_rules.keys())
                active = [int(self.assembly.active_learning_rules[rule]) for rule in rules]
                rule_names = [rule.value for rule in rules]
                
                ax.bar(rule_names, active)
                ax.set_title('Active Learning Rules')
                ax.set_ylim(0, 1.2)
                ax.set_ylabel('Active (1) / Inactive (0)')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, "Learning rules not available\nUse Enhanced Assembly", 
                       ha='center', va='center')
                
        elif viz_type == "Reward History":
            # Visualiza histórico de recompensa
            ax = self.figure.add_subplot(111)
            if hasattr(self.assembly, 'reward_history') and self.assembly.reward_history:
                ax.plot(self.assembly.reward_history)
                ax.set_title('Reward History')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Reward')
            else:
                ax.text(0.5, 0.5, "Reward history not available\nUse Enhanced Assembly with Reinforcement Learning", 
                       ha='center', va='center')
                
        elif viz_type == "Neural Assemblies":
            # Visualiza assembleias neurais detectadas
            ax = self.figure.add_subplot(111)
            if hasattr(self.assembly, 'neural_assemblies') and self.assembly.neural_assemblies:
                assembly_matrix = np.zeros((self.assembly.size, self.assembly.size))
                for i, assembly_neurons in enumerate(self.assembly.neural_assemblies):
                    for neuron in assembly_neurons:
                        assembly_matrix[neuron, :] = i + 1
                im = ax.imshow(assembly_matrix, cmap='tab10')
                self.figure.colorbar(im, ax=ax)
                ax.set_title('Detected Neural Assemblies')
            else:
                ax.text(0.5, 0.5, "Neural assemblies not available\nUse Enhanced Assembly", 
                       ha='center', va='center')
                
        elif viz_type == "Metaplasticity Thresholds":
            # Visualiza limiares de metaplasticidade
            ax = self.figure.add_subplot(111)
            if hasattr(self.assembly, 'plasticity_thresholds'):
                ax.plot(self.assembly.plasticity_thresholds)
                ax.set_title('Metaplasticity Thresholds')
                ax.set_xlabel('Neuron Index')
                ax.set_ylabel('Threshold')
            else:
                ax.text(0.5, 0.5, "Metaplasticity thresholds not available\nUse Enhanced Assembly", 
                       ha='center', va='center')
                
        elif viz_type == "BCM Activity Averages":
            # Visualiza médias de atividade BCM
            ax = self.figure.add_subplot(111)
            if hasattr(self.assembly, 'activity_averages'):
                ax.plot(self.assembly.activity_averages)
                ax.set_title('BCM Activity Averages')
                ax.set_xlabel('Neuron Index')
                ax.set_ylabel('Activity Average')
            else:
                ax.text(0.5, 0.5, "BCM activity averages not available\nUse Enhanced Assembly with BCM learning", 
                       ha='center', va='center')
                
        self.canvas.draw()


class LearningRulesTab(QWidget):
    """Tab para configurar regras de aprendizado"""
    def __init__(self, assembly):
        super().__init__()
        self.assembly = assembly
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Só mostra configurações avançadas se for uma EnhancedLearningAssembly
        if not isinstance(self.assembly, EnhancedLearningAssembly):
            main_layout.addWidget(QLabel("Advanced learning rules are only available with Enhanced Neural Assembly.\n"
                                         "Please restart the application and select 'Enhanced Neural Assembly' in the settings."))
            return
        
        # Título
        main_layout.addWidget(QLabel("<h2>Learning Rules Configuration</h2>"))
        
        # Checkboxes para ativar/desativar regras
        rules_group = QGroupBox("Active Learning Rules")
        rules_layout = QVBoxLayout()
        
        self.rule_checkboxes = {}
        for rule in LearningRule:
            checkbox = QCheckBox(f"{rule.value.capitalize()}")
            checkbox.setChecked(self.assembly.active_learning_rules[rule])
            checkbox.stateChanged.connect(lambda state, r=rule: self.toggle_rule(r, state))
            self.rule_checkboxes[rule] = checkbox
            rules_layout.addWidget(checkbox)
        
        rules_group.setLayout(rules_layout)
        main_layout.addWidget(rules_group)
        
        # Parâmetros de aprendizado
        params_group = QGroupBox("Learning Parameters")
        params_layout = QGridLayout()
        
        # Adiciona sliders para diferentes parâmet
        # Parâmetros de aprendizado
        params_group = QGroupBox("Learning Parameters")
        params_layout = QGridLayout()
        
        # Adiciona sliders para diferentes parâmetros
        row = 0
        
        # Taxa de aprendizado
        params_layout.addWidget(QLabel("Learning Rate:"), row, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.001, 0.5)
        self.learning_rate_spin.setSingleStep(0.001)
        self.learning_rate_spin.setValue(self.assembly.learning_parameters.learning_rate)
        self.learning_rate_spin.valueChanged.connect(self.update_learning_rate)
        params_layout.addWidget(self.learning_rate_spin, row, 1)
        row += 1
        
        # Parâmetros BCM
        params_layout.addWidget(QLabel("BCM Threshold:"), row, 0)
        self.bcm_threshold_spin = QDoubleSpinBox()
        self.bcm_threshold_spin.setRange(0.1, 1.0)
        self.bcm_threshold_spin.setSingleStep(0.05)
        self.bcm_threshold_spin.setValue(self.assembly.learning_parameters.bcm_threshold)
        self.bcm_threshold_spin.valueChanged.connect(self.update_bcm_threshold)
        params_layout.addWidget(self.bcm_threshold_spin, row, 1)
        row += 1
        
        params_layout.addWidget(QLabel("BCM Time Constant:"), row, 0)
        self.bcm_time_spin = QDoubleSpinBox()
        self.bcm_time_spin.setRange(10, 500)
        self.bcm_time_spin.setSingleStep(10)
        self.bcm_time_spin.setValue(self.assembly.learning_parameters.bcm_time_constant)
        self.bcm_time_spin.valueChanged.connect(self.update_bcm_time_constant)
        params_layout.addWidget(self.bcm_time_spin, row, 1)
        row += 1
        
        # Parâmetros de reforço
        params_layout.addWidget(QLabel("Eligibility Decay:"), row, 0)
        self.eligibility_decay_spin = QDoubleSpinBox()
        self.eligibility_decay_spin.setRange(0.5, 0.99)
        self.eligibility_decay_spin.setSingleStep(0.01)
        self.eligibility_decay_spin.setValue(self.assembly.learning_parameters.eligibility_decay)
        self.eligibility_decay_spin.valueChanged.connect(self.update_eligibility_decay)
        params_layout.addWidget(self.eligibility_decay_spin, row, 1)
        row += 1
        
        params_layout.addWidget(QLabel("Reward Discount:"), row, 0)
        self.reward_discount_spin = QDoubleSpinBox()
        self.reward_discount_spin.setRange(0.5, 0.99)
        self.reward_discount_spin.setSingleStep(0.01)
        self.reward_discount_spin.setValue(self.assembly.learning_parameters.reward_discount)
        self.reward_discount_spin.valueChanged.connect(self.update_reward_discount)
        params_layout.addWidget(self.reward_discount_spin, row, 1)
        row += 1
        
        # Parâmetros de aprendizado competitivo
        params_layout.addWidget(QLabel("Winner Strength:"), row, 0)
        self.winner_strength_spin = QDoubleSpinBox()
        self.winner_strength_spin.setRange(0.5, 2.0)
        self.winner_strength_spin.setSingleStep(0.1)
        self.winner_strength_spin.setValue(self.assembly.learning_parameters.winner_strength)
        self.winner_strength_spin.valueChanged.connect(self.update_winner_strength)
        params_layout.addWidget(self.winner_strength_spin, row, 1)
        row += 1
        
        params_layout.addWidget(QLabel("Inhibition Strength:"), row, 0)
        self.inhibition_strength_spin = QDoubleSpinBox()
        self.inhibition_strength_spin.setRange(0.1, 1.5)
        self.inhibition_strength_spin.setSingleStep(0.1)
        self.inhibition_strength_spin.setValue(self.assembly.learning_parameters.inhibition_strength)
        self.inhibition_strength_spin.valueChanged.connect(self.update_inhibition_strength)
        params_layout.addWidget(self.inhibition_strength_spin, row, 1)
        row += 1
        
        # Parâmetros de metaplasticidade
        params_layout.addWidget(QLabel("Metaplasticity Rate:"), row, 0)
        self.metaplasticity_rate_spin = QDoubleSpinBox()
        self.metaplasticity_rate_spin.setRange(0.0001, 0.01)
        self.metaplasticity_rate_spin.setSingleStep(0.0001)
        self.metaplasticity_rate_spin.setDecimals(4)
        self.metaplasticity_rate_spin.setValue(self.assembly.learning_parameters.metaplasticity_rate)
        self.metaplasticity_rate_spin.valueChanged.connect(self.update_metaplasticity_rate)
        params_layout.addWidget(self.metaplasticity_rate_spin, row, 1)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
    
    def toggle_rule(self, rule, state):
        """Ativa ou desativa uma regra de aprendizado"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.set_learning_rule(rule, state == Qt.Checked)
    
    def update_learning_rate(self, value):
        """Atualiza a taxa de aprendizado"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.learning_parameters.learning_rate = value
    
    def update_bcm_threshold(self, value):
        """Atualiza o limiar BCM"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.learning_parameters.bcm_threshold = value
    
    def update_bcm_time_constant(self, value):
        """Atualiza a constante de tempo BCM"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.learning_parameters.bcm_time_constant = value
    
    def update_eligibility_decay(self, value):
        """Atualiza o decaimento de elegibilidade"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.learning_parameters.eligibility_decay = value
    
    def update_reward_discount(self, value):
        """Atualiza o desconto de recompensa"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.learning_parameters.reward_discount = value
    
    def update_winner_strength(self, value):
        """Atualiza a força do vencedor"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.learning_parameters.winner_strength = value
    
    def update_inhibition_strength(self, value):
        """Atualiza a força de inibição"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.learning_parameters.inhibition_strength = value
    
    def update_metaplasticity_rate(self, value):
        """Atualiza a taxa de metaplasticidade"""
        if isinstance(self.assembly, EnhancedLearningAssembly):
            self.assembly.learning_parameters.metaplasticity_rate = value


class NeuralSimulationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configuração da janela
        self.setWindowTitle("Neural Network Simulation Interface")
        self.setGeometry(100, 100, 1200, 800)
        
        # Inicializa variáveis
        self.timestep = 0
        self.assembly = None
        self.visualizer = None
        self.assembly_type = "standard"  # Padrão ou avançado
        self.assembly_size = 100
        
        # Configuração inicial
        self.show_setup_dialog()
        
        # Parâmetros da simulação
        self.running = False
        self.update_interval = 100  # ms
        self.simulation_speed = 1  # passos por atualização
        self.current_reward = 0.0
        
        # Padrões de entrada predefinidos
        self.input_patterns = {
            "Pattern A": np.zeros(self.assembly_size),
            "Pattern B": np.zeros(self.assembly_size),
            "Random": np.random.random(self.assembly_size)
        }
        self.input_patterns["Pattern A"][int(self.assembly_size*0.2):int(self.assembly_size*0.4)] = 1
        self.input_patterns["Pattern B"][int(self.assembly_size*0.6):int(self.assembly_size*0.8)] = 1
        self.current_pattern = "Pattern A"
        
        # Timer para simulação
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        
        # Configuração da interface
        self.setup_ui()
        
    def show_setup_dialog(self):
        """Mostra diálogo para configurar a simulação"""
        # Mudamos de QWidget para QDialog
        from PyQt5.QtWidgets import QDialog
        
        dialog = QDialog(self)  # Agora é um QDialog com parent 'self'
        dialog.setWindowTitle("Simulation Setup")
        dialog.setGeometry(300, 300, 400, 200)
        
        layout = QVBoxLayout()
        
        # Seleção de tipo de assembleia
        type_group = QGroupBox("Neural Assembly Type")
        type_layout = QVBoxLayout()
        
        self.type_buttons = QButtonGroup()
        standard_button = QRadioButton("Standard Neural Assembly")
        standard_button.setChecked(True)
        enhanced_button = QRadioButton("Enhanced Neural Assembly (with advanced learning)")
        
        self.type_buttons.addButton(standard_button, 1)
        self.type_buttons.addButton(enhanced_button, 2)
        
        type_layout.addWidget(standard_button)
        type_layout.addWidget(enhanced_button)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Tamanho da rede
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Network Size:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(10, 500)
        self.size_spin.setValue(100)
        self.size_spin.setSingleStep(10)
        size_layout.addWidget(self.size_spin)
        layout.addLayout(size_layout)
        
        # Botões OK/Cancel
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(lambda: self.apply_setup(dialog))
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(lambda: dialog.reject())  # Use reject() para fechar o diálogo
        
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)
        
        dialog.setLayout(layout)
        dialog.exec_()  # Agora podemos usar exec_() no QDialog
    
    def apply_setup(self, dialog):
        """Aplica configurações do diálogo"""
        # Obtém tipo de assembleia
        if self.type_buttons.checkedId() == 1:
            self.assembly_type = "standard"
        else:
            self.assembly_type = "enhanced"
        
        # Obtém tamanho da rede
        self.assembly_size = self.size_spin.value()
        
        # Cria assembleia apropriada
        if self.assembly_type == "standard":
            self.assembly = ComplexNeuralAssembly(self.assembly_size)
        else:
            self.assembly = EnhancedLearningAssembly(self.assembly_size)
            
        # Cria visualizador
        self.visualizer = ComplexInteractionVisualizer(self.assembly)
        
        # Reconfigura padrões de entrada com novo tamanho
        self.input_patterns = {
            "Pattern A": np.zeros(self.assembly_size),
            "Pattern B": np.zeros(self.assembly_size),
            "Random": np.random.random(self.assembly_size)
        }
        self.input_patterns["Pattern A"][int(self.assembly_size*0.2):int(self.assembly_size*0.4)] = 1
        self.input_patterns["Pattern B"][int(self.assembly_size*0.6):int(self.assembly_size*0.8)] = 1
        
        dialog.accept()  # Use accept() em vez de close()
        
    def setup_ui(self):
        # Widget principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Layout superior - controles
        controls_layout = QHBoxLayout()
        
        # Grupo de controles de simulação
        sim_group = QGroupBox("Simulation Controls")
        sim_layout = QVBoxLayout()
        
        # Botões de controle
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_simulation)
        buttons_layout.addWidget(self.start_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        buttons_layout.addWidget(self.reset_button)
        
        self.save_button = QPushButton("Save State")
        self.save_button.clicked.connect(self.save_simulation_state)
        buttons_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("Load State")
        self.load_button.clicked.connect(self.load_simulation_state)
        buttons_layout.addWidget(self.load_button)
        
        sim_layout.addLayout(buttons_layout)
        
        # Controle de velocidade
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Simulation Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(20)
        self.speed_slider.setValue(1)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self.set_simulation_speed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("1")
        speed_layout.addWidget(self.speed_label)
        sim_layout.addLayout(speed_layout)
        
        # Contador de passos
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Current Step:"))
        self.step_display = QLabel("0")
        step_layout.addWidget(self.step_display)
        sim_layout.addLayout(step_layout)
        
        # Recompensa (para aprendizado por reforço)
        if self.assembly_type == "enhanced":
            reward_layout = QHBoxLayout()
            reward_layout.addWidget(QLabel("Reward:"))
            self.reward_slider = QSlider(Qt.Horizontal)
            self.reward_slider.setMinimum(-100)
            self.reward_slider.setMaximum(100)
            self.reward_slider.setValue(0)
            self.reward_slider.setTickPosition(QSlider.TicksBelow)
            self.reward_slider.setTickInterval(10)
            self.reward_slider.valueChanged.connect(self.set_reward)
            reward_layout.addWidget(self.reward_slider)
            self.reward_label = QLabel("0.00")
            reward_layout.addWidget(self.reward_label)
            sim_layout.addLayout(reward_layout)
        
        sim_group.setLayout(sim_layout)
        controls_layout.addWidget(sim_group)
        
        # Grupo de padrões de entrada
        input_group = QGroupBox("Input Pattern")
        input_layout = QVBoxLayout()
        
        # Seleção de padrão
        self.pattern_combo = QComboBox()
        for pattern_name in self.input_patterns.keys():
            self.pattern_combo.addItem(pattern_name)
        self.pattern_combo.currentTextChanged.connect(self.change_pattern)
        input_layout.addWidget(self.pattern_combo)
        
        # Visualização do padrão
        self.pattern_display = pg.PlotWidget()
        self.pattern_curve = self.pattern_display.plot(pen='y')
        self.update_pattern_display()
        input_layout.addWidget(self.pattern_display)
        
        # Editor de padrão
        pattern_edit_layout = QHBoxLayout()
        pattern_edit_layout.addWidget(QLabel("Add noise:"))
        self.noise_spinner = QDoubleSpinBox()
        self.noise_spinner.setRange(0, 1)
        self.noise_spinner.setSingleStep(0.05)
        self.noise_spinner.setValue(0.1)
        pattern_edit_layout.addWidget(self.noise_spinner)
        input_layout.addLayout(pattern_edit_layout)
        
        # Botão para adicionar novo padrão
        self.add_pattern_button = QPushButton("Add New Pattern")
        self.add_pattern_button.clicked.connect(self.add_new_pattern)
        input_layout.addWidget(self.add_pattern_button)
        
        input_group.setLayout(input_layout)
        controls_layout.addWidget(input_group)
        
        # Grupo de neuromoduladores
        neuro_group = QGroupBox("Neuromodulators")
        neuro_layout = QVBoxLayout()
        
        # Sliders para neuromoduladores
        self.neuromod_sliders = {}
        for name in ["Dopamine", "Serotonin", "Acetylcholine", "Norepinephrine"]:
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel(f"{name}:"))
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(200)
            slider.setValue(100)  # Valor padrão = 1.0
            slider.valueChanged.connect(lambda val, nm=name.lower(): self.set_neuromodulator(nm, val/100))
            self.neuromod_sliders[name.lower()] = slider
            slider_layout.addWidget(slider)
            value_label = QLabel("1.00")
            slider_layout.addWidget(value_label)
            # Salva o label para atualizações
            self.neuromod_sliders[f"{name.lower()}_label"] = value_label
            neuro_layout.addLayout(slider_layout)
        
        neuro_group.setLayout(neuro_layout)
        controls_layout.addWidget(neuro_group)
        
        main_layout.addLayout(controls_layout)
        
        # Cria tabs para visualizações
        self.tabs = QTabWidget()
        
        # Tab básica
        self.basic_tab = QWidget()
        basic_layout = QVBoxLayout(self.basic_tab)
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        basic_layout.addWidget(self.canvas)
        self.tabs.addTab(self.basic_tab, "Basic Visualization")
        
        # Tab para visualizações avançadas (só se for assembleia avançada)
        if self.assembly_type == "enhanced":
            self.enhanced_tab = EnhancedVisualizationTab(self.assembly)
            self.tabs.addTab(self.enhanced_tab, "Advanced Visualization")
            
            # Tab para configurações de regras de aprendizado
            self.learning_tab = LearningRulesTab(self.assembly)
            self.tabs.addTab(self.learning_tab, "Learning Rules")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Simulation ready")
        
        # Inicializa visualização
        self.update_visualization()
    
    def update_pattern_display(self):
        """Atualiza a visualização do padrão atual"""
        pattern = self.input_patterns[self.current_pattern].copy()
        self.pattern_curve.setData(pattern)
    
    def change_pattern(self, pattern_name):
        """Muda o padrão de entrada atual"""
        self.current_pattern = pattern_name
        self.update_pattern_display()
    
    def add_new_pattern(self):
        """Adiciona um novo padrão de entrada"""
        # Cria um padrão aleatório
        new_pattern = np.random.binomial(1, 0.3, self.assembly_size)
        
        # Adiciona à lista de padrões
        pattern_name = f"Pattern {len(self.input_patterns) + 1}"
        self.input_patterns[pattern_name] = new_pattern
        
        # Atualiza o combobox
        self.pattern_combo.addItem(pattern_name)
        self.pattern_combo.setCurrentText(pattern_name)
    
    def set_simulation_speed(self, value):
        """Define a velocidade da simulação"""
        self.simulation_speed = value
        self.speed_label.setText(str(value))
    
    def set_neuromodulator(self, name, value):
        """Atualiza o valor do neuromodulador"""
        # Atualiza o valor do neuromodulador
        if name == "dopamine":
            self.assembly.neuromodulators.dopamine = value
        elif name == "serotonin":
            self.assembly.neuromodulators.serotonin = value
        elif name == "acetylcholine":
            self.assembly.neuromodulators.acetylcholine = value
        elif name == "norepinephrine":
            self.assembly.neuromodulators.norepinephrine = value
        
        # Atualiza o label
        self.neuromod_sliders[f"{name}_label"].setText(f"{value:.2f}")
    
    def set_reward(self, value):
        """Define o valor de recompensa para aprendizado por reforço"""
        self.current_reward = value / 100.0
        self.reward_label.setText(f"{self.current_reward:.2f}")
    
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
    
    def reset_simulation(self):
        """Reinicia a simulação"""
        self.timestep = 0
        
        # Recria a assembleia do tipo apropriado
        if self.assembly_type == "standard":
            self.assembly = ComplexNeuralAssembly(self.assembly_size)
        else:
            self.assembly = EnhancedLearningAssembly(self.assembly_size)
            
        self.visualizer = ComplexInteractionVisualizer(self.assembly)
        
        # Atualiza tabs se necessário
        if self.assembly_type == "enhanced":
            self.enhanced_tab.assembly = self.assembly
            self.learning_tab.assembly = self.assembly
        
        self.step_display.setText("0")
        self.update_visualization()
        self.statusBar().showMessage("Simulation reset")
    
    def save_simulation_state(self):
        """Salva o estado atual da simulação"""
        try:
            import pickle
            
            # Abre diálogo para selecionar arquivo
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Simulation State", "", "Pickle Files (*.pkl)")
            
            if file_path:
                # Salva estado usando pickle
                with open(file_path, 'wb') as f:
                    pickle.dump({
                        'timestep': self.timestep,
                        'assembly': self.assembly,
                        'assembly_type': self.assembly_type,
                        'assembly_size': self.assembly_size,
                        'input_patterns': self.input_patterns,
                        'current_pattern': self.current_pattern
                    }, f)
                
                self.statusBar().showMessage(f"Simulation state saved to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving simulation state: {str(e)}")
    
    def load_simulation_state(self):
        """Carrega um estado de simulação previamente salvo"""
        try:
            import pickle
            
            # Abre diálogo para selecionar arquivo
            file_path, _ = QFileDialog.getOpenFileName(self, "Load Simulation State", "", "Pickle Files (*.pkl)")
            
            if file_path:
                # Carrega estado usando pickle
                with open(file_path, 'rb') as f:
                    state = pickle.load(f)
                
                # Aplica estado
                self.timestep = state['timestep']
                self.assembly = state['assembly']
                self.assembly_type = state['assembly_type']
                self.assembly_size = state['assembly_size']
                self.input_patterns = state['input_patterns']
                self.current_pattern = state['current_pattern']
                
                # Cria novo visualizador
                self.visualizer = ComplexInteractionVisualizer(self.assembly)
                
                # Atualiza interface
                self.step_display.setText(str(self.timestep))
                
                # Atualiza pattern combobox
                self.pattern_combo.clear()
                for pattern_name in self.input_patterns.keys():
                    self.pattern_combo.addItem(pattern_name)
                self.pattern_combo.setCurrentText(self.current_pattern)
                self.update_pattern_display()
                
                # Atualiza tabs se necessário
                if self.assembly_type == "enhanced":
                    if not hasattr(self, 'enhanced_tab'):
                        self.enhanced_tab = EnhancedVisualizationTab(self.assembly)
                        self.tabs.addTab(self.enhanced_tab, "Advanced Visualization")
                        
                        self.learning_tab = LearningRulesTab(self.assembly)
                        self.tabs.addTab(self.learning_tab, "Learning Rules")
                    else:
                        self.enhanced_tab.assembly = self.assembly
                        self.learning_tab.assembly = self.assembly
                        
                self.update_visualization()
                self.statusBar().showMessage(f"Simulation state loaded from {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading simulation state: {str(e)}")
    
    def update_simulation(self):
        """Atualiza a simulação para o próximo passo"""
        for _ in range(self.simulation_speed):
            # Obtém o padrão de entrada atual
            input_pattern = self.input_patterns[self.current_pattern].copy()
            
            # Adiciona ruído
            noise_level = self.noise_spinner.value()
            input_pattern += np.random.normal(0, noise_level, len(input_pattern))
            
            # Atualiza a assembleia
            if self.assembly_type == "enhanced":
                # Use a versão avançada com recompensa
                self.assembly.update(input_pattern, self.timestep, self.current_reward)
            else:
                # Use a versão padrão
                self.assembly.update(input_pattern, self.timestep)
            
            # Incrementa o contador de passos
            self.timestep += 1
        
        # Atualiza o display
        self.step_display.setText(str(self.timestep))
        
        # Atualiza a visualização a cada 10 passos para desempenho
        if self.timestep % 10 == 0:
            self.update_visualization()
            # Atualiza a visualização avançada se disponível
            if self.assembly_type == "enhanced" and self.tabs.currentWidget() == self.enhanced_tab:
                self.enhanced_tab.update_visualization()
    
    def update_visualization(self):
        """Atualiza a visualização básica"""
        # Limpa a figura
        self.figure.clear()
        
        # Matriz de conectividade
        ax1 = self.figure.add_subplot(231)
        matrix = np.zeros((self.assembly.size, self.assembly.size))
        for (i, j), conn in self.assembly.connections.items():
            matrix[i, j] = conn.weight if conn.type == InteractionType.EXCITATORY else -conn.weight
        im = ax1.imshow(matrix, cmap='RdBu_r')
        ax1.set_title('Connectivity Matrix')
        self.figure.colorbar(im, ax=ax1)
        
        # Níveis de cálcio
        ax2 = self.figure.add_subplot(232)
        ax2.plot(self.assembly.calcium_levels)
        ax2.set_title('Calcium Levels')
        
        # Distribuição de pesos
        ax3 = self.figure.add_subplot(233)
        weights = [conn.weight for conn in self.assembly.connections.values()]
        ax3.hist(weights, bins=20)
        ax3.set_title('Weight Distribution')
        
        # Neuromoduladores
        ax4 = self.figure.add_subplot(234)
        levels = [
            self.assembly.neuromodulators.dopamine,
            self.assembly.neuromodulators.serotonin,
            self.assembly.neuromodulators.acetylcholine,
            self.assembly.neuromodulators.norepinephrine
        ]
        ax4.bar(['DA', '5-HT', 'ACh', 'NE'], levels)
        ax4.set_title('Neuromodulator Levels')
        
        # Atividade recente
        ax5 = self.figure.add_subplot(235)
        history_len = min(100, len(self.assembly.activation_history))
        if history_len > 0:
            recent = self.assembly.activation_history[-history_len:]
            activity = [len(act) for act in recent]
            ax5.plot(activity)
            ax5.set_title('Recent Activity')
            
        # Síntese de proteínas
        ax6 = self.figure.add_subplot(236)
        ax6.plot(self.assembly.protein_synthesis)
        ax6.set_title('Protein Synthesis')
        
        self.figure.tight_layout()
        self.canvas.draw()