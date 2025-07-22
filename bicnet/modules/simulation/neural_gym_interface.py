# neural_gym_interface.py

import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSpinBox, 
                            QCheckBox, QGroupBox, QTabWidget, QSlider)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch

from isaac_gym_wrapper import (
    is_isaac_gym_available, 
    get_available_isaac_environments,
    IsaacGymHumanoid,
    IsaacGymAnt
)


# Importa nossas classes neurais
from enhanced_learning import EnhancedLearningAssembly, LearningRule
from complex_neural import InteractionType

class GymEnvironmentWrapper:
    """Wrapper para ambientes do Gymnasium que se conecta à nossa rede neural"""
    
    def __init__(self, env_name="CartPole-v1"):
        """Inicializa o ambiente e variáveis de estado"""
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.observation_space_size = self.env.observation_space.shape[0]
        
        # Para ambientes com espaço de ações discreto
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_space_size = self.env.action_space.n
            self.is_discrete = True
        else:
            self.action_space_size = self.env.action_space.shape[0]
            self.is_discrete = False
        
        # Estado atual
        self.observation = None
        self.last_action = None
        self.reward = 0
        self.done = False
        self.info = None
        self.truncated = False
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
        self.step_count = 0
        
        # Para renderização
        self.last_render = None
    
    def reset(self):
        """Reseta o ambiente e retorna a observação inicial"""
        self.observation, self.info = self.env.reset()
        self.reward = 0
        self.done = False
        self.truncated = False
        self.last_action = None
        self.current_episode_reward = 0
        self.step_count = 0
        return self.observation
    
    def step(self, action):
        """Executa uma ação no ambiente"""
        self.last_action = action
        self.observation, self.reward, self.done, self.truncated, self.info = self.env.step(action)
        self.current_episode_reward += self.reward
        self.step_count += 1
        
        if self.done or self.truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            
        return self.observation, self.reward, self.done, self.truncated, self.info
    
    def render(self):
        """Renderiza o ambiente"""
        self.last_render = self.env.render()
        return self.last_render
    
    def close(self):
        """Fecha o ambiente"""
        self.env.close()
    
    def preprocess_observation(self, observation, size=100):
        """Converte a observação para um formato adequado para a rede neural
        
        Parâmetros:
            observation: observação original do ambiente
            size: tamanho do vetor de entrada para a rede neural
        """
        # Normaliza observação para faixa [0, 1]
        if isinstance(observation, np.ndarray):
            # Verifique se os limites do espaço de observação são finitos
            if np.all(np.isfinite(self.env.observation_space.low)) and np.all(np.isfinite(self.env.observation_space.high)):
                # Normaliza entre [0, 1] usando os limites do espaço de observação
                normalized = (observation - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)
            else:
                # Se os limites não são finitos, use tanh para normalizar
                normalized = np.tanh(observation) * 0.5 + 0.5
        else:
            # Caso seja um único número
            normalized = np.array([float(observation)])
        
        # Verifica se a normalização funcionou corretamente
        if np.any(np.isnan(normalized)):
            normalized = np.zeros_like(normalized)
        
        # Escala para o tamanho desejado para a rede neural
        # Repete os valores para preencher o vetor de tamanho 'size'
        repetitions = size // len(normalized) + 1
        input_pattern = np.tile(normalized, repetitions)[:size]
        
        # Garante que os valores estão entre 0 e 1
        input_pattern = np.clip(input_pattern, 0, 1)
        
        return input_pattern
    
    def process_neural_output(self, neural_activity):
        """Converte a atividade neural em ações para o ambiente"""
        if self.is_discrete:
            # Para ambientes com ações discretas, divide a rede em regiões
            # e escolhe a ação com base na região mais ativa
            region_size = len(neural_activity) // self.action_space_size
            regions = [
                np.mean(neural_activity[i*region_size:(i+1)*region_size])
                for i in range(self.action_space_size)
            ]
            action = np.argmax(regions)
            
        else:
            # Para ambientes com ações contínuas, mapeia atividade para intervalo de ação
            # Vamos simplificar e usar a média de cada região para cada dimensão da ação
            region_size = len(neural_activity) // self.action_space_size
            action = np.zeros(self.action_space_size)
            
            for i in range(self.action_space_size):
                if i * region_size < len(neural_activity):
                    region_activity = neural_activity[i*region_size:(i+1)*region_size]
                    # Mapeia média da atividade [0,1] para o intervalo da ação [-1,1]
                    region_mean = np.mean(region_activity)
                    action[i] = region_mean * 2 - 1
            
            # Garante que a ação está dentro dos limites
            action = np.clip(action, -1, 1)
        
        return action
    
   #@staticmethod
    def get_available_environments():
      """Retorna todos os ambientes disponíveis por tipo"""
      environments = {}

      # Gymnasium
      try:
          gym_envs = list(gym.envs.registry.keys())
          environments['gym'] = sorted(set(gym_envs))
      except Exception as e:
          print(f"Erro ao obter ambientes Gymnasium: {e}")
          environments['gym'] = []

      # Isaac Gym
      if is_isaac_gym_available():
          environments['isaac'] = get_available_isaac_environments()
      else:
          environments['isaac'] = []

      return environments
    
    #@staticmethod
    def create_environment(env_name, env_type, **kwargs):
        """Factory method para criar o wrapper apropriado"""
        if env_type == 'gym':
            return GymEnvironmentWrapper(env_name)
        elif env_type == 'isaac':
            if env_name == "Humanoid":
                return IsaacGymHumanoid(headless=kwargs.get('headless', False))
            elif env_name == "Ant":
                return IsaacGymAnt(headless=kwargs.get('headless', False))
            else:
                raise ValueError(f"Ambiente Isaac Gym desconhecido: {env_name}")
        else:
            raise ValueError(f"Tipo de ambiente desconhecido: {env_type}") 


class NeuralGymGUI(QMainWindow):
    """Interface gráfica para controlar agentes neurais em ambientes 3D"""
    
    def __init__(self):
        super().__init__()
        
        # Configuração da janela
        self.setWindowTitle("Neural Gym Interface")
        self.setGeometry(100, 100, 1200, 800)
        
        # Estado da simulação
        self.running = False
        self.update_interval = 50  # ms
        self.neural_size = 100
        
        # Cria assembleia neural com aprendizado por reforço
        self.assembly = EnhancedLearningAssembly(self.neural_size)
        self.assembly.set_learning_rule(LearningRule.REINFORCEMENT, True)
        self.assembly.set_learning_rule(LearningRule.BCM, True)
        self.assembly.set_learning_rule(LearningRule.COMPETITIVE, True)
        
        # Obtém ambientes disponíveis
        self.available_environments = GymEnvironmentWrapper.get_available_environments()
        
        # Define ambiente e tipo padrão
        self.env_type = 'gym'  # padrão para Gymnasium
        if 'gym' in self.available_environments and self.available_environments['gym']:
            self.selected_environment = self.available_environments['gym'][0]
        elif 'isaac' in self.available_environments and self.available_environments['isaac']:
            self.env_type = 'isaac'
            self.selected_environment = self.available_environments['isaac'][0]
        else:
            self.selected_environment = "CartPole-v1"  # fallback
        
        # Inicializa ambiente
        self.env = None
        
        # Timer para simulação
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        
        # Configura interface
        self.setup_ui()
        
        # Inicializa o ambiente selecionado
        self.initialize_environment()
    
    def setup_ui(self):
        """Configura a interface do usuário"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Controles superiores
        controls_layout = QHBoxLayout()
        
        # Grupo de controles de ambiente
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        # Seleção de tipo de ambiente
        env_type_layout = QHBoxLayout()
        env_type_layout.addWidget(QLabel("Environment Type:"))
        self.env_type_combo = QComboBox()
        
        # Adiciona tipos de ambiente disponíveis
        for env_type in self.available_environments.keys():
            if self.available_environments[env_type]:  # Se houver ambientes deste tipo
                self.env_type_combo.addItem(env_type)
        
        self.env_type_combo.setCurrentText(self.env_type)
        self.env_type_combo.currentTextChanged.connect(self.change_environment_type)
        env_type_layout.addWidget(self.env_type_combo)
        env_layout.addLayout(env_type_layout)
        
        # Seleção de ambiente
        env_selection = QHBoxLayout()
        env_selection.addWidget(QLabel("Environment:"))
        self.env_combo = QComboBox()
        self.update_environment_list(self.env_type)
        self.env_combo.currentTextChanged.connect(self.change_environment)
        env_selection.addWidget(self.env_combo)
        env_layout.addLayout(env_selection)
        
        # Botões de controle
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_simulation)
        buttons_layout.addWidget(self.start_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        buttons_layout.addWidget(self.reset_button)
        
        env_layout.addLayout(buttons_layout)
        
        # Informações do ambiente
        env_info = QVBoxLayout()
        self.episode_label = QLabel("Episode: 0")
        self.reward_label = QLabel("Current Reward: 0.0")
        self.total_reward_label = QLabel("Total Reward: 0.0")
        env_info.addWidget(self.episode_label)
        env_info.addWidget(self.reward_label)
        env_info.addWidget(self.total_reward_label)
        env_layout.addLayout(env_info)
        
        env_group.setLayout(env_layout)
        controls_layout.addWidget(env_group)
        
        # Grupo de controles neurais (mantém o mesmo)
        neural_group = QGroupBox("Neural Controller")
        neural_layout = QVBoxLayout()
        
        # Controle de learning rules
        rules_layout = QVBoxLayout()
        rules_layout.addWidget(QLabel("Learning Rules:"))
        
        for rule in [LearningRule.REINFORCEMENT, LearningRule.BCM, LearningRule.HEBBIAN, 
                    LearningRule.COMPETITIVE, LearningRule.OJA]:
            checkbox = QCheckBox(rule.value.capitalize())
            checkbox.setChecked(self.assembly.active_learning_rules[rule])
            checkbox.stateChanged.connect(lambda state, r=rule: self.toggle_learning_rule(r, state))
            rules_layout.addWidget(checkbox)
        
        neural_layout.addLayout(rules_layout)
        
        # Parâmetros de aprendizado
        params_layout = QVBoxLayout()
        params_layout.addWidget(QLabel("Learning Rate:"))
        self.learning_rate_slider = QSlider(Qt.Horizontal)
        self.learning_rate_slider.setMinimum(1)
        self.learning_rate_slider.setMaximum(100)
        self.learning_rate_slider.setValue(int(self.assembly.learning_parameters.learning_rate * 1000))
        self.learning_rate_slider.valueChanged.connect(
            lambda v: setattr(self.assembly.learning_parameters, 'learning_rate', v/1000)
        )
        params_layout.addWidget(self.learning_rate_slider)
        
        params_layout.addWidget(QLabel("Reward Discount:"))
        self.reward_discount_slider = QSlider(Qt.Horizontal)
        self.reward_discount_slider.setMinimum(50)
        self.reward_discount_slider.setMaximum(99)
        self.reward_discount_slider.setValue(int(self.assembly.learning_parameters.reward_discount * 100))
        self.reward_discount_slider.valueChanged.connect(
            lambda v: setattr(self.assembly.learning_parameters, 'reward_discount', v/100)
        )
        params_layout.addWidget(self.reward_discount_slider)
        
        neural_layout.addLayout(params_layout)
        
        neural_group.setLayout(neural_layout)
        controls_layout.addWidget(neural_group)
        
        main_layout.addLayout(controls_layout)
        
        # Área de visualização com tabs (mantém o mesmo)
        self.tabs = QTabWidget()
        
        # Tab de renderização do ambiente
        self.env_tab = QWidget()
        env_tab_layout = QVBoxLayout(self.env_tab)
        self.env_figure = Figure(figsize=(5, 5), dpi=100)
        self.env_canvas = FigureCanvas(self.env_figure)
        env_tab_layout.addWidget(self.env_canvas)
        self.tabs.addTab(self.env_tab, "Environment")
        
        # Tab de visualização da rede neural
        self.neural_tab = QWidget()
        neural_tab_layout = QVBoxLayout(self.neural_tab)
        self.neural_figure = Figure(figsize=(10, 5), dpi=100)
        self.neural_canvas = FigureCanvas(self.neural_figure)
        neural_tab_layout.addWidget(self.neural_canvas)
        self.tabs.addTab(self.neural_tab, "Neural Network")
        
        # Tab de histórico de recompensas
        self.reward_tab = QWidget()
        reward_tab_layout = QVBoxLayout(self.reward_tab)
        self.reward_figure = Figure(figsize=(10, 5), dpi=100)
        self.reward_canvas = FigureCanvas(self.reward_figure)
        reward_tab_layout.addWidget(self.reward_canvas)
        self.tabs.addTab(self.reward_tab, "Reward History")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def update_environment_list(self, env_type):
        """Atualiza a lista de ambientes com base no tipo selecionado"""
        self.env_combo.clear()
        if env_type in self.available_environments:
            self.env_combo.addItems(self.available_environments[env_type])
            if self.env_combo.count() > 0:
                self.env_combo.setCurrentIndex(0)
                self.selected_environment = self.env_combo.currentText()
    
    def change_environment_type(self, env_type):
        """Muda o tipo de ambiente"""
        self.env_type = env_type
        self.update_environment_list(env_type)
        if self.env_combo.count() > 0:
            self.selected_environment = self.env_combo.currentText()
            self.initialize_environment()
    
    def initialize_environment(self):
        """Inicializa o ambiente selecionado"""
        # Fecha o ambiente atual se existir
        if self.env:
            self.env.close()
        
        # Cria um novo ambiente com base no tipo
        try:
            self.env = GymEnvironmentWrapper.create_environment(
                self.selected_environment, 
                self.env_type,
                headless=False  # Para Isaac Gym, podemos usar a visualização embutida ou não
            )
            
            # Reseta o ambiente
            observation = self.env.reset()
            
            # Renderiza o frame inicial
            self.env.render()
            
            # Atualiza visualizações
            self.update_environment_visualization()
            self.update_neural_visualization()
            self.update_reward_visualization()
            
            self.statusBar().showMessage(f"Environment {self.selected_environment} initialized")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Error initializing environment: {str(e)}")
    
    
    def change_environment(self, env_name):
        """Muda para um novo ambiente"""
        self.selected_environment = env_name
        self.initialize_environment()
    
    def toggle_learning_rule(self, rule, state):
        """Ativa ou desativa uma regra de aprendizado"""
        self.assembly.set_learning_rule(rule, state == Qt.Checked)
    
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
        # Reseta o ambiente
        self.env.reset()
        
        # Reinicia a rede neural - Opção: criar nova ou manter os pesos aprendidos
        # Vamos manter os pesos para que o aprendizado seja contínuo
        
        # Atualiza visualizações
        self.update_environment_visualization()
        self.update_neural_visualization()
        self.update_reward_visualization()
        
        self.statusBar().showMessage("Simulation reset")
    
    def update_simulation(self):
        """Atualiza a simulação para o próximo passo"""
        if not self.env:
            return
        
        # Processa a observação atual para entrada da rede neural
        observation = self.env.observation
        input_pattern = self.env.preprocess_observation(observation, self.neural_size)
        
        # Alimenta a rede neural com a observação e obtém ativação
        activation = self.assembly.update(input_pattern, self.env.step_count, self.env.reward)
        
        # Converte a ativação neural em ação
        neural_activity = activation.copy()
        action = self.env.process_neural_output(neural_activity)
        
        # Executa a ação no ambiente
        observation, reward, done, truncated, info = self.env.step(action)
        
        # Atualiza labels
        self.episode_label.setText(f"Episode: {self.env.episode_count}")
        self.reward_label.setText(f"Current Reward: {reward:.2f}")
        self.total_reward_label.setText(f"Total Reward: {self.env.current_episode_reward:.2f}")
        
        # Renderiza o ambiente
        self.env.render()
        
        # Reinicia se o episódio terminou
        if done or truncated:
            self.env.reset()
        
        # Atualiza visualizações periodicamente (não a cada passo para desempenho)
        if self.env.step_count % 5 == 0:
            self.update_environment_visualization()
        
        if self.env.step_count % 10 == 0:
            self.update_neural_visualization()
            
        if done or truncated:
            self.update_reward_visualization()
    
    def update_environment_visualization(self):
        """Atualiza a visualização do ambiente"""
        if not self.env or self.env.last_render is None:
            return
        
        self.env_figure.clear()
        ax = self.env_figure.add_subplot(111)
        
        # Mostra o frame atual
        ax.imshow(self.env.last_render)
        ax.set_title(f"Environment: {self.selected_environment}")
        ax.axis('off')
        
        self.env_figure.tight_layout()
        self.env_canvas.draw()
    
    def update_neural_visualization(self):
        """Atualiza a visualização da rede neural"""
        self.neural_figure.clear()
        
        # Mostra matriz de conectividade
        ax1 = self.neural_figure.add_subplot(121)
        matrix = np.zeros((self.assembly.size, self.assembly.size))
        for (i, j), conn in self.assembly.connections.items():
            matrix[i, j] = conn.weight if conn.type == InteractionType.EXCITATORY else -conn.weight
        im = ax1.imshow(matrix, cmap='RdBu_r')
        ax1.set_title('Neural Connectivity')
        self.neural_figure.colorbar(im, ax=ax1)
        
        # Mostra atividade recente
        ax2 = self.neural_figure.add_subplot(122)
        history_len = min(50, len(self.assembly.activation_history))
        if history_len > 0:
            recent = self.assembly.activation_history[-history_len:]
            activity = [len(act) for act in recent]
            ax2.plot(range(len(activity)), activity)
            ax2.set_title('Neural Activity')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Active Neurons')
        
        self.neural_figure.tight_layout()
        self.neural_canvas.draw()
    
    def update_reward_visualization(self):
        """Atualiza a visualização do histórico de recompensas"""
        self.reward_figure.clear()
        
        # Mostra histórico de recompensa por episódio
        ax = self.reward_figure.add_subplot(111)
        if self.env and len(self.env.episode_rewards) > 0:
            episodes = range(1, len(self.env.episode_rewards) + 1)
            ax.plot(episodes, self.env.episode_rewards, marker='o')
            
            # Adiciona linha de média móvel
            if len(self.env.episode_rewards) >= 5:
                window_size = min(10, len(self.env.episode_rewards))
                moving_avg = np.convolve(self.env.episode_rewards, 
                                        np.ones(window_size)/window_size, 
                                        mode='valid')
                ax.plot(range(window_size, len(self.env.episode_rewards) + 1), 
                        moving_avg, 'r-', label='Moving Average')
                ax.legend()
            
            ax.set_title('Reward Per Episode')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.grid(True)
        
        self.reward_figure.tight_layout()
        self.reward_canvas.draw()
    
    def closeEvent(self, event):
        """Chamado quando a janela é fechada"""
        # Encerra ambiente do gymnasium
        if self.env:
            self.env.close()
        event.accept()


# Função para executar a aplicação
def run_neural_gym():
    """Executa a interface do gymnasium com rede neural"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = NeuralGymGUI()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    run_neural_gym()