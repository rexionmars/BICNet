# robosuite_integration.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSpinBox, 
                            QCheckBox, QGroupBox, QTabWidget, QSlider, QDoubleSpinBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
import time
import cv2

# Importa nossas classes neurais
from enhanced_learning import EnhancedLearningAssembly, LearningRule
from complex_neural import InteractionType

class RobosuiteWrapper:
    """Wrapper para ambientes do Robosuite que se conecta à nossa rede neural"""
    
    def __init__(self, env_name="Lift", robot="Panda", controller="OSC_POSE", render_mode="human"):
      """Inicializa o ambiente do Robosuite com configuração robusta a diferentes versões"""
      try:
          import robosuite as suite
      except ImportError:
          raise ImportError("Robosuite not installed. Please install with: pip install robosuite")
      
      self.env_name = env_name
      self.robot_name = robot
      self.controller_name = controller  # Armazenamos, mas não usamos diretamente
      self.render_mode = render_mode
      
      # Configuração mínima do ambiente - sem especificar controlador explicitamente
      try:
          # Cria o ambiente da forma mais simples possível
          self.env = suite.make(
              env_name=env_name,
              robots=robot,
              has_renderer=(render_mode == "human"),
              has_offscreen_renderer=True,
          )
          print(f"Created environment {env_name} with robot {robot}")
      except Exception as e:
          print(f"Error creating environment: {str(e)}")
          raise
      
      # Informações sobre o espaço de ação e observação
      try:
          # Tenta obter a dimensão da ação de diferentes maneiras
          if hasattr(self.env, 'action_spec'):
              self.action_dim = self.env.action_spec[0].shape[0]
          elif hasattr(self.env, 'action_space'):
              self.action_dim = self.env.action_space.shape[0]
          else:
              # Alguns valores padrão comuns
              if robot == "Panda":
                  self.action_dim = 7
              elif robot == "Sawyer":
                  self.action_dim = 7
              else:
                  self.action_dim = 6  # Default genérico
          
          print(f"Action dimension: {self.action_dim}")
      except Exception as e:
          print(f"Error determining action dimension: {str(e)}")
          self.action_dim = 7  # Default razoável para a maioria dos robôs
      
      # Estado atual
      self.observation = None
      self.last_action = None
      self.reward = 0
      self.done = False
      self.info = {}
      self.truncated = False
      self.episode_rewards = []
      self.current_episode_reward = 0
      self.episode_count = 0
      self.step_count = 0
      self.success_count = 0
      
      # Para renderização
      self.last_render = None
      self.rgb_array = None
    
    def reset(self):
        """Reseta o ambiente e retorna a observação inicial"""
        self.observation = self.env.reset()
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
        
        # Executa a ação
        observation, reward, done, info = self.env.step(action)
        
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.truncated = False  # robosuite não implementa truncated
        
        self.current_episode_reward += reward
        self.step_count += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            
            # Verifica se a tarefa foi concluída com sucesso
            if info.get("success", False):
                self.success_count += 1
            
        return observation, reward, done, self.truncated, info
    
    def render(self):
        """Renderiza o ambiente"""
        if self.render_mode == "human":
            self.env.render()
        
        # Obtém renderização offscreen
        self.rgb_array = self.env.sim.render(
            camera_name="frontview",
            width=512,
            height=512,
            depth=False
        )
        self.last_render = self.rgb_array
        
        return self.last_render
    
    def close(self):
        """Fecha o ambiente"""
        self.env.close()
    
    def preprocess_observation(self, observation, size=100):
        """Converte a observação para um formato adequado para a rede neural"""
        # Verificar se observation é um dict (caso de algumas tarefas do robosuite)
        if isinstance(observation, dict):
            # Extrair dados relevantes (depende da tarefa específica)
            if "robot-state" in observation:
                obs_vector = observation["robot-state"]
            elif "robot0_proprio-state" in observation:
                obs_vector = observation["robot0_proprio-state"]
            else:
                # Tenta combinar todas as observações em um vetor
                obs_vector = np.concatenate([
                    value for key, value in observation.items() 
                    if isinstance(value, np.ndarray) and value.ndim == 1
                ])
        else:
            obs_vector = observation
        
        # Normaliza para faixa [0, 1]
        # Como não temos limites predefinidos, usamos normalização com base em heurística
        obs_vector = np.clip(obs_vector, -10, 10)  # Clipa valores extremos
        normalized = (obs_vector + 10) / 20.0  # Mapeia [-10, 10] para [0, 1]
        
        # Verifica se a normalização funcionou corretamente
        if np.any(np.isnan(normalized)):
            normalized = np.zeros_like(normalized)
        
        # Escala para o tamanho desejado para a rede neural
        repetitions = size // len(normalized) + 1
        input_pattern = np.tile(normalized, repetitions)[:size]
        
        # Garante que os valores estão entre 0 e 1
        input_pattern = np.clip(input_pattern, 0, 1)
        
        return input_pattern
    
    def process_neural_output(self, neural_activity):
        """Converte a atividade neural em ações para o ambiente"""
        # Divide a atividade neural em regiões para cada dimensão da ação
        region_size = len(neural_activity) // self.action_dim
        action = np.zeros(self.action_dim)
        
        for i in range(self.action_dim):
            if i * region_size < len(neural_activity):
                region_activity = neural_activity[i*region_size:(i+1)*region_size]
                # Mapeia média da atividade [0,1] para o intervalo da ação [-1,1]
                region_mean = np.mean(region_activity)
                action[i] = region_mean * 2 - 1
        
        # Garante que a ação está dentro dos limites
        action = np.clip(action, -1, 1)
        
        return action
    
    @staticmethod
    def get_available_robots():
        """Retorna a lista de robôs disponíveis no Robosuite"""
        try:
            import robosuite as suite
            return suite.robots.available_robots()
        except ImportError:
            return []
    
    # Correção para a função get_available_environments no RobosuiteWrapper
    @staticmethod
    def get_available_environments():
        """Retorna a lista de ambientes disponíveis no Robosuite"""
        try:
            import robosuite
            # O robosuite não tem um método auxiliar para listar ambientes,
            # então listaremos manualmente os ambientes padrão disponíveis
            available_envs = [
                "Lift",
                "Stack",
                "NutAssembly",
                "NutAssemblySingle",
                "NutAssemblySquare",
                "NutAssemblyRound",
                "PickPlace",
                "PickPlaceSingle",
                "PickPlaceMilk",
                "PickPlaceBread",
                "PickPlaceCereal",
                "PickPlaceCan",
                "Door",
                "Wipe",
                "TwoArmLift",
                "TwoArmPegInHole",
                "TwoArmHandover"
            ]
            return available_envs
        except ImportError:
            return []
    
    @staticmethod
    def get_available_robots():
        """Retorna a lista de robôs disponíveis no Robosuite"""
        try:
            import robosuite
            # Lista de robôs padrão no robosuite
            available_robots = [
                "Panda",
                "Sawyer",
                "IIWA",
                "Jaco",
                "Kinova3",
                "UR5e",
                "Baxter",
                "Spot"
            ]
            return available_robots
        except ImportError:
            return []
    @staticmethod
    def get_available_controllers():
      """Retorna a lista de controladores disponíveis no Robosuite"""
      try:
        import robosuite as suite
        # Verifica quais controladores estão disponíveis na instalação atual
        available_controllers = []

        # Tenta carregar cada controlador para verificar disponibilidade
        controller_names = ["OSC_POSE", "OSC_POSITION", "JOINT_VELOCITY", 
                            "JOINT_POSITION", "JOINT_TORQUE"]

        for name in controller_names:
            try:
                suite.controllers.config.load_controller_config(default_controller=name)
                available_controllers.append(name)
            except:
                pass

        # Se nenhum controlador for encontrado, retorna pelo menos o OSC_POSE
        if not available_controllers:
            available_controllers = ["OSC_POSE", "OSC_POSITION", "JOINT_VELOCITY", "JOINT_POSITION", "JOINT_TORQUE"]

        return available_controllers
      except ImportError:
        return ["OSC_POSE", "OSC_POSITION", "JOINT_VELOCITY", "JOINT_POSITION", "JOINT_TORQUE"]



class RobosuiteGUI(QMainWindow):
    """Interface para controlar robôs do Robosuite com redes neurais"""
    
    def __init__(self):
        super().__init__()
        
        # Configuração da janela
        self.setWindowTitle("Neural Robosuite Interface")
        self.setGeometry(100, 100, 1200, 800)
        
        # Estado da simulação
        self.running = False
        self.update_interval = 100  # ms
        self.neural_size = 100
        
        # Cria assembleia neural com aprendizado por reforço
        self.assembly = EnhancedLearningAssembly(self.neural_size)
        self.assembly.set_learning_rule(LearningRule.REINFORCEMENT, True)
        self.assembly.set_learning_rule(LearningRule.BCM, True)
        self.assembly.set_learning_rule(LearningRule.COMPETITIVE, True)
        
        # Inicializa informações do Robosuite
        self.available_envs = RobosuiteWrapper.get_available_environments()
        self.available_robots = RobosuiteWrapper.get_available_robots()
        self.available_controllers = RobosuiteWrapper.get_available_controllers()
        
        self.selected_env = self.available_envs[0] if self.available_envs else "Lift"
        self.selected_robot = self.available_robots[0] if self.available_robots else "Panda"
        self.selected_controller = self.available_controllers[0] if self.available_controllers else "OSC_POSE"
        
        # Inicializa o ambiente
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
        env_group = QGroupBox("Robosuite Environment")
        env_layout = QVBoxLayout()
        
        # Seleção de ambiente
        env_selection = QHBoxLayout()
        env_selection.addWidget(QLabel("Environment:"))
        self.env_combo = QComboBox()
        self.env_combo.addItems(self.available_envs)
        self.env_combo.setCurrentText(self.selected_env)
        self.env_combo.currentTextChanged.connect(self.update_selected_env)
        env_selection.addWidget(self.env_combo)
        env_layout.addLayout(env_selection)
        
        # Seleção de robô
        robot_selection = QHBoxLayout()
        robot_selection.addWidget(QLabel("Robot:"))
        self.robot_combo = QComboBox()
        self.robot_combo.addItems(self.available_robots)
        self.robot_combo.setCurrentText(self.selected_robot)
        self.robot_combo.currentTextChanged.connect(self.update_selected_robot)
        robot_selection.addWidget(self.robot_combo)
        env_layout.addLayout(robot_selection)
        
        # Seleção de controlador
        controller_selection = QHBoxLayout()
        controller_selection.addWidget(QLabel("Controller:"))
        self.controller_combo = QComboBox()
        self.controller_combo.addItems(self.available_controllers)
        self.controller_combo.setCurrentText(self.selected_controller)
        self.controller_combo.currentTextChanged.connect(self.update_selected_controller)
        controller_selection.addWidget(self.controller_combo)
        env_layout.addLayout(controller_selection)
        
        # Botões de controle
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_simulation)
        buttons_layout.addWidget(self.start_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        buttons_layout.addWidget(self.reset_button)
        
        self.initialize_button = QPushButton("Initialize Environment")
        self.initialize_button.clicked.connect(self.initialize_environment)
        buttons_layout.addWidget(self.initialize_button)
        
        env_layout.addLayout(buttons_layout)
        
        # Informações do ambiente
        env_info = QVBoxLayout()
        self.episode_label = QLabel("Episode: 0")
        self.reward_label = QLabel("Current Reward: 0.0")
        self.total_reward_label = QLabel("Total Reward: 0.0")
        self.success_label = QLabel("Success Rate: 0.0%")
        env_info.addWidget(self.episode_label)
        env_info.addWidget(self.reward_label)
        env_info.addWidget(self.total_reward_label)
        env_info.addWidget(self.success_label)
        env_layout.addLayout(env_info)
        
        env_group.setLayout(env_layout)
        controls_layout.addWidget(env_group)
        
        # Grupo de controles neurais
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
        params_layout = QGridLayout()
        
        row = 0
        params_layout.addWidget(QLabel("Learning Rate:"), row, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.001, 0.1)
        self.learning_rate_spin.setSingleStep(0.001)
        self.learning_rate_spin.setDecimals(3)
        self.learning_rate_spin.setValue(self.assembly.learning_parameters.learning_rate)
        self.learning_rate_spin.valueChanged.connect(
            lambda v: setattr(self.assembly.learning_parameters, 'learning_rate', v)
        )
        params_layout.addWidget(self.learning_rate_spin, row, 1)
        
        row += 1
        params_layout.addWidget(QLabel("Reward Discount:"), row, 0)
        self.reward_discount_spin = QDoubleSpinBox()
        self.reward_discount_spin.setRange(0.5, 0.99)
        self.reward_discount_spin.setSingleStep(0.01)
        self.reward_discount_spin.setDecimals(2)
        self.reward_discount_spin.setValue(self.assembly.learning_parameters.reward_discount)
        self.reward_discount_spin.valueChanged.connect(
            lambda v: setattr(self.assembly.learning_parameters, 'reward_discount', v)
        )
        params_layout.addWidget(self.reward_discount_spin, row, 1)
        
        row += 1
        params_layout.addWidget(QLabel("Eligibility Decay:"), row, 0)
        self.eligibility_decay_spin = QDoubleSpinBox()
        self.eligibility_decay_spin.setRange(0.5, 0.99)
        self.eligibility_decay_spin.setSingleStep(0.01)
        self.eligibility_decay_spin.setDecimals(2)
        self.eligibility_decay_spin.setValue(self.assembly.learning_parameters.eligibility_decay)
        self.eligibility_decay_spin.valueChanged.connect(
            lambda v: setattr(self.assembly.learning_parameters, 'eligibility_decay', v)
        )
        params_layout.addWidget(self.eligibility_decay_spin, row, 1)
        
        neural_layout.addLayout(params_layout)
        
        neural_group.setLayout(neural_layout)
        controls_layout.addWidget(neural_group)
        
        main_layout.addLayout(controls_layout)
        
        # Área de visualização com tabs
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
    
    def update_selected_env(self, env_name):
        """Atualiza o ambiente selecionado"""
        self.selected_env = env_name
    
    def update_selected_robot(self, robot_name):
        """Atualiza o robô selecionado"""
        self.selected_robot = robot_name
    
    def update_selected_controller(self, controller_name):
        """Atualiza o controlador selecionado"""
        self.selected_controller = controller_name
    
    def initialize_environment(self):
      """Inicializa o ambiente com as configurações atuais"""
      # Fecha o ambiente atual se existir
      if self.env:
          self.env.close()
      
      # Criamos um novo ambiente
      self.statusBar().showMessage(f"Initializing environment: {self.selected_env} with robot: {self.selected_robot}...")
      
      try:
          # Thread para não bloquear a interface durante inicialização (inicialização pode ser lenta)
          import threading
          
          def init_env():
              try:
                  # Importa diretamente dentro da thread para evitar problemas de importação
                  import robosuite as suite
                  
                  # Cria o wrapper com tratamento de erros
                  try:
                      self.env = RobosuiteWrapper(
                          env_name=self.selected_env,
                          robot=self.selected_robot,
                          controller=self.selected_controller,
                          render_mode="human"
                      )
                      
                      # Reseta o ambiente
                      observation = self.env.reset()
                      
                      # Renderiza o frame inicial
                      self.env.render()
                      
                      # Atualiza visualizações
                      self.update_environment_visualization()
                      self.update_neural_visualization()
                      self.update_reward_visualization()
                      
                      self.statusBar().showMessage(f"Environment initialized with action dimension: {self.env.action_dim}")
                  
                  except AttributeError as e:
                      if "load_controller_config" in str(e):
                          self.statusBar().showMessage("Error: Controller configuration issue. This might be due to Robosuite version incompatibility.")
                          print("Detailed error:", str(e))
                          print("Suggestion: Check if you're using the expected version of Robosuite.")
                      else:
                          self.statusBar().showMessage(f"Error: {str(e)}")
                  
                  except Exception as e:
                      self.statusBar().showMessage(f"Error initializing environment: {str(e)}")
                      print("Detailed error:", str(e))
              
              except Exception as e:
                  self.statusBar().showMessage(f"Error: {str(e)}")
          
          # Inicia thread para inicialização
          threading.Thread(target=init_env).start()
          
      except Exception as e:
          self.statusBar().showMessage(f"Error: {str(e)}")
    
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
        if not self.env:
            return
            
        # Reseta o ambiente
        self.env.reset()
        
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
        self.reward_label.setText(f"Current Reward: {reward:.4f}")
        self.total_reward_label.setText(f"Total Reward: {self.env.current_episode_reward:.4f}")
        
        # Calcula e atualiza taxa de sucesso
        if self.env.episode_count > 0:
            success_rate = (self.env.success_count / self.env.episode_count) * 100
            self.success_label.setText(f"Success Rate: {success_rate:.2f}%")
        
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
        # Robosuite retorna a imagem em formato RGB, mas matplotlib espera BGR
        # Invertemos os canais de cor
        rgb_image = self.env.last_render
        if rgb_image is not None:
            # Converte se for imagem OpenGL (invertida verticalmente)
            if rgb_image.shape[0] > rgb_image.shape[1]:
                rgb_image = cv2.flip(rgb_image, 0)
            
            ax.imshow(rgb_image)
            ax.set_title(f"Environment: {self.selected_env} with {self.selected_robot}")
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
        # Encerra ambiente do robosuite
        if self.env:
            self.env.close()
        event.accept()


# Função para executar a aplicação
def run_robosuite_neural():
    """Executa a interface do Robosuite com rede neural"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = RobosuiteGUI()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    run_robosuite_neural()