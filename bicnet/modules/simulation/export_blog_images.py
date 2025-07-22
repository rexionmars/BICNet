# export_blog_images.py

import os
import numpy as np
from complex_neural import ComplexNeuralAssembly, InteractionType
from enhanced_learning import EnhancedLearningAssembly, LearningRule
from neural_visualization_exporter import NeuralVisualizationExporter

def export_standard_assembly_images():
    """Gera imagens para a assembleia neural padrão"""
    print("Generating standard assembly images...")
    
    # Cria o exportador
    exporter = NeuralVisualizationExporter("./blog_images")
    
    # Cria assembleia neural
    assembly = ComplexNeuralAssembly(100)
    
    # Simula interações complexas
    for t in range(5000):
        # Gera padrão de entrada
        if t % 100 < 50:  # Alterna entre padrões
            input_pattern = np.zeros(100)
            input_pattern[20:40] = 1  # Padrão A
        else:
            input_pattern = np.zeros(100)
            input_pattern[60:80] = 1  # Padrão B
            
        # Adiciona ruído
        input_pattern += np.random.normal(0, 0.1, 100)
        
        # Atualiza assembleia
        assembly.update(input_pattern, t)
        
        # Atualiza neuromoduladores baseado em "recompensa" simulada
        assembly.neuromodulators.update(
            reward=np.sin(t/100),  # Recompensa oscilante
            attention=0.5 + 0.5*np.sin(t/50),  # Atenção variável
            arousal=0.5 + 0.3*np.cos(t/75)  # Arousal variável
        )
        
        # Gera visualizações em pontos-chave
        if t == 10:
            print(f"  Generating images at timestep {t} (initial state)")
            exporter.generate_standard_assembly_visualizations(assembly, t)
        elif t == 1000:
            print(f"  Generating images at timestep {t} (early learning)")
            exporter.generate_standard_assembly_visualizations(assembly, t)
        elif t == 3000:
            print(f"  Generating images at timestep {t} (mid learning)")
            exporter.generate_standard_assembly_visualizations(assembly, t)
        elif t == 4990:
            print(f"  Generating images at timestep {t} (final state)")
            exporter.generate_standard_assembly_visualizations(assembly, t)
    
    print("Standard assembly images complete!")
    return assembly

def export_enhanced_assembly_images():
    """Gera imagens para a assembleia neural aprimorada"""
    print("Generating enhanced assembly images...")
    
    # Cria o exportador
    exporter = NeuralVisualizationExporter("./blog_images")
    
    # Cria assembleia com aprendizado ampliado
    assembly = EnhancedLearningAssembly(100)
    
    # Ativa múltiplos mecanismos de aprendizado
    assembly.set_learning_rule(LearningRule.BCM, True)
    assembly.set_learning_rule(LearningRule.REINFORCEMENT, True)
    assembly.set_learning_rule(LearningRule.COMPETITIVE, True)
    assembly.set_learning_rule(LearningRule.HEBBIAN, True)
    
    # Padrões de entrada para demonstração
    patterns = {
        'A': np.zeros(100),
        'B': np.zeros(100),
        'C': np.zeros(100)
    }
    patterns['A'][10:30] = 1
    patterns['B'][40:60] = 1
    patterns['C'][70:90] = 1
    
    # Simula aprendizado
    for t in range(5000):
        # Seleciona padrão baseado no tempo
        if t % 200 < 100:
            pattern_name = 'A'
        elif t % 200 < 150:
            pattern_name = 'B'
        else:
            pattern_name = 'C'
        
        pattern = patterns[pattern_name].copy()
        
        # Adiciona ruído
        pattern += np.random.normal(0, 0.1, 100)
        
        # Calcula recompensa baseada no padrão
        if pattern_name == 'A':
            reward = 0.5 + 0.5 * np.sin(t / 200)
        elif pattern_name == 'B':
            reward = 1.0 - 0.5 * np.sin(t / 200)
        else:
            reward = 0.1
        
        # Atualiza assembleia
        assembly.update(pattern, t, reward)
        
        # Gera visualizações em pontos-chave
        if t == 10:
            print(f"  Generating images at timestep {t} (initial state)")
            exporter.generate_enhanced_assembly_visualizations(assembly, t)
        elif t == 1000:
            print(f"  Generating images at timestep {t} (early learning)")
            exporter.generate_enhanced_assembly_visualizations(assembly, t)
        elif t == 3000:
            print(f"  Generating images at timestep {t} (mid learning)")
            exporter.generate_enhanced_assembly_visualizations(assembly, t)
        elif t == 4990:
            print(f"  Generating images at timestep {t} (final state)")
            exporter.generate_enhanced_assembly_visualizations(assembly, t)
            
        # Periodicamente detecta assembleias neurais
        if t % 500 == 0:
            assembly.detect_neural_assemblies()
    
    print("Enhanced assembly images complete!")
    return assembly

def export_demonstration_images():
    """Gera imagens para as demonstrações neurais"""
    print("Generating demonstration images...")
    
    # Cria o exportador
    exporter = NeuralVisualizationExporter("./blog_images")
    
    # === Pattern Recognition ===
    print("  Setting up Pattern Recognition demo...")
    assembly = EnhancedLearningAssembly(100)
    
    # Ativa aprendizado por reforço e BCM
    assembly.set_learning_rule(LearningRule.REINFORCEMENT, True)
    assembly.set_learning_rule(LearningRule.BCM, True)
    
    # Estado da demonstração
    demo_state = {
        "patterns": {
            "A": np.zeros(assembly.size),
            "B": np.zeros(assembly.size),
            "C": np.zeros(assembly.size)
        },
        "current_pattern": "A",
        "pattern_history": [],
        "response_history": [],
        "noise_level": 0.1
    }
    
    # Define os padrões
    a_start = int(assembly.size * 0.1)
    a_end = int(assembly.size * 0.3)
    demo_state["patterns"]["A"][a_start:a_end] = 1
    
    b_start = int(assembly.size * 0.4)
    b_end = int(assembly.size * 0.6)
    demo_state["patterns"]["B"][b_start:b_end] = 1
    
    c_start = int(assembly.size * 0.7)
    c_end = int(assembly.size * 0.9)
    demo_state["patterns"]["C"][c_start:c_end] = 1
    
    # Simulação
    for t in range(1000):
        # Alterna padrões para treinamento
        if t % 300 < 100:
            demo_state["current_pattern"] = "A"
            reward = 0.8
        elif t % 300 < 200:
            demo_state["current_pattern"] = "B"
            reward = 0.2
        else:
            demo_state["current_pattern"] = "C"
            reward = -0.2
        
        # Obtém o padrão atual
        current_pattern = demo_state["patterns"][demo_state["current_pattern"]].copy()
        
        # Adiciona ruído
        current_pattern += np.random.normal(0, demo_state["noise_level"], assembly.size)
        current_pattern = np.clip(current_pattern, 0, 1)
        
        # Atualiza a rede neural
        activation = assembly.update(current_pattern, t, reward)
        
        # Registra o histórico
        demo_state["pattern_history"].append(demo_state["current_pattern"])
        
        # Calcula a resposta para cada região
        responses = {}
        for pattern_name, pattern in demo_state["patterns"].items():
            active_region = np.where(pattern > 0.5)[0]
            if len(active_region) > 0:
                responses[pattern_name] = np.mean(activation[active_region])
            else:
                responses[pattern_name] = 0
        
        demo_state["response_history"].append(responses)
        
        # Gera visualizações em pontos-chave
        if t == 10:
            print(f"    Generating Pattern Recognition images at timestep {t}")
            exporter.generate_demonstration_visualizations("Pattern Recognition", demo_state, assembly, t)
        elif t == 300:
            print(f"    Generating Pattern Recognition images at timestep {t}")
            exporter.generate_demonstration_visualizations("Pattern Recognition", demo_state, assembly, t)
        elif t == 900:
            print(f"    Generating Pattern Recognition images at timestep {t}")
            exporter.generate_demonstration_visualizations("Pattern Recognition", demo_state, assembly, t)
    
    # === Neuroplasticity ===
    print("  Setting up Neuroplasticity demo...")
    assembly = ComplexNeuralAssembly(100)
    
    # Estado da demonstração
    demo_state = {
        "weight_history": [],
        "selected_connections": [],
        "stimulus_strength": 0.8,
        "stimulus_pattern": np.zeros(assembly.size)
    }
    
    # Seleciona algumas conexões para rastrear
    connections = list(assembly.connections.keys())
    if connections:
        # Seleciona 5 conexões aleatórias
        n_connections = min(5, len(connections))
        demo_state["selected_connections"] = np.random.choice(
            connections, n_connections, replace=False
        ).tolist()
    
    # Define um padrão de estímulo
    start = int(assembly.size * 0.3)
    end = int(assembly.size * 0.7)
    demo_state["stimulus_pattern"] = np.zeros(assembly.size)
    demo_state["stimulus_pattern"][start:end] = 1
    
    # Simulação
    for t in range(500):
        # Obtém o padrão de estímulo atual
        stimulus = demo_state["stimulus_pattern"].copy() * demo_state["stimulus_strength"]
        
        # Adiciona ruído
        stimulus += np.random.normal(0, 0.1, assembly.size)
        stimulus = np.clip(stimulus, 0, 1)
        
        # Atualiza a rede neural
        assembly.update(stimulus, t)
        
        # Registra o histórico
        weights = {}
        for conn_key in demo_state["selected_connections"]:
            if conn_key in assembly.connections:
                weights[conn_key] = assembly.connections[conn_key].weight
        
        demo_state["weight_history"].append(weights)
        
        # Periodicamente muda o padrão de estímulo
        if t == 200:
            # Muda para um padrão diferente
            demo_state["stimulus_pattern"] = np.zeros(assembly.size)
            demo_state["stimulus_pattern"][10:30] = 1
            demo_state["stimulus_pattern"][60:80] = 1
        
        # Gera visualizações em pontos-chave
        if t == 10:
            print(f"    Generating Neuroplasticity images at timestep {t}")
            exporter.generate_demonstration_visualizations("Neuroplasticity Visualization", demo_state, assembly, t)
        elif t == 200:
            print(f"    Generating Neuroplasticity images at timestep {t}")
            exporter.generate_demonstration_visualizations("Neuroplasticity Visualization", demo_state, assembly, t)
        elif t == 480:
            print(f"    Generating Neuroplasticity images at timestep {t}")
            exporter.generate_demonstration_visualizations("Neuroplasticity Visualization", demo_state, assembly, t)
    
    # === Oscillatory Dynamics ===
    print("  Setting up Oscillatory Dynamics demo...")
    assembly = ComplexNeuralAssembly(100)
    
    # Estado da demonstração
    demo_state = {
        "oscillation_mode": "Gamma",
        "activity_history": [],
        "frequency_history": [],
        "visualization_mode": "Time Domain",
        "excitation": 0.7,
        "inhibition": 0.5
    }
    
    # Simulação para Gamma
    for t in range(300):
        # Configura parâmetros de oscilação
        excitation = demo_state["excitation"]
        inhibition = demo_state["inhibition"]
        
        # Definimos os diferentes padrões de entrada para os modos
        oscillation_mode = demo_state["oscillation_mode"]
        
        if oscillation_mode == "Gamma":
            frequency = 0.2  # mais rápido
            duration = 0.3   # mais curto
        
        # Gera um padrão oscilatório
        time_phase = t * frequency
        amplitude = (np.sin(time_phase) + 1) / 2  # Escala para [0, 1]
        
        # Padrão base
        input_pattern = np.zeros(assembly.size)
        
        # Região de neurônios excitatórios
        exc_start = int(assembly.size * 0.2)
        exc_end = int(assembly.size * 0.4)
        input_pattern[exc_start:exc_end] = amplitude * excitation
        
        # Região de neurônios inibitórios (ativamos com atraso de fase)
        inh_start = int(assembly.size * 0.6)
        inh_end = int(assembly.size * 0.8)
        input_pattern[inh_start:inh_end] = ((1-amplitude) * inhibition) * duration
        
        # Atualiza a rede neural
        activation = assembly.update(input_pattern, t)
        
        # Registra o histórico de atividade
        activity_level = len(assembly.activation_history[-1]) if assembly.activation_history else 0
        demo_state["activity_history"].append(activity_level)
        
        # Gera visualizações em pontos-chave
        if t == 100:
            print(f"    Generating Oscillatory Dynamics (Gamma) images at timestep {t}")
            exporter.generate_demonstration_visualizations("Oscillatory Dynamics", demo_state, assembly, t)
    
    # Reconfigura para Beta
    demo_state["oscillation_mode"] = "Beta"
    assembly = ComplexNeuralAssembly(100)  # Reset
    demo_state["activity_history"] = []
    
    # Simulação para Beta
    for t in range(300):
        excitation = demo_state["excitation"]
        inhibition = demo_state["inhibition"]
        
        # Beta: oscilações de amplitude média
        frequency = 0.1
        duration = 0.5
        
        # Gera padrão
        time_phase = t * frequency
        amplitude = (np.sin(time_phase) + 1) / 2
        
        input_pattern = np.zeros(assembly.size)
        exc_start = int(assembly.size * 0.2)
        exc_end = int(assembly.size * 0.4)
        input_pattern[exc_start:exc_end] = amplitude * excitation
        
        inh_start = int(assembly.size * 0.6)
        inh_end = int(assembly.size * 0.8)
        input_pattern[inh_start:inh_end] = ((1-amplitude) * inhibition) * duration
        
        activation = assembly.update(input_pattern, t)
        
        activity_level = len(assembly.activation_history[-1]) if assembly.activation_history else 0
        demo_state["activity_history"].append(activity_level)
        
        # Gera visualizações
        if t == 100:
            print(f"    Generating Oscillatory Dynamics (Beta) images at timestep {t}")
            exporter.generate_demonstration_visualizations("Oscillatory Dynamics", demo_state, assembly, t)
    
    # Reconfigura para Alpha
    demo_state["oscillation_mode"] = "Alpha"
    assembly = ComplexNeuralAssembly(100)  # Reset
    demo_state["activity_history"] = []
    
    # Simulação para Alpha
    for t in range(300):
        excitation = demo_state["excitation"]
        inhibition = demo_state["inhibition"]
        
        # Alpha: oscilações mais lentas
        frequency = 0.05
        duration = 0.7
        
        # Gera padrão
        time_phase = t * frequency
        amplitude = (np.sin(time_phase) + 1) / 2
        
        input_pattern = np.zeros(assembly.size)
        exc_start = int(assembly.size * 0.2)
        exc_end = int(assembly.size * 0.4)
        input_pattern[exc_start:exc_end] = amplitude * excitation
        
        inh_start = int(assembly.size * 0.6)
        inh_end = int(assembly.size * 0.8)
        input_pattern[inh_start:inh_end] = ((1-amplitude) * inhibition) * duration
        
        activation = assembly.update(input_pattern, t)
        
        activity_level = len(assembly.activation_history[-1]) if assembly.activation_history else 0
        demo_state["activity_history"].append(activity_level)
        
        # Gera visualizações
        if t == 100:
            print(f"    Generating Oscillatory Dynamics (Alpha) images at timestep {t}")
            exporter.generate_demonstration_visualizations("Oscillatory Dynamics", demo_state, assembly, t)
    
    # === Neuromodulation Effects ===
    print("  Setting up Neuromodulation Effects demo...")
    assembly = EnhancedLearningAssembly(100)
    assembly.set_learning_rule(LearningRule.REINFORCEMENT, True)
    
    # Estado da demonstração
    demo_state = {
        "current_modulator": "dopamine",
        "modulator_history": [],
        "response_history": [],
        "learning_rate_history": [],
        "stimulus_type": "reward"
    }
    
    # Simulação com diferentes estados de neuromoduladores
    for t in range(800):
        # Configura neuromoduladores para diferentes fases
        if t < 200:
            # Estado normal/baseline
            assembly.neuromodulators.dopamine = 1.0
            assembly.neuromodulators.serotonin = 1.0
            assembly.neuromodulators.acetylcholine = 1.0
            assembly.neuromodulators.norepinephrine = 1.0
            demo_state["stimulus_type"] = "neutral"
        elif t < 400:
            # Estado de recompensa
            assembly.neuromodulators.dopamine = 1.8
            assembly.neuromodulators.serotonin = 1.2
            assembly.neuromodulators.acetylcholine = 1.0
            assembly.neuromodulators.norepinephrine = 1.2
            demo_state["stimulus_type"] = "reward"
        elif t < 600:
            # Estado de estresse
            assembly.neuromodulators.dopamine = 0.7
            assembly.neuromodulators.serotonin = 0.5
            assembly.neuromodulators.acetylcholine = 1.0
            assembly.neuromodulators.norepinephrine = 1.8
            demo_state["stimulus_type"] = "aversive"
        else:
            # Estado de foco/atenção
            assembly.neuromodulators.dopamine = 1.2
            assembly.neuromodulators.serotonin = 1.0
            assembly.neuromodulators.acetylcholine = 1.6
            assembly.neuromodulators.norepinephrine = 1.4
            demo_state["stimulus_type"] = "neutral"
        
        # Prepara o estímulo
        stimulus = np.zeros(assembly.size)
        
        if demo_state["stimulus_type"] == "reward":
            # Estímulo associado a recompensa (região anterior)
            start = int(assembly.size * 0.2)
            end = int(assembly.size * 0.4)
            stimulus[start:end] = 1.0
        elif demo_state["stimulus_type"] == "aversive":
            # Estímulo aversivo (região posterior)
            start = int(assembly.size * 0.6)
            end = int(assembly.size * 0.8)
            stimulus[start:end] = 1.0
        else:  # neutral
            # Estímulo neutro (região central)
            start = int(assembly.size * 0.4)
            end = int(assembly.size * 0.6)
            stimulus[start:end] = 1.0
        
        # Adiciona ruído
        stimulus += np.random.normal(0, 0.1, assembly.size)
        stimulus = np.clip(stimulus, 0, 1)
        
        # Atualiza a rede neural
        activation = assembly.update(stimulus, t, 0.0)  # Sem recompensa externa
        
        # Registra o histórico
        modulator_levels = {
            "dopamine": assembly.neuromodulators.dopamine,
            "serotonin": assembly.neuromodulators.serotonin,
            "acetylcholine": assembly.neuromodulators.acetylcholine,
            "norepinephrine": assembly.neuromodulators.norepinephrine
        }
        
        demo_state["modulator_history"].append(modulator_levels)
        
        # Registra a resposta (atividade neural)
        response = len(assembly.activation_history[-1]) if assembly.activation_history else 0
        demo_state["response_history"].append(response)
        
        # Registra a taxa de aprendizado efetiva
        effective_rate = assembly.learning_parameters.learning_rate * assembly.neuromodulators.dopamine
        demo_state["learning_rate_history"].append(effective_rate)
        
        # Gera visualizações em pontos-chave
        if t == 100:
            print(f"    Generating Neuromodulation (Baseline) images at timestep {t}")
            exporter.generate_demonstration_visualizations("Neuromodulation Effects", demo_state, assembly, t)
        elif t == 300:
            print(f"    Generating Neuromodulation (Reward) images at timestep {t}")
            exporter.generate_demonstration_visualizations("Neuromodulation Effects", demo_state, assembly, t)
        elif t == 500:
            print(f"    Generating Neuromodulation (Stress) images at timestep {t}")
            exporter.generate_demonstration_visualizations("Neuromodulation Effects", demo_state, assembly, t)
        elif t == 700:
            print(f"    Generating Neuromodulation (Focus) images at timestep {t}")
            exporter.generate_demonstration_visualizations("Neuromodulation Effects", demo_state, assembly, t)
    
    print("Demonstration images complete!")

if __name__ == "__main__":
    # Exporta todas as imagens
    export_standard_assembly_images()
    export_enhanced_assembly_images()
    export_demonstration_images()
    
    print("\nAll blog images generated successfully!")