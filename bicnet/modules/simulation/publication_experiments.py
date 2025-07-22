import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importa seus modelos originais
from complex_neural import ComplexNeuralAssembly, InteractionType, SynapticConnection
from enhanced_learning import EnhancedLearningAssembly, LearningRule

class PublicationVisualizer:
    """Classe para criar visualizações de alta qualidade para publicação"""
    
    def __init__(self, output_dir="publication_figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuração para figuras de qualidade de publicação
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.linewidth': 1.2,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def save_figure(self, filename, formats=["pdf", "png"]):
        """Salva figura em múltiplos formatos"""
        for fmt in formats:
            plt.savefig(f"{self.output_dir}/{filename}.{fmt}")
    
    def weight_matrix_visualization(self, assembly, title="Weight Matrix"):
        """Visualização da matriz de pesos sinápticos"""
        plt.figure(figsize=(8, 7))
        
        # Cria matriz de pesos
        matrix = np.zeros((assembly.size, assembly.size))
        for (i, j), conn in assembly.connections.items():
            if conn.type == InteractionType.EXCITATORY:
                matrix[i, j] = conn.weight
            else:
                matrix[i, j] = -conn.weight
        
        # Plot com mapa de calor de alta qualidade
        sns.heatmap(matrix, cmap="RdBu_r", center=0, 
                   vmin=-1, vmax=1, square=True,
                   cbar_kws={"shrink": 0.8, "label": "Synaptic Weight"})
        
        plt.title(title)
        plt.xlabel("Postsynaptic Neuron Index")
        plt.ylabel("Presynaptic Neuron Index")
        plt.tight_layout()
        
        self.save_figure("weight_matrix")
        plt.close()
    
    def learning_rule_comparison(self, results_dict, metric="accuracy"):
        """Compara diferentes regras de aprendizado"""
        plt.figure(figsize=(10, 6))
        
        for rule_name, values in results_dict.items():
            # Suaviza a curva para melhor visualização
            window_size = min(50, len(values)//10)
            smoothed = pd.Series(values).rolling(window=window_size, center=True).mean()
            plt.plot(smoothed, label=rule_name, linewidth=2)
        
        plt.xlabel("Training Epoch")
        plt.ylabel(metric.capitalize())
        plt.title("Comparison of Learning Rules")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        self.save_figure("learning_rule_comparison")
        plt.close()
    
    def model_comparison_bar(self, model_results):
        """Gráfico de barras comparando diferentes modelos"""
        plt.figure(figsize=(10, 6))
        
        models = list(model_results.keys())
        accuracies = [model_results[model] for model in models]
        
        # Barra horizontal para melhor legibilidade
        bars = plt.barh(models, accuracies, color=sns.color_palette("muted"))
        
        # Adiciona valores nas barras
        for i, v in enumerate(accuracies):
            plt.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        plt.xlim(0, max(accuracies) * 1.1)
        plt.xlabel("Performance Metric")
        plt.title("Model Comparison")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        self.save_figure("model_comparison")
        plt.close()
    
    def neuromodulator_dynamics(self, history):
        """Visualiza dinâmica dos neuromoduladores ao longo do tempo"""
        plt.figure(figsize=(10, 6))
        
        times = range(len(history["dopamine"]))
        
        plt.plot(times, history["dopamine"], label="Dopamine", linewidth=2)
        plt.plot(times, history["serotonin"], label="Serotonin", linewidth=2)
        plt.plot(times, history["acetylcholine"], label="Acetylcholine", linewidth=2)
        plt.plot(times, history["norepinephrine"], label="Norepinephrine", linewidth=2)
        
        plt.xlabel("Time Step")
        plt.ylabel("Neuromodulator Level")
        plt.title("Neuromodulator Dynamics During Learning")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        self.save_figure("neuromodulator_dynamics")
        plt.close()
    
    def noise_robustness_plot(self, results):
        """Visualiza robustez a ruído"""
        plt.figure(figsize=(8, 6))
        
        noise_levels = sorted(list(results.keys()))
        accuracies = [results[level] for level in noise_levels]
        
        plt.plot(noise_levels, accuracies, 'o-', linewidth=2)
        plt.xlabel("Noise Level (σ)")
        plt.ylabel("Recognition Accuracy")
        plt.title("Noise Robustness Analysis")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        self.save_figure("noise_robustness")
        plt.close()
    
    def assembly_formation_visualization(self, assembly_history, timesteps):
        """Visualiza formação de assembleias neurais em pontos-chave"""
        n_plots = len(timesteps)
        fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        for i, t in enumerate(timesteps):
            if t < len(assembly_history):
                assemblies = assembly_history[t]
                matrix = np.zeros((assembly.size, assembly.size))
                
                # Marca cada neurônio com sua assembleia
                for idx, assembly_set in enumerate(assemblies):
                    for neuron in assembly_set:
                        matrix[neuron, :] = idx + 1
                
                sns.heatmap(matrix, cmap="tab10", ax=axes[i], cbar=False)
                axes[i].set_title(f"t = {t}")
                axes[i].set_xlabel("Neuron Group")
                
                if i == 0:
                    axes[i].set_ylabel("Neuron ID")
        
        plt.tight_layout()
        self.save_figure("assembly_formation")
        plt.close()

class ModelBenchmark:
    """Sistema de benchmark para modelos neurais"""
    
    def __init__(self):
        self.results = {}
    
    def prepare_patterns(self, size=100):
        """Prepara padrões para testes"""
        patterns = {
            'A': np.zeros(size),
            'B': np.zeros(size),
            'C': np.zeros(size)
        }
        
        # Padrão A: primeiros 20% de neurônios
        start_a = 0
        end_a = int(size * 0.2)
        patterns['A'][start_a:end_a] = 1
        
        # Padrão B: neurônios do meio (40-60%)
        start_b = int(size * 0.4)
        end_b = int(size * 0.6)
        patterns['B'][start_b:end_b] = 1
        
        # Padrão C: últimos 20% de neurônios
        start_c = int(size * 0.8)
        end_c = size
        patterns['C'][start_c:end_c] = 1
        
        return patterns
    
    def pattern_recognition_benchmark(self, model, num_epochs=1000, noise_level=0.1):
        """Benchmark de reconhecimento de padrões"""
        patterns = self.prepare_patterns(model.size)
        pattern_names = list(patterns.keys())
        
        accuracy_history = []
        
        # Para cada época
        for epoch in tqdm(range(num_epochs), desc="Pattern Recognition Benchmark"):
            epoch_accuracy = 0
            
            # Apresenta cada padrão
            np.random.shuffle(pattern_names)  # Ordem aleatória
            
            for pattern_name in pattern_names:
                pattern = patterns[pattern_name]
                
                # Adiciona ruído
                noisy_pattern = pattern.copy()
                noisy_pattern += np.random.normal(0, noise_level, len(pattern))
                
                # Processa padrão
                if hasattr(model, 'update'):
                    # Para nosso modelo biologicamente inspirado
                    activation = model.update(noisy_pattern, epoch)
                    
                    # Determina neurônios ativos
                    active_neurons = set(np.where(activation > 0.5)[0])
                    pattern_neurons = set(np.where(pattern > 0.5)[0])
                    
                    # Calcula precisão usando coeficiente de Jaccard
                    if len(active_neurons.union(pattern_neurons)) > 0:
                        similarity = len(active_neurons.intersection(pattern_neurons)) / \
                                    len(active_neurons.union(pattern_neurons))
                        epoch_accuracy += similarity / len(patterns)
                else:
                    # Para modelos convencionais (torch, etc)
                    with torch.no_grad():
                        output = model(torch.FloatTensor(noisy_pattern))
                        predicted = (output > 0.5).float().numpy()
                        
                        # Calcula precisão
                        correct = np.sum((predicted > 0.5) == (pattern > 0.5))
                        accuracy = correct / len(pattern)
                        epoch_accuracy += accuracy / len(patterns)
            
            # Registra precisão média desta época
            accuracy_history.append(epoch_accuracy)
            
            # Log a cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Accuracy = {epoch_accuracy:.4f}")
        
        return {
            "accuracy_history": accuracy_history,
            "final_accuracy": accuracy_history[-1]
        }
    
    def noise_robustness_benchmark(self, model, noise_levels=None, trials=10):
        """Avalia robustez a diferentes níveis de ruído"""
        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
            
        patterns = self.prepare_patterns(model.size)
        results = {}
        
        for noise_level in noise_levels:
            print(f"Testing noise level: {noise_level}")
            accuracies = []
            
            for trial in range(trials):
                trial_accuracy = 0
                
                for pattern_name, pattern in patterns.items():
                    # Adiciona ruído controlado
                    noisy_pattern = pattern.copy()
                    noisy_pattern += np.random.normal(0, noise_level, len(pattern))
                    
                    # Processa padrão
                    if hasattr(model, 'update'):
                        activation = model.update(noisy_pattern, 0)
                        
                        # Determina neurônios ativos
                        active_neurons = set(np.where(activation > 0.5)[0])
                        pattern_neurons = set(np.where(pattern > 0.5)[0])
                        
                        # Calcula Jaccard
                        if len(active_neurons.union(pattern_neurons)) > 0:
                            similarity = len(active_neurons.intersection(pattern_neurons)) / \
                                        len(active_neurons.union(pattern_neurons))
                            trial_accuracy += similarity / len(patterns)
                    else:
                        # Para modelos torch, etc
                        with torch.no_grad():
                            output = model(torch.FloatTensor(noisy_pattern))
                            predicted = (output > 0.5).float().numpy()
                            
                            correct = np.sum((predicted > 0.5) == (pattern > 0.5))
                            accuracy = correct / len(pattern)
                            trial_accuracy += accuracy / len(patterns)
                
                accuracies.append(trial_accuracy)
            
            # Média das tentativas para este nível de ruído
            results[noise_level] = np.mean(accuracies)
        
        return results
    
    def sequence_learning_benchmark(self, model, num_epochs=1000):
        """Avalia capacidade de aprender sequências temporais"""
        # Define sequências para teste
        patterns = self.prepare_patterns(model.size)
        sequences = [
            ["A", "B", "C"],
            ["B", "A", "C"],
            ["C", "C", "A"]
        ]
        
        prediction_accuracy = []
        
        # Treina o modelo nas sequências
        for epoch in tqdm(range(num_epochs), desc="Sequence Learning Benchmark"):
            epoch_accuracy = 0
            
            # Apresenta cada sequência
            for sequence in sequences:
                # Apresenta os primeiros n-1 elementos da sequência
                for i in range(len(sequence) - 1):
                    pattern = patterns[sequence[i]]
                    noisy_pattern = pattern.copy() + np.random.normal(0, 0.1, len(pattern))
                    
                    # Atualiza o modelo com este elemento
                    if hasattr(model, 'update'):
                        model.update(noisy_pattern, epoch)
                    else:
                        # Para modelos convencionais
                        model(torch.FloatTensor(noisy_pattern))
                
                # Agora testa a capacidade de prever o próximo elemento
                last_pattern = patterns[sequence[-2]]
                next_pattern = patterns[sequence[-1]]
                
                # Apresenta o último elemento conhecido e observa previsão
                if hasattr(model, 'update'):
                    activation = model.update(last_pattern, epoch)
                    
                    # Compara ativação com o próximo elemento esperado
                    active_neurons = set(np.where(activation > 0.5)[0])
                    expected_neurons = set(np.where(next_pattern > 0.5)[0])
                    
                    # Calcula precisão Jaccard para sequência
                    if len(active_neurons.union(expected_neurons)) > 0:
                        similarity = len(active_neurons.intersection(expected_neurons)) / \
                                    len(active_neurons.union(expected_neurons))
                        epoch_accuracy += similarity / len(sequences)
                else:
                    # Para modelos convencionais
                    with torch.no_grad():
                        output = model(torch.FloatTensor(last_pattern))
                        predicted = (output > 0.5).float().numpy()
                        
                        correct = np.sum((predicted > 0.5) == (next_pattern > 0.5))
                        accuracy = correct / len(next_pattern)
                        epoch_accuracy += accuracy / len(sequences)
            
            prediction_accuracy.append(epoch_accuracy)
            
            # Log a cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Prediction Accuracy = {epoch_accuracy:.4f}")
        
        return {
            "accuracy_history": prediction_accuracy,
            "final_accuracy": prediction_accuracy[-1]
        }

class ModelComparison:
    """Compara o modelo proposto com modelos de referência"""
    
    def __init__(self, size=100):
        self.size = size
        self.benchmark = ModelBenchmark()
        self.baseline_models = {}
        self.results = {}
    
    def create_baseline_models(self):
        """Cria modelos baseline para comparação"""
        # 1. Modelo STDP padrão (sem regras avançadas)
        stdp_model = ComplexNeuralAssembly(self.size)
        
        # 2. Modelo Hebbiano simples
        hebbian_model = EnhancedLearningAssembly(self.size)
        hebbian_model.set_learning_rule(LearningRule.HEBBIAN, True)
        hebbian_model.set_learning_rule(LearningRule.STDP, False)
        
        # 3. Modelo BCM
        bcm_model = EnhancedLearningAssembly(self.size)
        bcm_model.set_learning_rule(LearningRule.BCM, True)
        bcm_model.set_learning_rule(LearningRule.STDP, False)
        
        # 4. Rede neural artificial simples (MLP)
        class SimpleMLP(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.fc1 = nn.Linear(size, size*2)
                self.fc2 = nn.Linear(size*2, size)
                
                # Para compatibilidade com o benchmark
                self.size = size
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                return torch.sigmoid(self.fc2(x))
        
        mlp_model = SimpleMLP(self.size)
        
        # 5. Modelo Hopfield
        class HopfieldNetwork:
            def __init__(self, size):
                self.size = size
                self.weights = np.zeros((size, size))
                self.stored_patterns = []
            
            def store_pattern(self, pattern):
                # Converte para representação bipolar (-1, 1)
                pattern_bipolar = 2 * (pattern > 0.5).astype(float) - 1
                
                # Atualiza matriz de pesos (regra de Hebb)
                pattern_matrix = np.outer(pattern_bipolar, pattern_bipolar)
                np.fill_diagonal(pattern_matrix, 0)  # Sem auto-conexões
                self.weights += pattern_matrix
                
                self.stored_patterns.append(pattern)
            
            def update(self, pattern, epoch=None):
                # Treina nos primeiros 20% das épocas
                if epoch is not None and epoch < 200:
                    self.store_pattern(pattern)
                
                # Converte para representação bipolar
                pattern_bipolar = 2 * (pattern > 0.5).astype(float) - 1
                
                # Recuperação síncrona
                output = np.sign(np.dot(self.weights, pattern_bipolar))
                
                # Converte de volta para representação (0, 1)
                return (output + 1) / 2
        
        hopfield_model = HopfieldNetwork(self.size)
        
        # Registra modelos
        self.baseline_models = {
            "STDP Standard": stdp_model,
            "Hebbian": hebbian_model,
            "BCM": bcm_model,
            "MLP": mlp_model,
            "Hopfield": hopfield_model
        }
        
        return self.baseline_models
    
    def run_comparison(self, proposed_model, task="pattern_recognition", **kwargs):
        """Compara o modelo proposto com baselines em uma tarefa específica"""
        if not self.baseline_models:
            self.create_baseline_models()
            
        # Adiciona modelo proposto
        models = {**self.baseline_models, "Proposed": proposed_model}
        
        results = {}
        
        # Seleciona benchmark apropriado
        if task == "pattern_recognition":
            benchmark_func = self.benchmark.pattern_recognition_benchmark
        elif task == "noise_robustness":
            benchmark_func = self.benchmark.noise_robustness_benchmark
        elif task == "sequence_learning":
            benchmark_func = self.benchmark.sequence_learning_benchmark
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Executa benchmark para cada modelo
        for name, model in models.items():
            print(f"\nBenchmarking {name} model on {task}...")
            start_time = time.time()
            
            try:
                model_results = benchmark_func(model, **kwargs)
                results[name] = model_results
                
                elapsed = time.time() - start_time
                print(f"Completed in {elapsed:.2f} seconds")
            except Exception as e:
                print(f"Error benchmarking {name}: {str(e)}")
        
        self.results[task] = results
        return results
    
    def run_all_comparisons(self, proposed_model, tasks=None, **kwargs):
        """Executa todas as comparações configuradas"""
        if tasks is None:
            tasks = ["pattern_recognition", "noise_robustness", "sequence_learning"]
            
        all_results = {}
        
        for task in tasks:
            print(f"\n=== Running {task} benchmark ===")
            task_results = self.run_comparison(proposed_model, task, **kwargs)
            all_results[task] = task_results
            
        return all_results
    
    def generate_latex_table(self, task_results):
        """Gera tabela LaTeX com resultados comparativos"""
        models = list(task_results.values())[0].keys()
        
        latex = "\\begin{table}[ht]\n"
        latex += "\\centering\n"
        latex += "\\caption{Model Performance Comparison}\n"
        latex += "\\label{tab:model_comparison}\n"
        latex += "\\begin{tabular}{l" + "c" * len(models) + "}\n"
        latex += "\\hline\n"
        
        # Cabeçalho
        latex += "Task & " + " & ".join([f"{model}" for model in models]) + " \\\\\n"
        latex += "\\hline\n"
        
        # Linhas de dados
        for task, results in task_results.items():
            readable_task = task.replace("_", " ").title()
            
            row = f"{readable_task} & "
            for model in models:
                if model in results:
                    model_result = results[model]
                    
                    # Extrai métricas relevantes
                    if isinstance(model_result, dict) and "final_accuracy" in model_result:
                        value = f"{model_result['final_accuracy']:.3f}"
                    elif isinstance(model_result, dict) and "accuracy_history" in model_result:
                        value = f"{model_result['accuracy_history'][-1]:.3f}"
                    elif isinstance(model_result, float):
                        value = f"{model_result:.3f}"
                    else:
                        value = "N/A"
                    
                    row += f"{value} & "
                else:
                    row += "N/A & "
            
            # Remove o último '& ' e adiciona nova linha
            row = row[:-3] + " \\\\"
            latex += row + "\n"
        
        # Finaliza tabela
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"
        
        return latex

def run_comprehensive_experiments():
    """Executa conjunto completo de experimentos para publicação"""
    # Configura modelo proposto
    print("Initializing enhanced learning assembly...")
    proposed_model = EnhancedLearningAssembly(100)
    
    # Ativa múltiplos mecanismos de aprendizado
    proposed_model.set_learning_rule(LearningRule.BCM, True)
    proposed_model.set_learning_rule(LearningRule.REINFORCEMENT, True)
    proposed_model.set_learning_rule(LearningRule.HEBBIAN, True)
    proposed_model.set_learning_rule(LearningRule.COMPETITIVE, True)
    
    # Configura sistema de comparação
    print("Setting up model comparison...")
    comparison = ModelComparison(size=100)
    visualizer = PublicationVisualizer()
    
    # Parâmetros de experimentação
    experiment_params = {
        "pattern_recognition": {"num_epochs": 1000, "noise_level": 0.1},
        "noise_robustness": {"noise_levels": [0.1, 0.2, 0.3, 0.4, 0.5], "trials": 5},
        "sequence_learning": {"num_epochs": 1000}
    }
    
    # Executa comparações
    print("Running model comparisons...")
    comparison_results = {}
    
    for task, params in experiment_params.items():
        print(f"\nRunning {task} benchmark...")
        task_results = comparison.run_comparison(proposed_model, task, **params)
        comparison_results[task] = task_results
    
    # Cria visualizações para resultados
    print("\nGenerating visualizations...")
    
    # 1. Matriz de pesos do modelo proposto
    visualizer.weight_matrix_visualization(proposed_model)
    
    # 2. Comparação de curvas de aprendizado
    if "pattern_recognition" in comparison_results:
        learning_curves = {}
        for model_name, results in comparison_results["pattern_recognition"].items():
            if isinstance(results, dict) and "accuracy_history" in results:
                learning_curves[model_name] = results["accuracy_history"]
        
        if learning_curves:
            visualizer.learning_rule_comparison(learning_curves)
    
    # 3. Gráfico de barras comparativo
    model_performances = {}
    for model_name in comparison_results["pattern_recognition"]:
        results = comparison_results["pattern_recognition"][model_name]
        if isinstance(results, dict) and "final_accuracy" in results:
            model_performances[model_name] = results["final_accuracy"]
    
    visualizer.model_comparison_bar(model_performances)
    
    # 4. Robustez a ruído
    if "noise_robustness" in comparison_results:
        noise_results = comparison_results["noise_robustness"]["Proposed"]
        visualizer.noise_robustness_plot(noise_results)
    
    # Gera tabela LaTeX para o paper
    latex_table = comparison.generate_latex_table(comparison_results)
    with open("publication_figures/model_comparison_table.tex", "w") as f:
        f.write(latex_table)
    
    print("\nExperiments completed. Results saved to 'publication_figures' directory.")
    return comparison_results

if __name__ == "__main__":
    run_comprehensive_experiments()