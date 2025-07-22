#!/usr/bin/env python3
"""
Simulador de Redes Neurais Complexas

Este programa integra as seguintes componentes:
1. Um modelo básico de rede neural com mecanismos biológicos (complex_neural.py)
2. Extensões de aprendizado avançado (enhanced_learning.py)
3. Interface gráfica para visualização e controle (neural_gui.py)
4. Integração com ambientes 3D, incluindo Isaac Gym (isaac_gym_wrapper.py)
"""

import sys
import numpy as np
import torch
import matplotlib
import logging
import argparse
from PyQt5.QtWidgets import QApplication

def main():
    """Função principal que inicializa e executa a aplicação"""
    import sys
    import argparse
    from PyQt5.QtWidgets import QApplication
    
    parser = argparse.ArgumentParser(description='Neural Network Simulation')
    parser.add_argument('--gym', action='store_true', help='Run with Gymnasium environments')
    parser.add_argument('--robosuite', action='store_true', help='Run with Robosuite environments')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    if args.robosuite:
        from robosuite_integration import RobosuiteGUI
        window = RobosuiteGUI()
    elif args.gym:
        from neural_gym_interface import NeuralGymGUI
        window = NeuralGymGUI()
    else:
        from neural_gui import NeuralSimulationGUI
        window = NeuralSimulationGUI()
    
    window.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())