# Tentativa de construir minha própria biblitoeca para automação da construção de modelos de aprendizado de máquina

# Import de bibliotecas iniciais para análise exploratória de dados e análises estatísticas
def importarBibliotecasEDA():
    try:
        import pandas as pd 
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np 

        print("Bibliotecas importadas com sucesso!")

    except ImportError as e:
        print(f"Erro ao importar bibliotecas: {e}")

