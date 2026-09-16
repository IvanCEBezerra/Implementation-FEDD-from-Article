from pathlib import Path
import sys

# Allow direct run, add project root to path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
try:
    from .nonlinear import generate_nonlinear_series_abrupt
    from .nonlinear import generate_nonlinear_series_gradual
except ImportError:
    from fedd.generators.nonlinear import generate_nonlinear_series_abrupt
    from fedd.generators.nonlinear import generate_nonlinear_series_gradual

def main_nonlinear():
    # Set seed for reproducibility
    np.random.seed(42)

    # General configs from article
    n_per_concept = 3000
    transition_window = 300 # 10% of concept
    n_series_per_type = 20  # 20 abrupt and 20 gradual per group

    # Configs from Table II
    CONFIGS = {
        'nonlinear_1': {
            'model_name': 'nonlinear', # Equation 1 (NLMA)
            'concepts': [
                {'alpha': [0.9, -0.2, 0.8, -0.5], 'sigma_squared': 0.5},
                {'alpha': [-0.3, 1.4, 0.4, -0.5], 'sigma_squared': 1.5},
                {'alpha': [1.5, -0.4, -0.3, 0.2], 'sigma_squared': 2.5},
                {'alpha': [-0.1, 1.4, 0.4, -0.7], 'sigma_squared': 3.5}
            ]
        },
        'nonlinear_2': {
            'model_name': 'star1', # Equation 2 (STAR 1)
            'concepts': [
                {'alpha': [0.9, -0.2, 0.8, -0.5], 'sigma_squared': 0.5},
                {'alpha': [-0.3, 1.4, 0.4, -0.5], 'sigma_squared': 1.5},
                {'alpha': [1.5, -0.4, -0.3, 0.2], 'sigma_squared': 2.5},
                {'alpha': [-0.1, 1.4, 0.4, -0.7], 'sigma_squared': 3.5}
            ]
        },
        'nonlinear_3': {
            'model_name': 'star2', # Equation 3 (STAR 2)
            'concepts': [
                {'alpha': [-0.5, 0.8, -0.2, 0.9], 'sigma_squared': 0.5},
                {'alpha': [-0.5, 0.4, 1.4, -0.3], 'sigma_squared': 1.5},
                {'alpha': [0.2, -0.3, -0.4, 1.5], 'sigma_squared': 2.5},
                {'alpha': [-0.7, 0.4, 1.4, -0.1], 'sigma_squared': 3.5}
            ]
        }
    }

    # Create output dirs
    base_out_dir = Path("data/generated/nonlinear")
    (base_out_dir / "abrupt").mkdir(parents=True, exist_ok=True)
    (base_out_dir / "gradual").mkdir(parents=True, exist_ok=True)

    print("Iniciando a geração de 120 Séries Temporais Não-Lineares...")

    # Main loop over groups
    for group_name, config in CONFIGS.items():
        print(f"\nGerando dados para {group_name} ({config['model_name']})...")
        concepts = config['concepts']
        model_name = config['model_name']

        # Generate 20 instances per type
        for i in range(1, n_series_per_type + 1):
            # Abrupt
            series_abrupt = generate_nonlinear_series_abrupt(concepts, n_per_concept, model_name)
            file_name_abrupt = f"{group_name}_{i}.npy"
            np.save(base_out_dir / "abrupt" / file_name_abrupt, series_abrupt)

            # Gradual
            series_gradual = generate_nonlinear_series_gradual(concepts, n_per_concept, transition_window, model_name)
            file_name_gradual = f"{group_name}_{i}.npy"
            np.save(base_out_dir / "gradual" / file_name_gradual, series_gradual)
            
            if i % 5 == 0:
                print(f"  - {i}/20 séries concluídas...")

    print("\nTodos os arquivos não-lineares foram gerados e salvos com sucesso em 'data/generated/nonlinear/'!")

if __name__ == "__main__":
    main_nonlinear()