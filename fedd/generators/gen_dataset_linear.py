from pathlib import Path
import sys

# Allow direct run, add project root to path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
try:
    from .configs import LINEAR_1
    from .configs import LINEAR_2
    from .configs import LINEAR_3
    from .linear import generate_linear_abrupt
    from .linear import generate_linear_gradual
except ImportError:
    from fedd.generators.configs import LINEAR_1
    from fedd.generators.configs import LINEAR_2
    from fedd.generators.configs import LINEAR_3
    from fedd.generators.linear import generate_linear_abrupt
    from fedd.generators.linear import generate_linear_gradual

def main_linear():
    # 1. Set seed for reproducibility
    np.random.seed(42)

    # 2. General configs from article
    n_per_concept = 3000  # 12000 total / 4 concepts
    n_transition = 300    # 10% of concept size
    n_series_per_type = 20  # 20 abrupt and 20 gradual per group

    # Linear groups mapping
    linear_groups = {
        "linear_1": LINEAR_1,
        "linear_2": LINEAR_2,
        "linear_3": LINEAR_3
    }

    # 3. Create output directories
    base_dir = Path("data/generated/linear")
    (base_dir / "abrupt").mkdir(parents=True, exist_ok=True)
    (base_dir / "gradual").mkdir(parents=True, exist_ok=True)

    print("Iniciando a geração das séries lineares...")

    # 4. Generate series per group
    for group_name, config in linear_groups.items():
        print(f"Gerando 40 séries para o grupo: {group_name}...")
        
        for i in range(n_series_per_type):
            # Generate and save abrupt series
            series_abrupt = generate_linear_abrupt(
                concepts=config, 
                n=n_per_concept
            )
            # Use .npy for fast and compact storage
            filename_abrupt = base_dir / "abrupt" / f"{group_name}_{i+1}.npy"
            np.save(filename_abrupt, series_abrupt)
            
            # Generate and save gradual series
            series_gradual = generate_linear_gradual(
                concepts=config, 
                n_per_concept=n_per_concept, 
                n_transition=n_transition
            )
            filename_gradual = base_dir / "gradual" / f"{group_name}_{i+1}.npy"
            np.save(filename_gradual, series_gradual)

    print("Geração concluída! 120 séries lineares foram salvas na pasta 'data/generated/linear/'.")

if __name__ == "__main__":
    main_linear()
