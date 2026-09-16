from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from fedd.drift_detection import FEDDDetector


def evaluate_performance(detected_drifts, real_drifts, series_length):
    """
    Calcula as métricas de performance oficiais do artigo FEDD:
    - Falsos Alarmes
    - Taxa de Miss-detection (%)
    - Atraso na detecção (Delay acumulado)
    """
    total_drifts = len(real_drifts)
    false_alarms = 0
    missed_drifts = 0
    total_delay = 0

    # Concept boundaries, last is end of series
    boundaries = real_drifts + [series_length]

    # 1. Count false alarms before first real drift
    for d in detected_drifts:
        if d < real_drifts[0]:
            false_alarms += 1

    # 2. Check detections per concept
    for i, true_drift in enumerate(real_drifts):
        window_start = true_drift
        window_end = boundaries[i + 1]

        # Detections inside current concept
        detections_in_window = [
            d for d in detected_drifts 
            if window_start <= d < window_end
        ]

        if len(detections_in_window) == 0:
            # Missed detection
            missed_drifts += 1
            
            # Penalty: full concept length as delay
            total_delay += (window_end - window_start)
        else:
            # First detection is true positive
            first_detection = detections_in_window[0]
            total_delay += (first_detection - true_drift)
            
            # Extra detections are false alarms
            false_alarms += (len(detections_in_window) - 1)

    miss_detection_rate = (missed_drifts / total_drifts) * 100.0

    return false_alarms, miss_detection_rate, total_delay


def main():

    # 1. Load time series
    series_path = (
        "data/generated/nonlinear/abrupt/"
        "nonlinear_1_1.npy"
    )

    series = np.load(series_path)
    series_length = len(series)

    print("========================================")
    print("TESTE DO FEDD")
    print("========================================")
    print(f"Tamanho da série: {series_length}")

    # 2. Setup detector
    detector = FEDDDetector(
        m=300,
        lambda_param=0.2,
        W=1.0,
        C=1.5,
        distance="pearson" 
    )

    # 3. Ground truth
    real_drifts = [3000, 6000, 9000]

    warnings = []
    drifts = []

    # 4. Process series
    for value in series:

        result = detector.update(value)
        t = result["t"]

        # Show initial feature vector when ready
        if (
            detector.fv0 is not None
            and t == detector.m - 1
        ):
            print("\n========================================")
            print("VETOR DE FEATURES INICIAL (FV0)")
            print("========================================")
            print(detector.fv0)
            print(f"\nQuantidade de features: {len(detector.fv0)}")
            print(f"Norma de FV0: {np.linalg.norm(detector.fv0):.6f}")

        # Handle warning/drift display
        if result["warning"] or result["drift"]:
            pass # Debug prints can go here

        # Save results

        if result["warning"]:
            warnings.append(t)

        if result["drift"]:
            drifts.append(t)

    # 5. Compute metrics
    false_alarms, miss_rate, total_delay = evaluate_performance(
        drifts, 
        real_drifts, 
        series_length
    )

    # 6. Final results

    print("\n========================================")
    print("RESULTADO")
    print("========================================")

    print(f"\nDrifts reais:           {real_drifts}")
    print(f"Drifts detectados:      {drifts}")
    
    print("\n========================================")
    print("MÉTRICAS OFICIAIS DO ARTIGO")
    print("========================================")
    print(f"Falsos Alarmes:         {false_alarms}")
    print(f"Taxa Miss-detection:    {miss_rate:.2f}%")
    print(f"Atraso (Delay) Total:   {total_delay} instâncias")


if __name__ == "__main__":
    main()