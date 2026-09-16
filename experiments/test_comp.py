from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from fedd.drift_detection import FEDDDetector
from fedd.drift_detection.elm_ecdd import ELM_ECDD_Detector


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

    boundaries = real_drifts + [series_length]

    # 1. Count false alarms before first drift
    for d in detected_drifts:
        if d < real_drifts[0]:
            false_alarms += 1

    # 2. Check detections per concept
    for i, true_drift in enumerate(real_drifts):
        window_start = true_drift
        window_end = boundaries[i + 1]

        detections_in_window = [
            d for d in detected_drifts 
            if window_start <= d < window_end
        ]

        if len(detections_in_window) == 0:
            missed_drifts += 1
            total_delay += (window_end - window_start)
        else:
            first_detection = detections_in_window[0]
            total_delay += (first_detection - true_drift)
            false_alarms += (len(detections_in_window) - 1)

    miss_detection_rate = (missed_drifts / total_drifts) * 100.0
    return false_alarms, miss_detection_rate, total_delay


def main():
    # 1. Load test series
    series_path = (
        "data/generated/nonlinear/abrupt/"
        "nonlinear_1_1.npy"
    )

    series = np.load(series_path)
    series_length = len(series)
    real_drifts = [3000, 6000, 9000]

    print("==================================================")
    print("COMPARAÇÃO FEDD vs ELM-ECDD")
    print("==================================================")
    print(f"Série analisada: {series_path}")
    print(f"Tamanho da série: {series_length} | Drifts Reais: {real_drifts}\n")

    # 2. Init detectors
    # FEDD (feature distance)
    fedd_detector = FEDDDetector(
        m=300, lambda_param=0.2, W=1.0, C=1.5, distance="pearson"
    )
    
    # ELM-ECDD (prediction error)
    elm_detector = ELM_ECDD_Detector(
        p_lags=5, h_neurons=10, initial_train=1000, n_retrain=400,
        lambda_param=0.2, W=1.0, C=1.5
    )

    fedd_drifts = []
    elm_drifts = []

    # 3. Run both detectors
    print("Processando a série temporal...")
    for value in series:
        # Update FEDD
        fedd_res = fedd_detector.update(value)
        if fedd_res["drift"]:
            fedd_drifts.append(fedd_res["t"])

        # Update ELM-ECDD
        elm_res = elm_detector.update(value)
        if elm_res["drift"]:
            elm_drifts.append(elm_res["t"])

    # 4. Compute metrics
    fedd_fa, fedd_miss, fedd_delay = evaluate_performance(fedd_drifts, real_drifts, series_length)
    elm_fa, elm_miss, elm_delay = evaluate_performance(elm_drifts, real_drifts, series_length)

    # 5. Final report
    print("\n==================================================")
    print("RESULTADOS COMPARATIVOS")
    print("==================================================")
    
    print(f"\n[ FEDD (Feature-based) ]")
    print(f"  Drifts Detectados:    {fedd_drifts}")
    print(f"  Falsos Alarmes:       {fedd_fa}")
    print(f"  Taxa de Miss:         {fedd_miss:.2f}%")
    print(f"  Atraso (Delay) Total: {fedd_delay} instâncias")

    print(f"\n[ ELM-ECDD (Error-based Baseline) ]")
    print(f"  Drifts Detectados:    {elm_drifts}")
    print(f"  Falsos Alarmes:       {elm_fa}")
    print(f"  Taxa de Miss:         {elm_miss:.2f}%")
    print(f"  Atraso (Delay) Total: {elm_delay} instâncias")
    print("==================================================")


if __name__ == "__main__":
    main()