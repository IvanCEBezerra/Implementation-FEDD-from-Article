import numpy as np
from fedd.drift_detection.ecdd import ECDDDetector
from fedd.features_extration.distances import cosine_distance
from fedd.features_extration.distances import pearson_distance
from fedd.features_extration.features import extract_features


class FEDDDetector:

    def __init__(
        self,
        m=300,
        lambda_param=0.2,
        W=1.0,
        C=1.5,
        distance="cosine"
    ):
        self.m = m
        self.lambda_param = lambda_param
        self.W = W
        self.C = C
        self.distance_name = distance

        self.samples = []

        self.s = 0
        self.fv0 = None

        self.t = -1

        self.warn = 0
        self.below_warn = 0

        self.ecdd = ECDDDetector(
            lambda_param=lambda_param,
            W=W,
            C=C
        )

    def update(self, value):

        self.samples.append(value)
        self.t += 1

        result = {
            "t": self.t,
            "distance": None,
            "warning": False,
            "warning_state": False,
            "drift": False,
            "Z_t": None,
            "mu_d": None,
            "warning_threshold": None,
            "drift_threshold": None
        }

        # Build initial feature vector
        if (
            self.fv0 is None
            and self.t - self.s + 1 >= self.m
        ):

            initial_window = np.asarray(
                self.samples[
                    self.s:self.s + self.m
                ]
            )

            self.fv0 = extract_features(
                initial_window
            )

        # Detection (wait for windows to separate)
        elif self.t - self.s >= 2 * self.m - 1:

            new_window = np.asarray(
                self.samples[
                    self.t - self.m + 1:self.t + 1
                ]
            )

            fvt = extract_features(
                new_window
            )

            # Compute distance
            if self.distance_name == "cosine":

                distance = cosine_distance(
                    self.fv0,
                    fvt
                )

            elif self.distance_name == "pearson":

                distance = pearson_distance(
                    self.fv0,
                    fvt
                )

            else:

                raise ValueError(
                    f"Unknown distance: {self.distance_name}"
                )

            # Update ECDD
            (
                warning_signal,
                drift_signal,
                below_warning
            ) = self.ecdd.update(distance)

            result["distance"] = distance
            result["warning"] = warning_signal
            result["drift"] = drift_signal

            result["Z_t"] = self.ecdd.Z_t
            result["mu_d"] = self.ecdd.mu_d
            result["warning_threshold"] = (
                self.ecdd.warning_threshold
            )
            result["drift_threshold"] = (
                self.ecdd.drift_threshold
            )

            # Handle warning
            if warning_signal and self.warn == 0:

                self.warn = self.t
                self.below_warn = 0

            # Handle drift
            if drift_signal:

                if self.warn != 0:
                    self.s = self.warn
                else:
                    self.s = self.t

                self.warn = 0
                self.below_warn = 0

                self.fv0 = None

                self.ecdd.reset()

            # Cancel warning if below threshold

            if self.warn != 0:

                if below_warning:
                    self.below_warn += 1
                else:
                    self.below_warn = 0

                if self.below_warn >= 10:
                    self.warn = 0
                    self.below_warn = 0

            result["warning_state"] = (
                self.warn != 0
            )

        return result