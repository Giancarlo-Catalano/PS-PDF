from typing import Optional

import numpy as np
from scipy.stats import t

from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Metric import Metric


class SignificantlyHighAverage(Metric):
    pRef: Optional[PRef]
    pRef_mean: Optional[float]

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.pRef_mean = None

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.pRef_mean = np.average(self.pRef.fitness_array)

    def __repr__(self):
        return "Significance of Core"

    def get_p_value_and_sample_mean(self, ps: PS) -> (float, float):
        observations = self.pRef.fitnesses_of_observations(ps)
        n = len(observations)
        sample_mean = np.average(observations)
        sample_stdev = np.std(observations)

        if n < 1 or sample_stdev == 0:
            return -1, -1

        t_score = (sample_mean - self.pRef_mean) / (sample_stdev / np.sqrt(n))
        p_value = 1 - t.cdf(abs(t_score), df=n - 1)
        return p_value, sample_mean

    def get_single_normalised_score(self, ps: PS) -> float:
        self.used_evaluations += 1
        observations = self.pRef.fitnesses_of_observations(ps)
        n = len(observations)
        sample_mean = np.average(observations)
        sample_stdev = np.std(observations)

        if n < 1 or sample_stdev == 0:
            return 0

        t_score = (sample_mean - self.normalised_population_mean) / (sample_stdev / np.sqrt(n))
        cumulative_score = t.cdf(abs(t_score), df=n - 1)  # p_value = 1 - cumulative_score

        def invert_and_augment(score: float):
            return 1. - np.sqrt(score * (2. - score))
            # return 1. - np.sqrt(1-np.square(score))

        return invert_and_augment(cumulative_score)
