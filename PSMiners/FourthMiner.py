import heapq
import random
import warnings
from math import ceil
from typing import TypeAlias

from pandas import DataFrame

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.Averager import Averager
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import MultipleMetrics, Metric
from PSMetric.SecondLinkage import SecondLinkage
from PSMetric.Simplicity import Simplicity
from PSMiners.Individual import Individual, add_metrics, with_aggregated_scores, add_normalised_metrics, \
    with_average_score, with_product_score, partition_by_simplicity
from PSMiners.SamplableSet import EfficientPopulation
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria, EvaluationBudgetLimit, AsLongAsWanted
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# performance issues:
# eq, required by set

Population: TypeAlias = list[Individual]


class FourthMiner:
    population_size: int
    offspring_population_size: int


    current_population: list[Individual]
    archive: set[Individual]
    metric: Metric

    search_space: SearchSpace
    used_evaluations: int

    def __init__(self,
                 population_size: int,
                 offspring_population_size: int,
                 metric: Metric,
                 pRef: PRef):
        self.metric = metric
        self.used_evaluations = 0
        self.metric.set_pRef(pRef)
        self.search_space = pRef.search_space
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.current_population = self.get_initial_population()
        self.archive = set()


    def evaluate(self, population: Population) -> Population:
        for individual in population:
            individual.aggregated_score = self.metric.get_single_normalised_score(individual.ps)
            self.used_evaluations += len(population)
        return population

    def get_initial_population(self) -> Population:
        """ basically takes the elite of the PRef, and converts them into PSs """
        """this is called get_init in the paper"""
        return [Individual(PS.empty(self.search_space))]

    def get_localities(self, individual: Individual) -> list[Individual]:
        return [Individual(ps) for ps in individual.ps.specialisations(self.search_space)]

    def select_one(self) -> Individual:
        tournament_size = 3
        tournament_pool = random.choices(self.current_population, k=tournament_size)
        return max(tournament_pool, key=lambda x: x.aggregated_score)

    def update_population(self):
        """Note how this relies on the current population being evaluated,
        and the new population will also be evaluated"""
        # select and add to archive, so that they won't appear in the population again
        offspring = set()

        remaining_population = set(self.current_population)
        while len(offspring) < self.offspring_population_size and len(remaining_population) > 0:
            selected = self.select_one()
            remaining_population.discard(selected)
            if selected not in self.archive:
                self.archive.add(selected)
                offspring.update(self.get_localities(selected))


        self.current_population = list(remaining_population)
        self.current_population.extend(self.evaluate(list(offspring)))
        self.current_population = self.top(self.population_size)

    def top(self, how_many: int):
        return heapq.nlargest(how_many, self.current_population, key=lambda x: x.aggregated_score)

    def show_best_of_current_population(self, how_many: int):
        for individual in self.top(how_many):
            print(individual)

    def run(self,
            termination_criteria: TerminationCriteria,
            show_each_generation=False):
        iteration = 0

        def termination_criteria_met():
            if len(self.current_population) == 0:
                warnings.warn("The run is ending because the population is empty!!!")
                return True

            return termination_criteria.met(iterations=iteration,
                                            evaluations=self.get_used_evaluations(),
                                            evaluated_population=self.current_population)  # TODO change this

        while not termination_criteria_met():
            self.update_population()
            if show_each_generation:
                print(f"Population at iteration {iteration}, used_budget = {self.get_used_evaluations()}--------------")
                self.show_best_of_current_population(12)
            iteration += 1

        print(f"Execution terminated with {iteration = } and used_budget = {self.get_used_evaluations()}")

    def get_results(self, quantity_returned=None) -> list[Individual]:
        if quantity_returned is None:
            quantity_returned = len(self.archive)
        return heapq.nlargest(quantity_returned, self.archive, key=lambda x: x.aggregated_score)

    def get_used_evaluations(self) -> int:
        return self.used_evaluations


def show_plot_of_individuals(individuals: list[Individual], metrics: MultipleMetrics):
    labels = metrics.get_labels()
    points = [i.metric_scores for i in individuals]

    utils.make_interactive_3d_plot(points, labels)


def test_fourth_archive_miner(problem: BenchmarkProblem,
                              show_each_generation=True):
    print(f"Testing the modified archive miner")
    pRef: PRef = problem.get_pRef(15000)

    budget_limit = AsLongAsWanted()
    # iteration_limit = TerminationCriteria.IterationLimit(12)
    # termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, iteration_limit)

    miner = FourthMiner(150,
                        offspring_population_size=300,
                        pRef=pRef,
                        metric = Averager([MeanFitness(), Linkage()]))

    miner.run(budget_limit, show_each_generation=show_each_generation)

    results = miner.get_results()
    print(f"The used budget is {miner.get_used_evaluations()}")

    print("The top 12 by mean fitness are")
    sorted_by_mean_fitness = sorted(results, key=lambda i: i.aggregated_score, reverse=True)
    for individual in sorted_by_mean_fitness[:12]:
        print(individual)