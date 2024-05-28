#!/usr/bin/env python3
import logging
import os
import sys
import traceback
import warnings

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.EfficientBTProblem.ManuallyConstructedBTInstances import get_bad_week_instance
from BenchmarkProblems.GraphColouring import GraphColouring
from Core import TerminationCriteria
from Core.Explainer import Explainer
from Explanation.Detector import Detector
from Explanation.HyperparameterEvaluator import HyperparameterEvaluator
from FSStochasticSearch.Operators import SinglePointFSMutation
from FSStochasticSearch.SA import SA
from PSMiners.DEAP.DEAPPSMiner import DEAPPSMiner
from PSMiners.Mining import get_history_pRef
from utils import announce, indent


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def show_overall_system(benchmark_problem: BenchmarkProblem):
    """
    This function gives an overview of the system:
        1. Generate a reference population (a PRef)
        2. Generate a Core Catalog using the Core Miner
        3. Sample new solutions from the catalog using Pick & Merge
        4. Explain those new solutions using the catalog

    :param benchmark_problem: a benchmark problem, find more in the BenchmarkProblems directory
    :return: Nothing! Just printing
    """

    print(f"The problem is {benchmark_problem}")

    # 1. Generating the reference population
    pRef_size = 10000
    with announce("Generating Reference Population"):
        pRef = get_history_pRef(benchmark_problem, sample_size=pRef_size, which_algorithm="SA")
    pRef.describe_self()

    # 2. Obtaining the Core catalog
    ps_miner = DEAPPSMiner.with_default_settings(pRef)
    ps_evaluation_budget = 10000
    termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_evaluation_budget)

    with announce("Running the PS Miner"):
        ps_miner.run(termination_criterion, verbose=True)

    ps_catalog = ps_miner.get_results(None)
    ps_catalog = list(set(ps_catalog))
    ps_catalog = [item for item in ps_catalog if not item.is_empty()]

    print("The catalog consists of:")
    for item in ps_catalog:
        print("\n")
        print(indent(f"{benchmark_problem.repr_ps(item)}"))

    # 3. Sampling new solutions
    print("\nFrom the catalog we can sample new solutions")
    new_solutions_to_produce = 12
    sampler = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   cooling_coefficient=0.9995)

    solutions = pRef.get_evaluated_FSs()
    solutions = list(set(solutions))
    solutions.sort(reverse=True)


    for index, sample in enumerate(solutions[:6]):
        print(f"[{index}]")
        print(indent(indent(f"{benchmark_problem.repr_fs(sample.full_solution)}, has fitness {sample.fitness:.2f}")))

    # 4. Explainability, at least locally.
    explainer = Explainer(benchmark_problem, ps_catalog, pRef)
    explainer.explanation_loop(solutions)

    print("And that concludes the showcase")

def get_bt_explainer() -> Detector:
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\MartinBT"
    problem = EfficientBTProblem.from_default_files()
    return Detector.from_folder(problem=problem,
                                  folder=experimental_directory,
                                  speciality_threshold=0.2,
                                  verbose=True)

def get_gc_explainer():
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\GC\Dummy"
    problem_file = os.path.join(experimental_directory, "fibre.json")
    problem = GraphColouring.from_file(problem_file)#GraphColouring.random(amount_of_colours=3, amount_of_nodes=5, chance_of_connection=0.40)
    problem.view()
    return Detector.from_folder(folder = experimental_directory,
                                  problem = problem,
                                  speciality_threshold=0.20,
                                  verbose=True)


def get_manual_bt_explainer() -> Detector:
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\TwoTeam"
    amount_of_skills = 12
    problem = get_bad_week_instance(amount_of_skills, workers_per_skill=4)
    #problem = get_start_and_end_instance(amount_of_skills)
    #problem = get_toestepping_instance(amount_of_skills=3)
    #problem = get_unfairness_instance(amount_of_skills=3)
    return Detector.from_folder(problem=problem,
                                folder=experimental_directory,
                                speciality_threshold=0.2,
                                verbose=True)


def get_problem_explainer() -> Detector:
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\GC\Dummy"
    gc_problem_file = os.path.join(experimental_directory, "islets.json")
    gc_problem = GraphColouring.from_file(gc_problem_file)
    gc_problem.view()
    problem = EfficientBTProblem.from_Graph_Colouring(gc_problem)
    return Detector.from_folder(problem=problem,
                                folder=experimental_directory,
                                speciality_threshold=0.2,
                                verbose=True)

def explanation():
    detector = get_problem_explainer()
    # detector.generate_files_with_default_settings(30000, 30000)
    detector.explanation_loop(amount_of_fs_to_propose=2, ps_show_limit=5, show_debug_info=True)

    #detector.explanation_loop(amount_of_fs_to_propose=3, show_debug_info=False, show_global_properties = False)
    #get_bt_explainer().get_variables_properties_table()


# def evaluation():
#     gc_problem = GraphColouring.make_insular_instance(amount_of_islands=4)
#     gc_problem.view()
#     bt_problem = EfficientBTProblem.from_Graph_Colouring(gc_problem)
#     experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\GC\Dummy"
#     detector = Detector.from_folder(problem=bt_problem,
#                                      folder=experimental_directory,
#                                      speciality_threshold=0.2,
#                                      verbose=True)
#     detector.generate_files_with_default_settings(30000, 30000)
#     mined_pss = detector.pss
#     count = count_found_targets(gc_problem, mined_pss)
#     print(f"Out of the {len(mined_pss)} mined pss, {count} were the true clique targets")


def grid_search():
    # hype = HyperparameterEvaluator(algorithms_to_test=["NSGAII", "NSGAIII", "SPEA2"],
    #                                island_amounts_to_test=[3, 6, 12],
    #                                pRef_sizes_to_test=[1000, 5000, 10000, 30000],
    #                                pRef_origin_methods = ["uniform", "SA"]
    #                                population_sizes_to_test=[50, 100, 500],
    #                                ps_budget=30000,
    #                                ps_budgets_per_run_to_test=[1000, 5000, 10000])
    hype = HyperparameterEvaluator(algorithms_to_test=["NSGAII", "NSGAIII", "MOEAD"],
                                   island_amounts_to_test=[6],
                                   pRef_sizes_to_test=[5000],
                                   population_sizes_to_test=[100],
                                   pRef_origin_methods = ["uniform", "SA", "GA"],
                                   ps_budget=3000,
                                   custom_crowding_operators_to_test = [True, False],
                                   ps_budgets_per_run_to_test=[1000])
    hype.get_data()






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.seterr(invalid="raise")
    warnings.showwarning = warn_with_traceback
    grid_search()
    # comment, again, again, again, again, again, again
    #test_linearity_between_gc_and_bt()






