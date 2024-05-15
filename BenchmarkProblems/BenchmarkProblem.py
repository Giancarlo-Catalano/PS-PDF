from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.SearchSpace import SearchSpace
from utils import announce


class BenchmarkProblem:
    """ This is an interface for toy problems, which makes my code much prettier"""
    """ The main components of this class are:
     -  a search space: the combinatorial search space
     -  fitness_function: the fitness function to be MAXIMISED
     -  get_targets: the ideal Core catalog
     -  repr_pr: a way to represent the Core which makes sense for the problem (ie checkerboard would use a grid)
     
     A useful related class to look at is UnitaryProblem
     """
    search_space: SearchSpace

    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def __repr__(self):
        raise Exception("An implementation of BenchmarkProblem does not implement __repr__")

    def get_reference_population(self, sample_size: int) -> PRef:
        with announce(f"Generating a uniform pRef with {sample_size} samples"):
            result = PRef.sample_from_search_space(search_space=self.search_space,
                                                 fitness_function=self.fitness_function,
                                                 amount_of_samples=sample_size)
        return result

    def repr_full_solution(self, fs: FullSolution) -> str:
        """default implementation"""
        return f"{fs}"

    def repr_ps(self, ps: PS) -> str:
        """default implementation"""
        return f"{ps}"

    def repr_fs(self, full_solution: FullSolution) -> str:
        return self.repr_ps(PS.from_FS(full_solution))

    def fitness_function(self, fs: FullSolution) -> float:
        raise Exception("An implementation of BenchmarkProblem does not implement the fitness function!!!")

    def get_targets(self) -> list[PS]:
        raise Exception("An implementation of BenchmarkProblem does not implement get_targets")

    def get_global_optima_fitness(self) -> float:
        raise Exception("An implementation of BenchmarkProblem does not implement get_global_optima_fitness")


    def ps_to_properties(self, ps: PS) -> dict:
        raise NotImplemented(f"The class {self.__repr__()} does not implement .ps_to_properties")

    def repr_property(self, property_name:str, property_value:str, property_rank_range:str):
        rank_lower_bound, rank_upper_bound = property_rank_range
        start = f"{property_name} = {property_value:.2f} is "


        if rank_upper_bound == 0:
            end = "the lowest observed"
        elif rank_lower_bound == 1.0:
            end = "the highest observed"
        elif rank_lower_bound > 0.5:
            end = f"relatively high (top {int((1-rank_upper_bound)*100)}% ~ {int((1-rank_lower_bound)*100)}%)"
        else:
            end = f"relatively low (bottom {int(rank_lower_bound*100)}% ~ {int(rank_upper_bound*100)}%)"

        return start + end


    def repr_extra_ps_info(self, ps: PS):
        return f"PS has {ps.fixed_count()} fixed variables"