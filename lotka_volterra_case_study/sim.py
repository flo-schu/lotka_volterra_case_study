from copy import deepcopy
import numpy as np
import xarray as xr
from pydantic import BaseModel, Field
from pymob.simulation import SimulationBase
from pymob.solvers.diffrax import JaxSolver
from pymob.sim.config import DataVariable, Param, PymobModel, OptionListStr, register_case_study_config
from lotka_volterra_case_study.mod import lotka_volterra, solve, solve_jax
from lotka_volterra_case_study.plot import plot_trajectory
from lotka_volterra_case_study import prob

from pymob.sim.config import Config

class LotkaVolterraSettings(PymobModel):
    """Options specific to the Lotka-Volterra case study."""
    test_setting_1: bool = True
    test_setting_2: str = "I am Lotka"
    test_setting_3: float = 1.0
    test_setting_4: OptionListStr = ["a", "b"]

# Register the model under the directory name (must match ``case_study.name``)
register_case_study_config("lotka_volterra", LotkaVolterraSettings)

DEFAULT_CONFIG = Config()

class Simulation(SimulationBase):
    solver = solve_jax
    model = lotka_volterra
    _prob = prob

    def initialize(self, input):
        self.model_parameters["parameters"] =  dict(
            alpha = 0.5,  # Prey birth rate
            beta = 0.02,  # Rate at which predators decrease prey population
            gamma = 0.3,  # Predator death rate
            delta = 0.01,  # Predator reproduction rate
        )
        
        self.observations = xr.load_dataset(input[1])
        self.observations["wolves"] = self.observations["wolves"] + 1e-8
        self.observations["rabbits"] = self.observations["rabbits"] + 1e-8

        self.model_parameters["y0"] = self.parse_input("y0", drop_dims=["time"])
        
    @staticmethod
    def parameterize(free_parameters: dict, model_parameters):
        """Should avoid using input arg but instead take a single dictionary as 
        an input. This also then provides an harmonized IO between model and 
        parameters, which in addition is serializable to json.

        model parameters is provided by `functools.partial` on model initialization
        """
        # Initial conditions and parameters
        y0 = model_parameters["y0"]
        parameters = deepcopy(model_parameters["parameters"])
        # mapping of parameters *theta* to the model parameters accessed by
        # the solver. This task is necessary for any model 
        parameters.update(free_parameters)

        return dict(y0=y0, parameters=parameters)

    def set_coordinates(self, input):
        if hasattr(self, "observations"):
            return self.observations.time.values
        
        # Time settings
        t_start = 0
        t_end = 200
        t_step = 0.1
        time = np.arange(t_start, t_end, t_step)

        return time
    
    def plot(self, results):
        fig = plot_trajectory(results)
        fig.savefig(f"{self.output_path}/trajectory.png")

class Simulation_v2(Simulation):
    solver = JaxSolver
    
    def initialize(self, input):
        self.observations = xr.load_dataset(input[0])

        y0 = self.parse_input("y0", drop_dims=["time"])
        self.model_parameters["y0"] = y0

        self.model_parameters["parameters"] = self.config.model_parameters.value_dict


class HierarchicalSimulation(Simulation_v2):
    def initialize(self, input):
        self.config.data_structure.indices = ["rabbit_species", "experiment"]
        self.observations = xr.load_dataset(input[0])
        
        y0 = self.parse_input("y0", drop_dims=["time"])
        self.model_parameters["y0"] = y0

        self.model_parameters["parameters"] = self.config.model_parameters.value_dict

    def initialize_from_script(self):
        # self.config.case_study.scenario = "test_hierarchical"

        self.config.simulation.batch_dimension = "id"

        # prey birth rate
        self.config.model_parameters.alpha_species = Param(
            value=0.5, free=True, hyper=True,
            dims=('rabbit_species','experiment'),
            # take good care to specify hyperpriors correctly. 
            # Dimensions are broadcasted following the normal rules of 
            # numpy. The below means, in dimension one, we have two different
            # assumptions 1, and 3. Dimension one is the dimension of the rabbit species.
            # The specification scale=[1,3] would be understood as [[1,3]] and
            # be understood as the experiment dimension. Ideally, the dimensionality
            # is so low that you can be specific about the priors. I.e.:
            # scale = [[1,1,1],[3,3,3]]. This of course expects you know about
            # the dimensionality of the prior (i.e. the unique coordinates of the dimensions)
            prior="halfnorm(scale=[[1],[3]])" # type: ignore
        )
        # prey birth rate
        self.config.model_parameters.alpha = Param(
            value=0.5, free=True, hyper=False,
            dims=("id",),
            prior="lognorm(s=0.1,scale=alpha_species[rabbit_species_index, experiment_index])" # type: ignore
        )
        # Rate at which predators decrease prey population.
        self.config.model_parameters.beta = Param(value=0.02, free=True, prior="lognorm(s=0.1,scale=0.02)") # type:ignore
        # Predator death rate
        self.config.model_parameters.gamma = Param(value=0.3, free=False)
        # Predator reproduction rate
        self.config.model_parameters.delta = Param(value=0.01, free=False)

        self.config.data_structure.rabbits = DataVariable(
            dimensions=["id", "time"], observed=False, dimensions_evaluator=["id", "time"]
        )
        self.config.data_structure.wolves = DataVariable(
            dimensions=["id", "time"], observed=False, dimensions_evaluator=["id", "time"]
        )

        self.define_observations_replicated_multi_experiment(n=12)
        self.coordinates["time"] = np.arange(0, 200, 0.1)

        y0 = self.parse_input("y0", drop_dims=["time"])
        self.model_parameters["y0"] = y0
        
        self.config.error_model.rabbits = "lognorm(scale=rabbits+EPS,s=0.1)"
        self.config.error_model.wolves = "lognorm(scale=wolves+EPS,s=0.1)"
        
    
    def load_presimulated_observations(self):
        # set indices and load data
        self.config.data_structure.wolves.observed = False
        self.config.data_structure.rabbits.observed = False
        
        self.define_observations_replicated_multi_experiment(n=120) # type: ignore
        self.observations = xr.load_dataset(f"{self.data_path}/simulated_data_hierarchical_species_year.nc")
        
        self.config.data_structure.wolves.observed = True
        self.config.data_structure.rabbits.observed = True

        y0 = self.parse_input("y0", drop_dims=["time"])
        self.model_parameters["y0"] = y0
        self.model_parameters["parameters"] = self.config.model_parameters.value_dict

    def define_observations_replicated_multi_experiment(self, n):
        """This sets up the data structure of a Lotka-Volterra case study with 
        an example of observations that were taken from two rabbit populations 
        (species: Cottontail and Jackrabbit), observed in 2010, 2011 and 2012
        in different valleys, which define the number of in-treatment replicates.

        Parameters
        ----------
        n : int
            total number of observations. This must be a multiple of the 
            factor combinations (here: 2x3 = 6) so anythin from 6, 12, 18, ...
        """

        self.coordinates["id"] = np.arange(n)

        years = ["2010","2011","2012"]
        species = ["Cottontail", "Jackrabbit"]
        replicates_per_year = int(n/len(years))
        replicates_per_species = int(replicates_per_year/len(species))

        self.config.data_structure.indices = ["rabbit_species", "experiment"]

        self.observations = xr.Dataset().assign_coords({
            "rabbit_species": xr.DataArray(
                list(np.repeat(species, replicates_per_species)) * len(years), 
                dims=("id"), coords={"id": np.arange(n)}
            ),
            "experiment": xr.DataArray(
                # repeats the array n/len(arr) each -> [2010,2010,2010,2010,2011,...] 
                np.repeat(years, replicates_per_year), 
                dims=("id"), coords={"id": np.arange(n)}
            )
        })

        # make up some initial population estimates        
        rng = np.random.default_rng(1)
        t0_wolves = list(rng.integers(2, 15, n))
        t0_rabbits = list(rng.integers(35, 70, n))
        self.config.simulation.y0 = [
            f"rabbits=Array({str(t0_rabbits).replace(' ','')})",
            f"wolves=Array({str(t0_wolves).replace(' ','')})"
        ]


    @staticmethod
    def index_coordinates(array):
        # Create a dictionary to map unique coordinates to indices
        # using np.unique thereby retains the order of the elements in
        # the order of their furst occurence
        unique_coordinates = np.unique(array)
        string_to_index = {
            coord: index for index, coord 
            in enumerate(unique_coordinates)
        }

        # Convert the original array to a list of indices
        index_list = [string_to_index[string] for string in array]
        return index_list
    
    def plot(self):
        pass