import pytest
from lotka_volterra_case_study.sim import HierarchicalSimulation, Simulation_v2


def construct_sim(scenario, simulation_class):
    """Helper function to construct simulations for debugging"""
    sim = simulation_class(f"scenarios/{scenario}/settings.cfg")

    # this sets a different output directory
    sim.config.case_study.scenario = "testing"
    sim.setup()
    return sim


# List test scenarios and simulations
@pytest.fixture(scope="session", params=[
    (HierarchicalSimulation, "lotka_volterra_hierarchical_final"),
    (Simulation_v2, "test_scenario_v2"),
])
def sim_and_scenario(request):
    return request.param


# Derive simulations for testing from fixtures
@pytest.fixture(scope="session")
def sim(sim_and_scenario):
    simulation_class, scenario = sim_and_scenario
    yield construct_sim(scenario, simulation_class)


# run tests with the Simulation fixtures
def test_setup(sim):
    """Tests the construction method"""
    assert True


def test_simulation(sim):
    """Tests if a forward simulation pass can be computed"""
    sim.dispatch_constructor()
    evaluator = sim.dispatch()
    evaluator()
    evaluator.results

    assert True
            

@pytest.mark.slow
@pytest.mark.parametrize("backend", ["numpyro"])
def test_inference(sim, backend):
    """Tests if prior predictions can be computed for arbitrary backends"""
    sim.dispatch_constructor()
    sim.set_inferer(backend)

    sim.config.inference.n_predictions = 2
    sim.prior_predictive_checks()
    
    sim.config.inference_numpyro.kernel = "svi"
    sim.config.inference_numpyro.svi_iterations = 1_000
    sim.config.inference_numpyro.svi_learning_rate = 0.05
    sim.config.inference_numpyro.draws = 100
    sim.config.inference.n_predictions = 100

    sim.inferer.run()

    sim.inferer.idata

    sim.posterior_predictive_checks()

if __name__ == "__main__":
    # test_inference(sim=construct_sim("test_scenario_v2", Simulation_v2), backend="numpyro")
    test_inference(sim=construct_sim("lotka_volterra_hierarchical_hyperpriors", HierarchicalSimulation), backend="numpyro")
