import numpyro
import jax.numpy as jnp
from numpyro.distributions import LogNormal, Normal

def dummy_preprocessing(obs, masks):
    return {"obs": obs, "masks": masks}

def parameter_only_model(solver, obs, masks, indices, make_predictions=False):
    numpyro.sample("alpha", fn=numpyro.distributions.LogNormal(jnp.log(0.5), 1)) # type: ignore
    numpyro.sample("beta", fn=numpyro.distributions.LogNormal(jnp.log(0.02), 1)) # type: ignore


def lotka_volterra(solver, obs, masks, indices, make_predictions=False):
    EPS = 1e-6
    alpha = numpyro.sample("alpha", fn=LogNormal(jnp.log(0.5), 0.1))
    beta = numpyro.sample("beta", fn=LogNormal(jnp.log(0.02), 0.1)) 
    
    # with numpyro.plate("replicate_id", size=len(indices["rabbit_species_index"])):

    theta = {"alpha": alpha, "beta": beta}
    y = solver(theta=theta)
    wolves = numpyro.deterministic("wolves", y["wolves"])
    rabbits = numpyro.deterministic("rabbits", y["rabbits"])

    with numpyro.plate("time", size=200):
        numpyro.sample(
            "wolves_obs", 
            LogNormal(jnp.log(wolves+EPS), 0.1).mask(masks["wolves"]),
            obs=obs["wolves"]
        )

        with numpyro.handlers.scale(scale=2):
            numpyro.sample(
                "rabbits_obs", 
                LogNormal(jnp.log(rabbits+EPS), 0.1).mask(masks["rabbits"]),
                obs=obs["rabbits"]
            )

def hierarchical_lotka_volterra(solver, obs, masks, indices, make_predictions=False):
    alpha_species = numpyro.sample(
        "alpha_species",
        numpyro.distributions.LogNormal(jnp.log(1), 1).expand((2, 3)), # type: ignore
    )
    
    beta = numpyro.sample("beta", fn=numpyro.distributions.LogNormal(jnp.log(0.02), 1)) # type: ignore

    # with numpyro.plate("replicate_id", size=len(indices["rabbit_species_index"])):

    alpha_species_indexed = alpha_species[(indices["rabbit_species_index"], indices["experiment_index"])] # type:ignore

    alpha = numpyro.sample(
        "alpha", 
        fn=numpyro.distributions.LogNormal(jnp.log(alpha_species_indexed), 1), # type: ignore
    ) 
    theta = {"alpha": alpha, "beta": beta}
    y = solver(theta=theta)
    wolves = numpyro.deterministic("wolves", y["wolves"])
    rabbits = numpyro.deterministic("rabbits", y["rabbits"])

    if make_predictions:
        obs = {k: None for k in obs.keys()}

    numpyro.sample(
        "wolves_obs", 
        numpyro.distributions.Normal(0, 1).mask(masks["wolves"]),
        obs=(obs["wolves"] - wolves)/jnp.sqrt(wolves)
    )


    numpyro.sample(
        "rabbits_obs", 
        numpyro.distributions.Normal(0, 1).mask(masks["rabbits"]),
        obs=(obs["rabbits"] - rabbits)/jnp.sqrt(rabbits)
    )