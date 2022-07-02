# Agent-based model of financial market

Current ABM of financial market was created by using Mesa framework (https://mesa.readthedocs.io/en/latest/overview.html) 
and contain four types of agents: noise traders, high-frequency traders, long-term strategy traders and market makers.
All agents described in agents.py as well as it's "step" method for placing orders.
The stages of model operating are described in model.py.
The scheduler.py file contains functions which activate different type of agents.
The run.py can be used for launching the model with different parameters and saving the results as CSV file.

ABM_classification.ipynb was used to determine the most powerful significant parameters of the model with CatBoost model and a comparison with other types of models.

ABM_clustering.ipynb consists of different clustering approaches implemented on Bollinger bands width values of trading sessions with the artificial price shock.
