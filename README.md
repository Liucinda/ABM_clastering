# Agent-based model of financial market

Current ABM of financial market was created by using Mesa framework (https://mesa.readthedocs.io/en/latest/overview.html) 
and contain four types of agents: noise traders, high-frequency traders, long-term strategy traders and market makers.
It consists of files model.py, agents.py, scheduler.py and run.py. The last one can be used for launching the model with different parameters and saving the results as CSV file.

ABM_classification.ipynb was used to determine the most powerful significant parameters of the model with CatBoost model.

ABM_clustering.ipynb consists of different clustering approach.
