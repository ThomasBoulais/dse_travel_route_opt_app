import numpy as np

from testing_grounds.evaluator import generate_itinerary, load_env

config_path = "C:/Users/thoma/Documents/python_projects/dse_travel_route_opt_app/travel_route_optimization/model_training/config.yaml"


result = generate_itinerary(
    model_name="tdtoptw_dqn",
    start_poi=229,
    start_day=0,
    num_days=1,
    config_path = "C:/Users/thoma/Documents/python_projects/dse_travel_route_opt_app/travel_route_optimization/model_training/config.yaml"
)

print(result["score"])
print(result["validation"])

# env = load_env(config_path)

# N = env.travel_time.shape[0]
# print("POIs:", N)

# print("Any INF in travel_time:", np.isinf(env.travel_time).any())

# isolated = np.where(np.all(np.isinf(env.travel_time), axis=1))[0]
# print("Isolated POIs:", isolated)
# print("Count:", len(isolated))

