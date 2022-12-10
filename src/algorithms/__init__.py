from src.algorithms.ABC import ABC
from src.algorithms.Bat import Bat
from src.algorithms.Cuckoo import Cuckoo
from src.algorithms.Cuckoo_greedy import Cuckoo_greedy
from src.algorithms.DE import DE
from src.algorithms.Firefly import Firefly
from src.algorithms.GA import GA
from src.algorithms.GA_BLXa import GA_BLXa
from src.algorithms.Harmony import Harmony
from src.algorithms.PSO import PSO
from src.algorithms.PfGA import PfGA
from src.algorithms.Tabu import Tabu
from src.algorithms.WOA import WOA

alg_list = [
    # GA,
    # PfGA,
    # ABC,
    # Bat,
    # Cuckoo,
    # Cuckoo_greedy,
    # DE,
    # Firefly,
    # Harmony,
    # PSO,
    # WOA,
    # GA_BLXa,
    Tabu,
]


def create_algorithm_for_optuna(alg_cls, trial):
    if alg_cls.__name__ == GA.__name__:
        alg = GA(
            trial.suggest_int('individual_max', 2, 50),
            trial.suggest_categorical('save_elite', [False, True]),
            trial.suggest_categorical('select_method', ["ranking", "roulette"]),
            trial.suggest_float('mutation', 0.0, 1.0),
        )
    elif alg_cls.__name__ == PfGA.__name__:
        alg = PfGA(
            trial.suggest_float('mutation', 0.0, 1.0),
        )
    elif alg_cls.__name__ == ABC.__name__:
        alg = ABC(
            trial.suggest_int('harvest_bee', 2, 50),
            trial.suggest_int('follow_bee', 1, 100),
            trial.suggest_int('visit_max', 1, 100),
        )
    elif alg_cls.__name__ == Bat.__name__:
        alg = Bat(
            trial.suggest_int('bat_max', 2, 50),
            trial.suggest_float('frequency_min', 0, 0),
            trial.suggest_float('frequency_max', 0, 1),
            trial.suggest_float('good_bat_rate', 0, 1),
            trial.suggest_float('volume_init', 0, 2),
            trial.suggest_float('volume_update_rate', 0, 1),
            trial.suggest_float('pulse_convergence_value', 0, 1),
            trial.suggest_float('pulse_convergence_speed', 0, 1),
        )
    elif alg_cls.__name__ == Cuckoo.__name__:
        alg = Cuckoo(
            trial.suggest_int('nest_max', 2, 50),
            trial.suggest_float('scaling_rate', 0, 2.0),
            trial.suggest_float('levy_rate', 0, 1.0),
            trial.suggest_float('bad_nest_rate', 0, 1.0),
        )
    elif alg_cls.__name__ == Cuckoo_greedy.__name__:
        alg = Cuckoo_greedy(
            trial.suggest_int('nest_max', 2, 50),
            trial.suggest_float('epsilon', 0, 1.0),
            trial.suggest_float('bad_nest_rate', 0, 1.0),
        )
    elif alg_cls.__name__ == DE.__name__:
        alg = DE(
            trial.suggest_int('agent_max', 4, 50),
            trial.suggest_float('crossover_rate', 0, 1.0),
            trial.suggest_float('scaling', 0, 2.0),
        )
    elif alg_cls.__name__ == Firefly.__name__:
        alg = Firefly(
            trial.suggest_int('firefly_max', 2, 50),
            trial.suggest_float('attracting_degree', 0.0, 1.0),
            trial.suggest_float('absorb', 0.0, 50.0),
            trial.suggest_float('alpha', 0.0, 1.0),
            trial.suggest_categorical('is_normalization', [False, True]),
        )
    elif alg_cls.__name__ == Harmony.__name__:
        alg = Harmony(
            trial.suggest_int('harmony_max', 2, 50),
            trial.suggest_float('bandwidth', 0.0, 1.0),
            trial.suggest_categorical('enable_bandwidth_rate', [False, True]),
            trial.suggest_float('select_rate', 0.0, 1.0),
            trial.suggest_float('change_rate', 0.0, 1.0),
        )
    elif alg_cls.__name__ == PSO.__name__:
        alg = PSO(
            trial.suggest_int('particle_max', 2, 50),
            trial.suggest_float('inertia', 0.0, 1.0),
            trial.suggest_float('global_acceleration', 0.0, 1.0),
            trial.suggest_float('personal_acceleration', 0.0, 1.0),
        )
    elif alg_cls.__name__ == WOA.__name__:
        alg = WOA(
            trial.suggest_int('whale_max', 2, 50),
            trial.suggest_float('a_decrease', 0.0, 0.1),
            trial.suggest_float('logarithmic_spiral', 0.0, 2.0),
        )
    elif alg_cls.__name__ == GA_BLXa.__name__:
        alg = GA_BLXa(
            trial.suggest_int('individual_max', 2, 50),
            trial.suggest_categorical('save_elite', [False, True]),
            trial.suggest_categorical('select_method', ["ranking", "roulette"]),
            trial.suggest_float('mutation', 0.0, 1.0),
            trial.suggest_float('blx_a', 0.0, 1.0),
        )
    elif alg_cls.__name__ == Tabu.__name__:
        alg = Tabu(
            trial.suggest_int('individual_max', 2, 50),
            trial.suggest_float('epsilon', 0.0, 1.0),
            trial.suggest_int('tabu_list_size', 1, 500),
            trial.suggest_float('tabu_range_rate', 0.0, 1.0),
        )
    else:
        raise ValueError
    return alg
