from src.problems.EightQueen import EightQueen
from src.problems.LifeGame import LifeGame
from src.problems.OneMax import OneMax
from src.problems.TSP import TSP
from src.problems.function_Ackley import function_Ackley
from src.problems.function_Griewank import function_Griewank
from src.problems.function_Michalewicz import function_Michalewicz
from src.problems.function_Rastrigin import function_Rastrigin
from src.problems.function_Schwefel import function_Schwefel
from src.problems.function_StyblinskiTang import function_StyblinskiTang
from src.problems.function_XinSheYang import function_XinSheYang
from src.problems.g2048 import g2048

prob_list = [
    # OneMax,
    # EightQueen,
    # TSP,
    # LifeGame,
    # g2048,
    function_Ackley,
    # function_Griewank,
    # function_Michalewicz,
    # function_Rastrigin,
    # function_Schwefel,
    # function_StyblinskiTang,
    # function_XinSheYang,
]


def create_problem(prob_cls):
    if prob_cls.__name__ == OneMax.__name__:
        return OneMax(10000)
    if prob_cls.__name__ == EightQueen.__name__:
        return EightQueen(20)
    if prob_cls.__name__ == TSP.__name__:
        return TSP(80)
    if prob_cls.__name__ == LifeGame.__name__:
        return LifeGame(20, max_turn=5)
    if prob_cls.__name__ == g2048.__name__:
        return g2048(max_turn=350)
    if prob_cls.__name__ == function_Ackley.__name__:
        return function_Ackley(50)
    if prob_cls.__name__ == function_Griewank.__name__:
        return function_Griewank(100)
    if prob_cls.__name__ == function_Michalewicz.__name__:
        return function_Michalewicz(70)
    if prob_cls.__name__ == function_Rastrigin.__name__:
        return function_Rastrigin(70)
    if prob_cls.__name__ == function_Schwefel.__name__:
        return function_Schwefel(50)
    if prob_cls.__name__ == function_StyblinskiTang.__name__:
        return function_StyblinskiTang(80)
    if prob_cls.__name__ == function_XinSheYang.__name__:
        return function_XinSheYang(200)
    raise ValueError()
