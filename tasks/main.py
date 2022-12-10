import glob
import json
import os
import random
import time

import joblib
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from loguru import logger

from src import PROJECT_ROOT
from src.algorithms import create_algorithm_for_optuna, alg_list
from src.problems import create_problem, prob_list


def objective_degree(prob_cls, alg_cls, timeout):
    def objective(trial):
        alg = create_algorithm_for_optuna(alg_cls, trial)
        score, _, _ = run(prob_cls, alg, 999999, timeout)
        return score

    return objective


def run(prob_cls, alg, epochs, timeout):
    prob = create_problem(prob_cls)
    random.seed(1)
    prob.init()
    random.seed()

    alg.init(prob)

    # loop
    t0 = time.time()
    for i in range(epochs):
        if time.time() - t0 > timeout:
            break
        alg.step()

    return alg.getMaxScore(), i, alg.count


def multi_run(params):
    prob_cls = params[0]
    alg_cls = params[1]
    best_params = params[2]

    alg = alg_cls(**best_params)
    max_data, step_count, alg_count = run(prob_cls, alg, 999999, timeout=2)

    return {
        "score": max_data,
        "step_count": step_count,
        "alg_count": alg_count,
    }


def main():
    out_dir = PROJECT_ROOT / "data" / "output"
    out_dir.mkdir(exist_ok=True)

    probs = prob_list
    algs = alg_list

    # optuna.logging.disable_default_handler()
    for prob_cls in probs:
        for alg_cls in algs:
            path = out_dir / "{}_{}.json".format(prob_cls.__name__, alg_cls.__name__)

            logger.info(f"=== start: {prob_cls.__name__} {alg_cls.__name__} ===")
            t0 = time.time()

            study = optuna.create_study(
                storage=f"sqlite:///{out_dir / 'optuna_study.db'}",
                direction="maximize",
            )
            study.optimize(objective_degree(prob_cls, alg_cls, timeout=2), n_trials=100, n_jobs=4)
            # logger.info(study.best_params)
            # logger.info(study.best_value)

            # ---
            params = (prob_cls, alg_cls, study.best_params)
            data = joblib.Parallel(n_jobs=4, verbose=0)([joblib.delayed(multi_run)(params) for _ in range(100)])

            result = {
                "time": time.time() - t0,
                "prob": prob_cls.__name__,
                "alg": alg_cls.__name__,
                "best_params": study.best_params,
                "best_value": study.best_value,
                "data": data
            }

            # output
            with open(path, "w") as f:
                json.dump(result, f, indent=4)


def view():
    out_dir = PROJECT_ROOT / "data" / "output"

    data = []
    for fn in out_dir.glob("*.json"):
        logger.info(fn)
        with open(fn, "r") as f:
            data.append(json.load(f))

    # データ加工
    probs = {}
    algs = {}
    for d in data:
        df = pd.DataFrame(d["data"])
        d["min"] = df["score"].min()
        d["mean"] = df["score"].mean()
        d["max"] = df["score"].max()

        # function_XinSheYangは値がすごく小さいので桁数で比較
        if d["prob"] == "function_XinSheYang":
            d["min"] = -int(str(d["min"]).split("e")[1])
            d["mean"] = -int(str(d["mean"]).split("e")[1])
            d["max"] = -int(str(d["max"]).split("e")[1])

        if d["prob"] not in probs:
            probs[d["prob"]] = []
        probs[d["prob"]].append(d)

        if d["alg"] not in algs:
            algs[d["alg"]] = []
        algs[d["alg"]].append({
            "prob": d["prob"],
            "params": d["best_params"],
        })

    # グラフ化
    for name, prob in probs.items():
        df = pd.DataFrame(prob)

        plt.style.use('ggplot')
        df.plot.bar(x="alg", y=["min", "mean", "max"], title="{}".format(name), xlabel="")

        path = out_dir / f"{name}.png"
        logger.info(path)
        plt.tight_layout()
        plt.savefig(path)
        # plt.show()

    # 表示
    for name, alg2 in algs.items():
        logger.info(name)

        # データ整形
        d = []
        for a in alg2:
            logger.info("{} {}".format(a["prob"], a["params"]))
            d2 = {}
            d2["name"] = a["prob"]
            for k, v in a["params"].items():
                d2[k] = v
            d.append(d2)

        df = pd.DataFrame(d)
        logger.info(df)


if __name__ == "__main__":
    main()
    view()
