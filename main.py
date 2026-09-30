"""Лабораторная работа 1: ГА для функции Цин, вариант 12."""

import csv
import json
import math
import random
import statistics
from pathlib import Path


LOWER, UPPER = -500.0, 500.0
HERE = Path(__file__).parent


def variant(number, group, year):
    group_index = {507: 0, 517: 1, 527: 2}[group]
    return 1 + (17 * number + 7 * group_index + year) % 20


def objective(x):
    """f(x) = sum((x_i**2 - i)**2), i = 1,...,d."""
    return sum((value * value - i) ** 2 for i, value in enumerate(x, 1))


def random_individual(rng, dimension):
    return [rng.uniform(LOWER, UPPER) for _ in range(dimension)]


def tournament(population, size, rng):
    """В популяции хранятся пары (вектор, значение функции)."""
    return min(rng.sample(population, size), key=lambda item: item[1])[0]


def make_child(first, second, crossover_probability, mutation_probability,
               sigma, rng):
    if rng.random() < crossover_probability:
        # По каждому гену выбираем значение одного из двух родителей.
        child = [a if rng.random() < 0.5 else b
                 for a, b in zip(first, second)]
    else:
        child = first.copy()

    for i in range(len(child)):
        if rng.random() < mutation_probability:
            child[i] += rng.gauss(0.0, sigma)
            # Возврат в допустимую область.
            child[i] = max(LOWER, min(UPPER, child[i]))
    return child


def run_ga(config, mutation_probability, seed, dimension):
    rng = random.Random(seed)
    size = config["population_size"]
    generations = config["generations"]
    population = []
    for _ in range(size):
        x = random_individual(rng, dimension)
        population.append((x, objective(x)))

    best_x, best_f = min(population, key=lambda item: item[1])
    best_x = best_x.copy()
    history = [best_f]

    for _ in range(1, generations):
        elite = min(population, key=lambda item: item[1])
        next_population = [(elite[0].copy(), elite[1])]
        while len(next_population) < size:
            first = tournament(population, config["tournament_size"], rng)
            second = tournament(population, config["tournament_size"], rng)
            child = make_child(first, second,
                               config["crossover_probability"],
                               mutation_probability, config["mutation_sigma"], rng)
            next_population.append((child, objective(child)))

        population = next_population
        x, value = min(population, key=lambda item: item[1])
        if value < best_f:
            best_x, best_f = x.copy(), value
        history.append(best_f)

    return best_x, best_f, history


def run_random_search(config, seed, dimension):
    rng = random.Random(seed)
    best_x = None
    best_f = math.inf
    history = []
    for generation in range(config["generations"]):
        # ГА тратит size вычислений при инициализации, затем size-1:
        # элитная особь переносится без повторного вычисления.
        batch = config["population_size"] if generation == 0 else config["population_size"] - 1
        for _ in range(batch):
            x = random_individual(rng, dimension)
            value = objective(x)
            if value < best_f:
                best_x, best_f = x, value
        history.append(best_f)
    return best_x, best_f, history


def describe(values):
    return {
        "best": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
        "worst": max(values),
    }


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_history(path, trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(trajectories), figsize=(15, 4.5),
                             sharey=True)
    for ax, (name, histories) in zip(axes, trajectories.items()):
        generations = range(len(histories[0]))
        minimum = [min(h[g] for h in histories) for g in generations]
        average = [statistics.mean(h[g] for h in histories)
                   for g in generations]
        maximum = [max(h[g] for h in histories) for g in generations]
        ax.plot(generations, minimum, label="minimum")
        ax.plot(generations, average, label="mean")
        ax.plot(generations, maximum, label="maximum")
        ax.set_title(name)
        ax.set_xlabel("Generation / checkpoint")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    axes[0].set_ylabel("Best f(x) so far")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    v = variant(config["student_number"], config["group"], config["year"])
    if v != 12:
        raise ValueError(f"Программа рассчитана на вариант 12, получен {v}")
    dimension = 5 + v % 6
    if config["population_size"] < 30 or config["generations"] < 3:
        raise ValueError("Нужны не менее 30 особей и 3 поколений")

    output_dir = HERE / "results"
    output_dir.mkdir(exist_ok=True)
    seeds = [config["first_seed"] + i for i in range(config["runs"])]
    trajectories = {}
    result_rows = []
    for probability in config["mutation_probabilities"]:
        name = f"GA p={probability:g}"
        trajectories[name] = []
        for seed in seeds:
            x, value, history = run_ga(config, probability, seed, dimension)
            result_rows.append({"method": name, "seed": seed,
                                "best_f": value, "best_x": json.dumps(x)})
            trajectories[name].append(history)

    trajectories["Random search"] = []
    for seed in seeds:
        x, value, history = run_random_search(config, seed, dimension)
        result_rows.append({"method": "Random search", "seed": seed,
                            "best_f": value, "best_x": json.dumps(x)})
        trajectories["Random search"].append(history)

    history_rows = []
    for name, histories in trajectories.items():
        for generation in range(config["generations"]):
            values = [history[generation] for history in histories]
            evaluations = (config["population_size"] + generation *
                           (config["population_size"] - 1))
            history_rows.append({"method": name, "generation": generation,
                                 "evaluations": evaluations,
                                 "minimum": min(values),
                                 "mean": statistics.mean(values),
                                 "maximum": max(values)})

    summary = {name: describe([history[-1] for history in histories])
               for name, histories in trajectories.items()}
    write_csv(output_dir / "runs.csv",
              ["method", "seed", "best_f", "best_x"], result_rows)
    write_csv(output_dir / "history.csv",
              ["method", "generation", "evaluations", "minimum", "mean", "maximum"],
              history_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_history(output_dir / "convergence.png", trajectories)

    print(f"N={config['student_number']}, group={config['group']}, V={v}, d={dimension}")
    print(f"Evaluations per run: {config['population_size'] + (config['generations'] - 1) * (config['population_size'] - 1)}")
    for name, values in summary.items():
        print(f"{name}: best={values['best']:.6g}, mean={values['mean']:.6g}, "
              f"median={values['median']:.6g}, std={values['std']:.6g}, "
              f"worst={values['worst']:.6g}")


if __name__ == "__main__":
    main()
