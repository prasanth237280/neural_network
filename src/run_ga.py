from deap import base, creator, tools, algorithms
import numpy as np
import random
from utils.preprocess import load_and_normalize
from train_nn import build_model

X, y, scaler = load_and_normalize("data/data.xlsx")
model = build_model()
model.fit(X[:300], y[:300], epochs=100, verbose=0)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 9)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    pred = model.predict(np.array(ind).reshape(1, -1))
    return (pred[0][0],)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=30)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=True)

best = tools.selBest(pop, 1)[0]
print("Best Input (Normalized):", best)
print("Predicted Minimum Output:", evaluate(best)[0])
print("Best Input (Original Scale):", scaler.inverse_transform([best]))
