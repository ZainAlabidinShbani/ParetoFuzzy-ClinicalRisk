# %%
# IMPORTS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skfuzzy import control as ctrl
import skfuzzy as fuzz
import pickle
import random
import copy
from joblib import Parallel, delayed

# %%
# LOAD DATA
file_path = r"..\data\processed\healthcare_processed.csv"
fuzzy_data = pd.read_csv(file_path)
X_fuzzy = fuzzy_data[['age', 'bmi','gender','ever_married','work_type','smoking_status','avg_glucose_level','hypertension', 'heart_disease', 'Residence_type']]
y_fuzzy = fuzzy_data['stroke_risk']
print("X shape:", X_fuzzy.shape)
print("y shape:", y_fuzzy.shape)

# %%
# LOAD FUZZY SYSTEM SPEC
with open(r"..\\data\\processed\\fuzzy_system_spec.pkl", "rb") as f:
    spec = pickle.load(f)

# %%
# HELPER: CREATE VARIABLES AND MFS
def create_var(name, universe_min, universe_max, step=1):
    uni = np.arange(universe_min, universe_max+step, step)
    return ctrl.Antecedent(uni, name)

def add_mf(var_obj, mf_name, mf_spec):
    if mf_spec['type'] == 'trimf':
        var_obj[mf_name] = fuzz.trimf(var_obj.universe, mf_spec['params'])
    elif mf_spec['type'] in ['trap', 'trapmf']:
        var_obj[mf_name] = fuzz.trapmf(var_obj.universe, mf_spec['params'])
    elif mf_spec['type'] == 'singleton':
        p = mf_spec['params'][0]
        eps = 0.001 * (var_obj.universe.max() - var_obj.universe.min())
        var_obj[mf_name] = fuzz.trimf(var_obj.universe, [p-eps, p, p+eps])
    else:
        raise ValueError("Unknown MF type: "+mf_spec['type'])

# %%
# BUILD VARIABLES
vars_ = {}
for vname, (lo, hi) in spec['universes'].items():
    vars_[vname] = create_var(vname, lo, hi)
for vname, mfs in spec['mfs'].items():
    for mf_name, mf_spec in mfs.items():
        add_mf(vars_[vname], mf_name, mf_spec)

# %%
# RECREATE RULES
rules_list = []
for antecedent, consequent in spec['rules']:
    left = None
    for feat, mf_name in antecedent.items():
        cond = vars_[feat][mf_name]
        left = cond if left is None else left & cond
    if 'stroke_risk' not in vars_:
        out = ctrl.Consequent(np.arange(0,3,1), 'stroke_risk')
        out['low'] = fuzz.trimf(out.universe, [0,0,1])
        out['medium'] = fuzz.trimf(out.universe, [0,1,2])
        out['high'] = fuzz.trimf(out.universe, [1,2,2])
        vars_['stroke_risk'] = out
    right_feat, right_mf = list(consequent.items())[0]
    right = vars_['stroke_risk'][right_mf]
    rules_list.append(ctrl.Rule(left, right))

stroke_ctrl = ctrl.ControlSystem(rules_list)
stroke_sim = ctrl.ControlSystemSimulation(stroke_ctrl)
print("Rebuilt fuzzy system from spec. Rules:", len(rules_list))

# %%
# BUILD CHROMOSOME
numeric_features = ['age', 'bmi', 'avg_glucose_level']

# NORMALIZATION HELPERS
def normalize_param(val, lo, hi):
    return 0.0 if hi==lo else (val-lo)/(hi-lo)

def denormalize_param(norm, lo, hi):
    return lo + norm*(hi-lo)

# BUILD INITIAL CHROMOSOME
chromosome_parts = []
numeric_mf_meta = {}
for feat in numeric_features:
    uni_lo, uni_hi = spec['universes'][feat]
    mfs = spec['mfs'][feat]
    mf_names = [n for n in ['young','low','normal','middle','elevated','high','old'] if n in mfs]
    if not mf_names: mf_names = sorted(mfs.keys())
    numeric_mf_meta[feat] = []
    for mf_name in mf_names:
        params = mfs[mf_name]['params']
        norm_params = [normalize_param(p, uni_lo, uni_hi) for p in params]
        chromosome_parts.extend(norm_params)
        numeric_mf_meta[feat].append({'mf_name': mf_name,'type': mfs[mf_name]['type'],'params_original': params,'uni_lo': uni_lo,'uni_hi': uni_hi,'n_params': len(params)})

initial_chromosome = np.array(chromosome_parts, dtype=float)

# INDEX MAP
index_map = {}
idx = 0
for feat, metas in numeric_mf_meta.items():
    n_params = sum(m['n_params'] for m in metas)
    index_map[feat] = (idx, n_params)
    idx += n_params

# RULE TEMPLATES
rule_templates = [{'antecedent': a,'consequent': c} for a,c in spec['rules']]

# %%
# BUILD FUZZY SYSTEM FROM CHROMOSOME (FUNCTION)
def build_fuzzy_system_from_chromosome(chrom, universes, numeric_mf_meta, index_map, rule_templates):
    fuzzy_vars = {}
    for var_name, (lo, hi) in universes.items():
        if var_name != 'stroke_risk':
            fuzzy_vars[var_name] = ctrl.Antecedent(np.arange(lo, hi+1, 1), var_name)
        else:
            fuzzy_vars[var_name] = ctrl.Consequent(np.arange(lo, hi+1, 1), var_name)
    for feat, metas in numeric_mf_meta.items():
        start_idx, _ = index_map[feat]
        idx = start_idx
        for meta in metas:
            segment = chrom[idx:idx+meta['n_params']]
            idx += meta['n_params']
            params = [meta['uni_lo'] + p*(meta['uni_hi']-meta['uni_lo']) for p in segment]
            if meta['type']=='trimf':
                fuzzy_vars[feat][meta['mf_name']] = fuzz.trimf(fuzzy_vars[feat].universe, params)
            elif meta['type'] in ['trap','trapmf']:
                fuzzy_vars[feat][meta['mf_name']] = fuzz.trapmf(fuzzy_vars[feat].universe, params)
    rules = []
    for rule in rule_templates:
        left_expr = None
        for feat, mf_name in rule['antecedent'].items():
            cond = fuzzy_vars[feat][mf_name]
            left_expr = cond if left_expr is None else left_expr & cond
        out_feat, out_mf = list(rule['consequent'].items())[0]
        right_expr = fuzzy_vars[out_feat][out_mf]
        rules.append(ctrl.Rule(left_expr, right_expr))
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return fuzzy_vars, rules, system, sim

# %%
# FITNESS FUNCTION
def fitness_function(chromosome):
    _, _, system, _ = build_fuzzy_system_from_chromosome(chromosome, spec['universes'], numeric_mf_meta, index_map, rule_templates)
    preds = []
    for i in range(len(X_fuzzy)):
        sim = ctrl.ControlSystemSimulation(system)
        for var in X_fuzzy.columns:
            sim.input[var] = float(X_fuzzy.iloc[i][var])
        try: sim.compute(); pred_value = sim.output['stroke_risk']
        except: pred_value = 1
        preds.append(int(round(pred_value)))
    return accuracy_score(y_fuzzy, preds)

# %%
# GA FUNCTION PLACEHOLDERS (tournament, crossover, mutation, penalty)
def tournament_selection(population, fitness_scores, k=4):
    idxs = random.sample(range(len(population)), k)
    return population[max(idxs, key=lambda i: fitness_scores[i])].copy()

def blx_alpha_crossover(p1, p2, alpha=0.4):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        low, high = min(p1[i], p2[i]), max(p1[i], p2[i])
        diff = high - low
        c1[i] = random.uniform(low-alpha*diff, high+alpha*diff)
        c2[i] = random.uniform(low-alpha*diff, high+alpha*diff)
    return np.clip(c1,0,1), np.clip(c2,0,1)

def adaptive_mutation(chrom, rate):
    child = chrom.copy()
    sigma = 0.08 + 0.3*rate
    mask = np.random.rand(child.size) < rate
    noise = np.random.normal(0,sigma, size=child.shape)
    child[mask] += noise[mask]
    return np.clip(child,0,1)

def penalty_function(chrom):
    pen = 0.0
    for feat, (start, length) in index_map.items():
        vals = chrom[start:start+length]
        if np.any(vals<0) or np.any(vals>1): pen+=0.02
        metas = numeric_mf_meta.get(feat, [])
        pos=0
        for meta in metas:
            seg=vals[pos:pos+meta['n_params']]; pos+=meta['n_params']
            for j in range(len(seg)-1):
                if seg[j]>seg[j+1]+1e-8: pen+=0.01
    return pen

def fitness_with_penalty(chrom):
    return fitness_function(chrom)-penalty_function(chrom)

# %%
# MAIN GA LOOP
def run_advanced_ga():
    population = [np.clip(initial_chromosome + np.random.normal(0,0.01,size=initial_chromosome.shape),0,1) for _ in range(40)]
    best_scores=[]; mutation_rate=0.25
    for gen in range(50):
        fitness_scores = np.array(Parallel(n_jobs=-1)(delayed(fitness_with_penalty)(c) for c in population))
        best_idx = np.argmax(fitness_scores); best_scores.append(fitness_scores[best_idx])
        print(f"Generation {gen+1}/50 | Best fitness = {best_scores[-1]:.4f}")
        elites = [population[i].copy() for i in np.argsort(fitness_scores)[-3:]]
        new_population = elites.copy()
        while len(new_population)<len(population):
            p1 = tournament_selection(population, fitness_scores)
            p2 = tournament_selection(population, fitness_scores)
            if random.random()<0.85: c1,c2 = blx_alpha_crossover(p1,p2)
            else: c1,c2=p1.copy(),p2.copy()
            c1=adaptive_mutation(c1, mutation_rate); c2=adaptive_mutation(c2, mutation_rate)
            new_population.append(c1)
            if len(new_population)<len(population): new_population.append(c2)
        population=new_population
        if gen>0: mutation_rate = min(0.4, mutation_rate*1.12) if best_scores[-1]<=best_scores[-2] else max(0.05, mutation_rate*0.92)
    fitness_scores = np.array(Parallel(n_jobs=-1)(delayed(fitness_with_penalty)(c) for c in population))
    best_chrom = population[np.argmax(fitness_scores)]
    print("\nGA DONE! Best GA fitness:", max(fitness_scores))
    return best_chrom, best_scores
