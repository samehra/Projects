# -*- coding: utf-8 -*-
"""
Created on Sat May 11 00:37:25 2019

@author: saura
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd

filePath = "data/ml-100k/u1.base"
#filePath = "Data/pinterest-20.train.rating"

with open(filePath, "rt") as dataPath:
    raw_data = dataPath.read().splitlines()
datapoints = [[int(i) for i in data.split("\t")] for data in raw_data]

# indexing on users/movies starts at 1, reset to index from 0, this will be important when we do testing
datapoints = np.array([[row[0], row[1], row[2], row[3]] for row in datapoints])
#np.random.shuffle(datapoints)

datapoints = datapoints[np.lexsort((datapoints[:,1], datapoints[:,0], datapoints[:,3]))]
time_index = datapoints[:,3]; user_index = datapoints[:,0]; item_index = datapoints[:,1]
user_ids = sorted(set([datapoint[0] for datapoint in datapoints]))
timestamps = sorted(set([datapoint[3] for datapoint in datapoints]))
df = pd.DataFrame(datapoints[:,0], columns = ['user_id'], index = time_index)
time_userids = df.to_dict('index')

n_users = len(user_ids)
max_user = max(user_ids) + 1
print("n users:", n_users)
items_ids = set([datapoint[1] for datapoint in datapoints])
n_items = len(items_ids)
max_item = max(items_ids) + 1
print("n items:", n_items)

def softmax(x):
    # subtract max value to prevent overflow\n"
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def stratified_resample(weights):
    N = len(weights)
    # make N subdivisions, chose a random position within each one
    positions = (np.random.random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def systematic_resample(weights):
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def Update_Posterior(particles, user_id, recommend_item, obs_rating, user_history, item_history):
    # line 17
    precision_u_i = []
    eta_u_i = []
    
    for particle in particles:
#        if user_id not in user_history:
#            precision_u_i.append(np.eye(k))
#            eta_u_i.append(np.zeros(k))
#        else:
        v_j = particle[1]["v"][user_history[user_id]["item_ids"], :]
        lambda_u_i = 1 / var * np.dot(v_j.T, v_j) + 1 / particle[1]["var_u"] * np.eye(k)

        precision_u_i.append(lambda_u_i)

        eta = np.sum(
            np.multiply(
                v_j,
                np.array(user_history[user_id]["ratings"]).reshape(-1, 1)
            ),
            axis=0
        )
        eta_u_i.append(eta.reshape(-1))

    # line 18
    weights = []
    mus = [1 / var * np.dot(np.linalg.inv(lambda_), eta) for lambda_, eta in zip(precision_u_i, eta_u_i)]
    for particle, mu, precision in zip(particles, mus, precision_u_i):
        v_j = particle[1]["v"][recommend_item, :]
        cov = 1 / var + np.dot(np.dot(v_j.T, precision), v_j)
        w = np.random.normal(np.dot(v_j.T, mu), cov)
        weights.append(w)
    normalized_weights = softmax(weights)

    # line 19
    eff_threshold = 4
    Neff = 1/sum([i**2 for i in normalized_weights])
    # perform all resampling methods, but only use the resampled particles named ds
    if Neff >= eff_threshold:
        multis = [np.random.choice(range(n_particles), p=normalized_weights) for _ in range(n_particles)]
        multi.append(len(set(multis)))

        str_idx = stratified_resample(normalized_weights)
        strat.append(len(set(str_idx)))
        
        # sys_idx
        ds = systematic_resample(normalized_weights)
        system.append(len(set(ds)))
    else:
        ds = [i for i in range(n_particles)]

    p_prime = [{"u": np.copy(particles[d][1]["u"]),
                "v": np.copy(particles[d][1]["v"]),
                "var_u": particles[d][1]["var_u"],
                "var_i": particles[d][1]["var_i"]} for d in ds]
        
    for idx, (particle, precision, e) in enumerate(zip(p_prime, precision_u_i, eta_u_i)):

        # line 21
        v_j = particle["v"][recommend_item, :]
        add_to_precision = 1 / var * np.dot(v_j.reshape(-1, 1), v_j.reshape(1, -1))
        precision += add_to_precision

        add_to_eta = obs_rating * v_j
        e += add_to_eta

        # line 22
        sampled_user_vector = np.random.multivariate_normal(
            1 / var * np.dot(np.linalg.inv(precision), e),
            np.linalg.inv(precision)
        )
        
        p_prime[idx]["u"][user_id, :] = sampled_user_vector

        # line 24
#        if recommend_item not in item_history:
#            precision_v_i = np.eye(k)
#            eta = np.zeros(k)
#        else:
        u_i = particle["u"][item_history[recommend_item]["user_ids"], :]
        precision_v_i = 1 / var * \
            np.dot(u_i.T, u_i) + \
            1 / particle["var_i"] * np.eye(k)

        eta = np.sum(
            np.multiply(
                u_i,
                np.array(item_history[recommend_item]["ratings"]).reshape(-1, 1)
            ),
            axis=0
        )
        # line 25
        item_sample = np.random.multivariate_normal(
            1 / var * np.dot(np.linalg.inv(precision_v_i), eta),
            np.linalg.inv(precision_v_i)
        )
        p_prime[idx]["v"][recommend_item, :] = item_sample
        
        a = n_users*k/2 + alpha; b = (1/2)*np.sum((p_prime[idx]["u"])**2) + beta              # Is the np.sum for entire u or u for one user #
        p_prime[idx]["var_u"] = 1/ss.gamma.rvs(a=a, scale = 1/b)

    particles = [(1 / n_particles, particle) for particle in p_prime]
    
    return particles


#    if user_id not in user_history:
#        user_history[user_id] = {"item_ids": [], "ratings": []}
#    if item_id not in item_history:
#        item_history[item_id] = {"user_ids": [], "ratings": []}
#    user_history[user_id]["item_ids"].append(item_id)
#    user_history[user_id]["ratings"].append(rating)
#    item_history[item_id]["user_ids"].append(user_id)
#    item_history[item_id]["ratings"].append(rating)

particle_vals = [5]
feature_vals = [2]
num_iters = 1
mse_matrix = np.zeros((len(particle_vals),num_iters))
cum_reg_matrix = np.zeros((len(particle_vals),num_iters))


for confg, (num_particles, num_features) in enumerate(zip(particle_vals, feature_vals)):
    for iters in range(num_iters):

        n_particles = num_particles; k = num_features; var = 0.5
        alpha = 2; beta = 0.5
        
        particles = [(1 / n_particles, {"u": np.random.normal(size=(max_user, k)),
                                        "v": np.random.normal(size=(max_item, k)),
                                        "var_u": 1.0,
                                        "var_i": 1.0}) for _ in range(n_particles)]
        
        # get mean rating to make rating data centered at 0
        #mean_rating = np.mean(datapoints[:, 2])
        mean_rating = np.array(pd.DataFrame(datapoints).groupby(datapoints[:,0]).mean())[:,2].reshape(-1,1)
        #data_store = {u_id: {row[1]: row[2] - mean_rating for row in datapoints if row[0] == u_id} for u_id in user_ids}
        data_store = {u_id: {row[1]: row[2] - mean_rating[row[0]-1] for row in datapoints if row[0] == u_id} for u_id in user_ids}
        
        #df_index = [timestamps, user_ids, item_ids]
        #df = pd.DataFrame(datapoints[:,[1,2]], columns = ['item_id','ratings'], index = [time_index, user_index])
        #data_dict = df.to_dict('index')
        
        user_history = {} # user_rating_history[user_id]["item_ids"], user_rating_history[user_id]["ratings"]
        item_history = {} # item_rating_history[item_id]["user_ids"], item_rating_history[item_id]["ratings"]
#        ctr = 0
#        ctr_hist = []
        se = []; regret = []; mse = []; cum_reg_hist = []
        uniParticlesList = []
        multi, strat, system = [], [], []
        #t=874724727
        # highest_rating -> actual_rating, item_id->recommend_item, rating->obs_rating
        
        for _idx, t in enumerate(timestamps):
            
        #    for user_id in time_userids[t]['user_id']:
        #        user_items = data_dict[t,user_id]['item_id']
            user_id = time_userids[t]['user_id']
        #    # randomly get a user
        #    user_id = np.random.choice([i for i in data_store[t].keys()])
            user_items = [i for i in data_store[user_id].keys()]
        #    
        #    # highest rating this user has
        #    highest_rating = max(data_store[user_id].values())
        #    # get highest rated items
        #    highest_rated_items = [x for x in data_store[user_id].keys() if data_store[user_id][x] >= highest_rating]
        #    
            # get indices for items this user rated
            indices = np.array(user_items)
        
            # randomly select a particle
            random_particle = np.random.choice(range(n_particles))
            particle = particles[random_particle]
                                                                                             # In a = NK/2, what is N? #
        #    a = n_users*k/2 + alpha; b = (1/2)*np.sum((particle[1]["u"])**2) + beta              # Is the np.sum for entire u or u for one user #
        #    particle[1]["var_u"] = 1/ss.gamma.rvs(a=a, scale = 1/b)
        
            # should we consider only items this particular user rated or all the items
        
            if user_id not in user_history:
                precision_init = np.eye(k)
                eta_init = np.zeros(k)
            else:
                v_init = particle[1]["v"][user_history[user_id]["item_ids"], :]
                precision_init = 1 / var * np.dot(v_init.T, v_init) + 1 / particle[1]["var_u"] * np.eye(k)
                eta_init = np.sum(np.multiply(v_init,np.array(user_history[user_id]["ratings"]).reshape(-1, 1)),axis=0)
            
            init_user_vector = np.random.multivariate_normal(
                    1 / var * np.dot(np.linalg.inv(precision_init), eta_init),
                    np.linalg.inv(precision_init))
        
            particle[1]["u"][user_id, :] = init_user_vector
            
            
            # predict a rating only for the items rated by that user
            predicted_rating = np.dot(particle[1]["u"][user_id, :], particle[1]["v"][indices, :].T)
            recommend_item_index = np.argmax(predicted_rating)
            recommend_item = [i for i in data_store[user_id].keys()][recommend_item_index]
        #    # get the item id
        #    max_rating_ind = np.argmax(predicted_rating)
        #    item_id = [i for i in data_store[user_id].keys()][max_rating_ind]
        #    
        #    # add to ctr if possible
        #    if item_id in highest_rated_items:
        #        ctr += 1
        #    ctr_hist.append(ctr / (_idx + 1))
        #    
        #    # get the true rating
        #    rating = data_store[user_id][item_id]
        #    
        #    # delete this item from this user
        #    del data_store[user_id][item_id]
        #    
        #    # delete the user from the data store if they have no reviews left
        #    if not data_store[user_id]:
        #        del data_store[user_id]
        
            obs_rating = data_store[user_id][recommend_item]
            error = predicted_rating[recommend_item_index] - obs_rating
            se.append(error ** 2)
            regret.append(error)
            if _idx % 101 == 0:
                mse.append(np.mean(se))
            if _idx % 1000 == 0:
                cum_reg_hist.append(np.sum(regret))
        #        print("squared error: {:.2f}".format(se))
                print("cumulative regret: {:.2f}".format(cum_reg_hist[_idx//1000]))
            mse_matrix[confg, iters] = np.mean(se)
            cum_reg_matrix[confg, iters] = np.sum(regret)
            
        
        #    if user_id not in user_history:
        #        user_history[user_id] = {t:{"item_ids": [], "ratings": []}}
        #    if recommend_item not in item_history:
        #        item_history[recommend_item] = {t:{"user_ids": [], "ratings": []}}
        #    user_history[user_id][t]["item_ids"].append(recommend_item)
        #    user_history[user_id][t]["ratings"].append(obs_rating)
        #    item_history[recommend_item][t]["user_ids"].append(user_id)
        #    item_history[recommend_item][t]["ratings"].append(obs_rating)
        
            if user_id not in user_history:
                user_history[user_id] = {"item_ids": [], "ratings": []}
            if recommend_item not in item_history:
                item_history[recommend_item] = {"user_ids": [], "ratings": []}
            user_history[user_id]["item_ids"].append(recommend_item)
            user_history[user_id]["ratings"].append(obs_rating)
            item_history[recommend_item]["user_ids"].append(user_id)
            item_history[recommend_item]["ratings"].append(obs_rating)
        
            particles = Update_Posterior(particles, user_id, recommend_item, obs_rating, user_history, item_history)

    
print("Avg num particles Multinomial:", round(np.mean(multi),2) ) # On average looks like multi uses less particles
print("Avg num particles Stratfied:", round(np.mean(strat),2) ) # Followed slightly by stratefied sampling
print("Avg num particles Systematic:", round(np.mean(system),2) ) # Systematic uses most -- wonder if should change from multi above to systematic?

def average_particles(p_hist, interval=4):
    avg = []
    for i in range(0,len(strat)-1, interval):
        avg.append(sum(strat[i : i + interval])/interval)
    return avg

# Need to make this just a smoothed line graph. Line graph isn't showing same result
def plot_avgParticles(avg, n_particles=n_particles, k=k):
    plt.plot(avg)
    plt.title("Final Particle Number: {:.0f} | Particles Num = {}, K = {}".format(avg[-1], n_particles, k))
    plt.xlabel('resample step')
    plt.show()
    
# Plot of multinomial average every 2 iters
# Like how it shows we use ~5 particles then -> 1
avg = average_particles(multi, 2)
plot_avgParticles(avg)

# Plot of stratfied resampling every  10 iters
avg = average_particles(strat, 10)
plot_avgParticles(avg)

# Plot of systematic resampling every  100 iters
# Smooth line but seems like only use ~2 -> 1
avg = average_particles(system, 100)
plot_avgParticles(avg)

def moving_average(x):
    avgs = []
    for i in range(1,len(x)):
        avgs.append(np.sum(x[:i]) / i)
    return avgs

mses = moving_average(np.array(se))
plt.figure()
plt.plot(mses)
plt.title("Final MSE: {:.2f} | Particles={} Num_Features={}".format(mses[-1], n_particles, k))
plt.xlabel('iterations')
plt.ylabel('Mean Squared Error')
plt.show()

plt.figure()
cum_regret = np.cumsum(regret)
plt.plot(cum_regret, label = 'PTS-B')
y_lim = (0,20000)
x_lim = (0,20000)
plt.plot(x_lim, y_lim, 'k-', color = 'r', label = 'random')
plt.title("Cumulative Regret: {:.2f} | Particles={} Num_Features={}".format(cum_regret[-1], n_particles, k))
plt.xlabel('iterations')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.show()

#plt.scatter(range(len(ctr_hist)), ctr_hist)
#plt.title("cumulative take rate {:.2f}".format(ctr_hist[-1]))
#plt.show()