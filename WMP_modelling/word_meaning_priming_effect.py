import pdb
import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from WMP_modelling.wmp_functions import *
sns.set()


n_models = 10
n_items = 32
n_word_type_conditions = 4
n_unamb_words = 8
n_amb_words_per_condition = 8

# Assigning the number of orthographic and semantic units in the network. Setting 10% of these units on.
n_orth_units = 100
n_orth_active = int(n_orth_units*0.1)

n_sem_units = 200
n_sem_active = int(n_sem_units*0.1)

# Initialise vectors representing the orthographic and semantic units.

orth = np.zeros([n_items,n_orth_units])

sem_meaning_1 = np.zeros([n_items,n_sem_units])
sem_meaning_2 = np.zeros([n_items,n_sem_units])

## Partitioning the 32 items into different 'word types'.
# 1-8 = unambiguous
# 9-16 = ambiguous, 90-10
# 17-24 = ambiguous, 75-25
# 25-32 = ambiguous, 50-50


## Activating random orth units.
for item in range(0,n_items):
    orth[item,0:n_orth_active] = 1
    np.random.shuffle(orth[item,:])

## Activating random sem units, depending on whether it falls on the ambiguous calss or not.

for item in range(0,n_items):
    if item >=0 and item < 8:
        sem_meaning_1[item,0:n_sem_active] = 1
        np.random.shuffle(sem_meaning_1[item,:])
        sem_meaning_2[item,:] = sem_meaning_1[item,:]
    elif item >= 8 and item <= 31:
        sem_meaning_1[item, 0:n_sem_active] = 1
        np.random.shuffle(sem_meaning_1[item,:])
        sem_meaning_2[item, 0:n_sem_active] = 1
        np.random.shuffle(sem_meaning_2[item,:])




### Training Procedure here...

learning_rate = 5
n_units = n_orth_units + n_sem_units

weights = np.zeros([n_units,n_units])

fifty_fifty = np.array([1,1,1,1,1,0,0,0,0,0])
seventyfive_twentyfive = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0])
ninety_ten = np.array([1,1,1,1,1,1,1,1,1,0])

n_epochs = 512


for training_epoch in range(0,n_epochs):
    print("Epoch: ", training_epoch + 1)

    item_numbers = np.random.permutation(32)   # Setting up the items in a random order to be trained for the network.

    for j,item in enumerate(item_numbers):
        # print("item", item)
        activation = np.zeros([1,300])
        activation[:,0:100] = orth[item,:]
        semantic = np.zeros([1,200])

        fifty_fifty_choice = np.random.choice(fifty_fifty,1)
        seventyfive_twentyfive_choice = np.random.choice(seventyfive_twentyfive,1)
        ninety_ten_choice = np.random.choice(ninety_ten,1)

        # Setting up the unambiguous items
        if (item >= 0 and item < 8):
            sem_meaning_2[item,:] = sem_meaning_1[item,:]
            semantic[:,:] = sem_meaning_1[item,:]

        # Setting up the 90-10 split.
        elif (item >= 8 and item < 16) and ninety_ten_choice == 1:                  # If dominant meaning.
            semantic[:,:] = sem_meaning_1[item,:]
        elif (item >= 8 and item < 16) and ninety_ten_choice == 0:
            semantic[:,:] = sem_meaning_2[item,:]    # If subordinate meaning.

        # Setting up the 75-25 split.
        elif (item >= 16 and item < 24) and seventyfive_twentyfive_choice == 1:
            semantic[:,:] = sem_meaning_1[item,:]
        elif (item >= 16 and item < 24) and seventyfive_twentyfive_choice == 0:
            semantic[:,:] = sem_meaning_2[item,:]

        # Setting up the 50-50 split.
        elif (item >= 24 and item < 32) and fifty_fifty_choice == 1:
            semantic[:,:] = sem_meaning_1[item,:]
        elif (item >= 24 and item < 32) and fifty_fifty_choice == 0:
            semantic[:,:] = sem_meaning_2[item,:]

        # Adding noise to the network to improve/make it's performance manageable.
        # This only requires taking one of the active units for a word and switching it off.
        index_array = np.where(semantic[0,:] == 1)
        index = np.random.choice(index_array[0])
        semantic[:, index] = 0
        activation[:,100:300] = semantic


        # Setting up the vectors for the learning algorithm to train the weights.
        # Setting up further variables containing intital weight values and sending units.
        delta_weights_1 = np.zeros([300,300])
        sending_units = np.zeros([1,300])
        receiving_units = np.zeros([1,300])


        sending_units[:,:] = activation[:,0:300] # Input (sending) vector.
        receiving_units[:,100:300] = activation[:,100:300] # Output (receiving) vector.
        receiving_units[:,0:100] = 0

        v = np.zeros([300,1])
        # Computing the dot product of the weights and the sending unit
        # to get the second term in the nominator of the learning algorithm.
        for unit in range(0,len(weights[0,:])):
            v[unit,:] = np.dot(weights[:,unit].transpose(),sending_units.transpose())

        # Ensuring that the dot product values do not subtract from the activations of r set to 0.
        v[0:100,:].fill(0)

        # Numerator of the learning algorithm.
        difference = receiving_units - v.transpose()
        numerator_terms = sending_units.transpose() * difference
        numerator = assign_zero_trace(numerator_terms) # This functions makes the diagonal of the matrix 0, so avoids self-connection within a unit.

        # The full learning algorithm equation from Rodd (2004).
        delta_weights_1[:,:] = (learning_rate*numerator)/n_units

        weights = weights + delta_weights_1


weights_show = plt.imshow(weights)
plt.colorbar(weights_show)

####################################################################################################################
"""Priming trial goes here. Give a single trial training to boost the subordinate semantic pattern for each item."""

n_prime_training_epoch = 1
prime_weights = weights

for epoch in range(0,n_prime_training_epoch):

    item_numbers = np.random.permutation(32)  # Setting up the items in a random order to be trained for the network.

    for j, item in enumerate(item_numbers):
        # print("item", item)
        activation = np.zeros([1, 300])
        activation[:, 0:100] = orth[item, :]

        if item >= 8:
            # Only the meaning that was subordinate during the training trial will be primed.
            semantic = np.zeros([1,200])
            semantic[:,:] = sem_meaning_2[item,:]

            # Adding noise to the network to improve/make it's performance manageable.
            # This only requires taking one of the active units for a word and switching it off.
            index_array = np.where(semantic[0, :] == 1)
            index = np.random.choice(index_array[0])
            semantic[:, index] = 0
            activation[:, 100:300] = semantic

            # Setting up the vectors for the learning algorithm to train the weights.
            # Setting up further variables containing intital weight values and sending units.
            delta_weights_1 = np.zeros([300, 300])
            sending_units = np.zeros([1, 300])
            receiving_units = np.zeros([1, 300])

            sending_units[:, :] = activation[:, 0:300]  # Input (sending) vector.
            receiving_units[:, 100:300] = activation[:, 100:300]  # Output (receiving) vector.
            receiving_units[:, 0:100] = 0

            v = np.zeros([300, 1])
            # Computing the dot product of the weights and the sending unit
            # to get the second term in the nominator of the learning algorithm.
            for unit in range(0, len(weights[0, :])):
                v[unit, :] = np.dot(weights[:, unit].transpose(), sending_units.transpose())

            # Ensuring that the dot product values do not subtract from the activations of r set to 0.
            v[0:100, :].fill(0)

            # Numerator of the learning algorithm.
            difference = receiving_units - v.transpose()
            numerator_terms = sending_units.transpose() * difference
            numerator = assign_zero_trace(
                numerator_terms)  # This functions makes the diagonal of the matrix 0, so avoids self-connection within a unit.

            # The full learning algorithm equation from Rodd (2004).
            delta_weights_1[:, :] = (learning_rate * numerator) / n_units

            prime_weights = prime_weights + delta_weights_1

prime_weights_show = plt.imshow(prime_weights)
plt.colorbar(prime_weights_show)


#### Test the network here randomly ordered and asynchronous. This loops over all the items and stores information on the
### activated word.

n_update_cycles = 2000
rounded_update_cycle = int((n_update_cycles/100) + 1)

timecourse = np.zeros([32,rounded_update_cycle,100])
meaning_timecourse = np.zeros([32,10,100])
correct_items = np.zeros([32,100])
meaning_selected = np.zeros([32,100])
settled_output = np.zeros([32,200,100])

prime_timecourse = np.zeros([32,rounded_update_cycle,100])
primed_meaning_timecourse = np.zeros([32,10,100])
primed_correct_items = np.zeros([32,100])
primed_meaning_selected = np.zeros([32,100])
primed_settled_output = np.zeros([32,200,100])

for repeat_test in range(0,100):
    print("for Repeat Test: ",repeat_test)

    item_numbers = np.random.permutation(32)

    for i,item in enumerate(item_numbers):

        activation = np.zeros([1,300])
        activation[:,0:100] = orth[item,:]

        primed_activation = np.zeros([1,300])
        primed_activation[:,0:100] = orth[item,:]

        for update_cycle in range(0,n_update_cycles):

            unit = np.random.randint(100,300)

            net_input = np.dot(activation[:,:],weights[:,unit])
            primed_net_input = np.dot(primed_activation[:,:],prime_weights[:,unit])
#
            # Weights test update
            if net_input > 0.5:
                activation[:,unit] = 1
            else:
                activation[:,unit] = 0

            if primed_net_input > 0.5:
                primed_activation[:,unit] = 1
            else:
                primed_activation[:,unit] = 0

            if (update_cycle) % 99 == 0:
                n_active_sem_units = activation[:,100:300].sum()
                rounded_update_cycle = int((update_cycle)/99)
                timecourse[item,rounded_update_cycle,repeat_test] = n_active_sem_units

                n_prime_active_sem_units = primed_activation[:,100:300].sum()
                prime_timecourse[item,rounded_update_cycle,repeat_test] = n_prime_active_sem_units


        # Calculating the number of features activated for each meaning. This is computed as the hamming distance between
        # the original semantic pattern for a given semantic patter and the activation pattern retreived from the test.
        n_features_meaning_1 = np.multiply(sem_meaning_1[item,:],activation[:,100:300]).sum()
        n_features_meaning_2 = np.multiply(sem_meaning_2[item,:],activation[:,100:300]).sum()

        n_prime_features_meaning_1 = np.multiply(sem_meaning_1[item,:],primed_activation[:,100:300]).sum()
        n_prime_features_meaning_2 = np.multiply(sem_meaning_2[item,:],primed_activation[:,100:300]).sum()
        # Check to see if 90% of the units are active for either of the meanings.
        activation_threshold = (n_sem_active*90)/100

        if (n_features_meaning_1 > activation_threshold) or (n_features_meaning_2 > activation_threshold):
            correct_items[item,repeat_test] = 1
            if (n_features_meaning_1 > n_features_meaning_2):
                meaning_selected[item,repeat_test] = 1
            elif (n_features_meaning_2 > n_features_meaning_1):
                meaning_selected[item,repeat_test] = 2
            else:
                meaning_selected[item,repeat_test] = 0      # This could nte nan, but not sure if that make computations downstream easier.

        settled_output[item,:,repeat_test] = activation[:,100:300]

        if (n_prime_features_meaning_1 > activation_threshold) or (n_prime_features_meaning_2 > activation_threshold):
            primed_correct_items[item,repeat_test] = 1
            if (n_prime_features_meaning_1 > n_prime_features_meaning_2):
                primed_meaning_selected[item,repeat_test] = 1
            elif (n_prime_features_meaning_2 > n_prime_features_meaning_1):
                primed_meaning_selected[item,repeat_test] = 2
            else:
                primed_meaning_selected[item,repeat_test] = 0

        primed_settled_output[item,:,repeat_test] = primed_activation[:,100:300]

### Plotting average timecourse for the model to settle for each word meaning condition.
average_timecourse = np.zeros([rounded_update_cycle,n_word_type_conditions])

for i in range(0,rounded_update_cycle):
    average_timecourse[i,0] = timecourse[0:7,i,:].mean()
    average_timecourse[i,1] = timecourse[8:15,i,:].mean()
    average_timecourse[i,2] = timecourse[16:23,i,:].mean()
    average_timecourse[i,3] = timecourse[24:32,i,:].mean()

####### Plotting Timecourse #######
# set up the variables to be plotted.
x = []
for i in range(0,20):
    x.append(i+1)
# Getting errors to plot in errorbar.
yerr = np.zeros([rounded_update_cycle,n_word_type_conditions])

for i in range(0,20):
    yerr[i,0] = (np.divide(average_timecourse[i,0],np.sqrt(8)))
    yerr[i,1] = (np.divide(average_timecourse[i,1],np.sqrt(8)))
    yerr[i,2] = (np.divide(average_timecourse[i,2],np.sqrt(8)))
    yerr[i,3] = (np.divide(average_timecourse[i,3],np.sqrt(8)))

yerr1 = stats.sem(yerr[:,0])
yerr2 = stats.sem(yerr[:,1])
yerr3 = stats.sem(yerr[:,2])
yerr4 = stats.sem(yerr[:,3])

fig, (ax1) = plt.subplots(1,1)

ax1.errorbar(x,average_timecourse[:,0],yerr=yerr1)
ax1.errorbar(x,average_timecourse[:,1],yerr=yerr2)
ax1.errorbar(x,average_timecourse[:,2],yerr=yerr3)
ax1.errorbar(x,average_timecourse[:,3],yerr=yerr4)
ax1.set_xlabel('Update Cycle (x100)')
ax1.set_ylabel('N Semantic Features')
ax1.legend(('Non-ambiguous','90-10','75-25','50-50'),loc='lower right')

#### Meaning selected ######
meaning_selected_array = np.array(meaning_selected)

ninety_meaning_selected = (m.fsum(i == 1 for i in meaning_selected_array[8:16,:].reshape(800))/800)*100
ten_meaning_selected = (m.fsum(i == 2 for i in meaning_selected_array[8:16,:].reshape(800))/800)*100
ninety_ten_unsettled = (m.fsum(i == 0 for i in meaning_selected_array[8:16,:].reshape(800))/800)*100

seventyfive_meaning_selected = (m.fsum(i == 1 for i in meaning_selected_array[16:24,:].reshape(800))/800)*100
twentyfive_meaning_selected = (m.fsum(i == 2 for i in meaning_selected_array[16:24,:].reshape(800))/800)*100
seventyfive_twentyfive_unsettled = (m.fsum(i ==0 for i in meaning_selected_array[16:24,:].reshape(800))/800)*100

fifty_first_meaning_selected = (m.fsum(i == 1 for i in meaning_selected_array[24:32,:].reshape(800))/800)*100
fifty_second_meaning_selected = (m.fsum(i == 2 for i in meaning_selected_array[24:32,:].reshape(800))/800)*100
fifty_fifty_unsettled = (m.fsum(i == 0 for i in meaning_selected_array[24:32,:].reshape(800))/800)*100

#### Group bar chart prep ###
meaning_one_selected = [ninety_meaning_selected,seventyfive_meaning_selected,fifty_first_meaning_selected]
meaning_two_selected = [ten_meaning_selected, twentyfive_meaning_selected, fifty_second_meaning_selected]
error_settling = [ninety_ten_unsettled, seventyfive_twentyfive_unsettled,fifty_fifty_unsettled]

labels = ['Ninety-ten', 'Seventyfive-twentyfive','fifty-fifty']
x = np.arange(len(labels))
width = 0.25

## Plotting bar chart to show meaning settlement patterns.
fig2, ax2 = plt.subplots()
meaning_one = ax2.bar(x - width/2, meaning_one_selected, width)
meaning_two = ax2.bar(x + width/2, meaning_two_selected, width)

ax2.set_title('Dominance Effect')
ax2.set_xlabel('Proportion of training split between semantic representations per word')
ax2.set_ylabel('Number of times settled on meaning attractor basin')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(['Attractor basin 1', 'Attractor basin 2'])

fig2.tight_layout()

###### Primed Meaning Selected ######

primed_meaning_selected_array = np.array(primed_meaning_selected)

primed_ninety_meaning_selected = (m.fsum(i == 1 for i in primed_meaning_selected_array[8:16,:].reshape(800))/800)*100
primed_ten_meaning_selected = (m.fsum(i == 2 for i in primed_meaning_selected_array[8:16,:].reshape(800))/800)*100
primed_ninety_ten_unsettled = (m.fsum(i == 0 for i in primed_meaning_selected_array[8:16,:].reshape(800))/800)*100

primed_seventyfive_meaning_selected = (m.fsum(i == 1 for i in primed_meaning_selected_array[16:24,:].reshape(800))/800)*100
primed_twentyfive_meaning_selected = (m.fsum(i == 2 for i in primed_meaning_selected_array[16:24,:].reshape(800))/800)*100
primed_seventyfive_twentyfive_unsettled = (m.fsum(i ==0 for i in primed_meaning_selected_array[16:24,:].reshape(800))/800)*100

primed_fifty_first_meaning_selected = (m.fsum(i == 1 for i in primed_meaning_selected_array[24:32,:].reshape(800))/800)*100
primed_fifty_second_meaning_selected = (m.fsum(i == 2 for i in primed_meaning_selected_array[24:32,:].reshape(800))/800)*100
primed_fifty_fifty_unsettled = (m.fsum(i == 0 for i in primed_meaning_selected_array[24:32,:].reshape(800))/800)*100

#### Group bar chart prep ###
primed_meaning_one_selected = [primed_ninety_meaning_selected,primed_seventyfive_meaning_selected,primed_fifty_first_meaning_selected]
primed_meaning_two_selected = [primed_ten_meaning_selected, primed_twentyfive_meaning_selected, primed_fifty_second_meaning_selected]
primed_error_settling = [primed_ninety_ten_unsettled, primed_seventyfive_twentyfive_unsettled,primed_fifty_fifty_unsettled]

labels = ['Ninety-ten', 'Seventyfive-twentyfive','fifty-fifty']
x = np.arange(len(labels))
width = 0.25

## Plotting bar chart to show meaning settlement patterns.
fig3, ax3 = plt.subplots()
primed_meaning_one = ax3.bar(x - width/2, primed_meaning_one_selected, width)
primed_meaning_two = ax3.bar(x + width/2, primed_meaning_two_selected, width)

ax3.set_title('Primed Trial Setting Pattern')
ax3.set_xlabel('Proportion of training split between semantic representations per word')
ax3.set_ylabel('Number of times settled on meaning attractor basin')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend(['Attractor basin 1', 'Attractor basin 2'])

fig3.tight_layout()


plt.show()



