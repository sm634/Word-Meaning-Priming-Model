import pdb
import math as m
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from WMP_modelling.wmp_functions import *
from tqdm import tqdm


sns.set()
n_models = 4
n_items = 32

"""Hyper Parameter optimisation recordings."""
max_features_retrieved = []
learning_rate_list = []
training_epochs_list = []


n_word_type_conditions = 2
n_amb_words_per_condition = int(n_items / n_word_type_conditions)

# Assigning the number of orthographic and semantic units in the network. Setting 10% of these units on.
n_orth_units = 50
n_orth_active = int(n_orth_units * 0.2)

n_sem_units = 200
# Generating the total active number of features using a normal distribution.
n_epochs = 512
learning_rate = 5

for model in range(n_models):
    # print('\nN. of items: ', n_items, '\n')
    print(n_sem_units)
    # Initialise vectors representing the orthographic and semantic units.

    orth = np.zeros([n_items, n_orth_units])

    sem_meaning_1 = np.zeros([n_items, n_sem_units])
    sem_meaning_2 = np.zeros([n_items, n_sem_units])

    """## Partitioning the items into different word types"""
    # 50% = ambiguous, 90-10
    # 50% = ambiguous, 50-50

    ## Activating random orth units.
    for item in range(0, n_items):
        orth[item, 0:n_orth_active] = 1
        np.random.shuffle(orth[item, :])

    # Activating random sem units, depending on whether it falls on the ambiguous calss or not.
    # Variable Version.
    n_sem_active = []
    for item in range(0, n_items):
        n_sem_active.append(int(np.random.normal(loc=15, scale=2.5)))

    for item in range(0, n_items):
        sem_meaning_1[item, 0: n_sem_active[item]] = 1
        np.random.shuffle(sem_meaning_1[item, :])
        sem_meaning_2[item, 0: n_sem_active[item]] = 1
        np.random.shuffle(sem_meaning_2[item, :])

    # Controlled Version
    # n_sem_active = int(n_sem_units * 0.1)
    # for item in range(n_items):
    #     sem_meaning_1[item, 0: n_sem_active] = 1
    #     np.random.shuffle(sem_meaning_1[item, :])
    #     sem_meaning_2[item, 0: n_sem_active] = 1
    #     np.random.shuffle(sem_meaning_2[item, :])

    # Training Procedure here...

    n_units = n_orth_units + n_sem_units

    weights = np.zeros([n_units, n_units])

    fifty_fifty = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    ninety_ten = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])


    for training_epoch in tqdm(range(n_epochs)):
        # print("Epoch: ", training_epoch + 1)

        item_numbers = np.random.permutation(
            n_items)  # Setting up the items in a random order to be trained for the network.

        for j, item in enumerate(item_numbers):
            activation = np.zeros([1, n_units])
            activation[:, 0:n_orth_units] = orth[item, :]
            semantic = np.zeros([1, n_sem_units])

            fifty_fifty_choice = np.random.choice(fifty_fifty, 1)
            ninety_ten_choice = np.random.choice(ninety_ten, 1)

            # Setting up the unambiguous items
            if (0 <= item < n_amb_words_per_condition) and ninety_ten_choice == 1:  # If dominant meaning.
                semantic[:, :] = sem_meaning_1[item, :]
            elif (0 <= item < n_amb_words_per_condition) and ninety_ten_choice == 0:
                semantic[:, :] = sem_meaning_2[item, :]  # If subordinate meaning.

            # Setting up the 75-25 split.
            elif (n_amb_words_per_condition <= item < n_items) and fifty_fifty_choice == 1:
                semantic[:, :] = sem_meaning_1[item, :]
            elif (n_amb_words_per_condition <= item < n_items) and fifty_fifty_choice == 0:
                semantic[:, :] = sem_meaning_2[item, :]

            # Adding noise to the network to improve/make it's performance manageable.
            # This only requires taking one of the active units for a word and switching it off.
            index_array = np.where(semantic[0, :] == 1)
            index = np.random.choice(index_array[0])
            semantic[:, index] = 0
            activation[:, n_orth_units:n_units] = semantic

            # Setting up the vectors for the learning algorithm to train the weights.
            # Setting up further variables containing initial weight values and sending units.
            delta_weights_1 = np.zeros([n_units, n_units])
            sending_units = np.zeros([1, n_units])
            receiving_units = np.zeros([1, n_units])

            sending_units[:, :] = activation[:, :]  # Input (sending) vector.
            receiving_units[:, n_orth_units:n_units] = activation[:, n_orth_units:n_units]  # Output (receiving) vector.
            receiving_units[:, 0:n_orth_units] = 0

            v = np.zeros([n_units, 1])
            # Computing the dot product of the weights and the sending unit
            # to get the second term in the nominator of the learning algorithm.
            for unit in range(0, len(weights[0, :])):
                v[unit, :] = np.dot(weights[:, unit].transpose(), sending_units.transpose())

            # Ensuring that the dot product values do not subtract from the activations of r set to 0.
            v[0:n_orth_units, :].fill(0)

            # Numerator of the learning algorithm.
            difference = receiving_units - v.transpose()
            numerator_terms = sending_units.transpose() * difference
            numerator = assign_zero_trace(
                numerator_terms)  # This functions makes the diagonal of the matrix 0, so avoids self-connection
            # within a unit.

            # The full learning algorithm equation from Rodd (2004).
            delta_weights_1[:, :] = (learning_rate * numerator) / n_units

            weights = weights + delta_weights_1

    weights_show = plt.imshow(weights)
    plt.colorbar(weights_show)

    # weights_df = pd.DataFrame(weights)
    # weights_df.to_csv('C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access '
    #                   'Model\\WMP\\replication\\rodd2004_dom_effect\\dom_effect_test2_weights.csv')

    # Test the network here randomly ordered and asynchronous. This loops over all the items and stores
    # information on the ## activated word.

    test_updates = 100
    n_update_cycles = 2000
    rounded_update_cycle = int((n_update_cycles / test_updates) + 1)

    timecourse = np.zeros([n_items, rounded_update_cycle, test_updates])
    meaning_timecourse = np.zeros([n_items, 10, test_updates])
    correct_items = np.zeros([n_items, test_updates])
    meaning_selected = np.zeros([n_items, test_updates])
    settled_output = np.zeros([n_items, n_sem_units, test_updates])

    for repeat_test in tqdm(range(test_updates)):
        # print("for Repeat Test: ", repeat_test)

        item_numbers = np.random.permutation(n_items)

        for i, item in enumerate(item_numbers):

            activation = np.zeros([1, n_units])
            activation[:, 0:n_orth_units] = orth[item, :]

            for update_cycle in range(0, n_update_cycles):

                unit = np.random.randint(n_orth_units, n_units)

                net_input = np.dot(activation[:, :], weights[:, unit])
                #
                # Weights test update
                if net_input > 0.5:
                    activation[:, unit] = 1
                else:
                    activation[:, unit] = 0

                if update_cycle % 99 == 0:
                    n_active_sem_units = activation[:, n_orth_units:n_units].sum()
                    rounded_update_cycle = int(update_cycle / 99)
                    timecourse[item, rounded_update_cycle, repeat_test] = n_active_sem_units

                    # Calculating the number of features activated for each meaning. This is computed as the hamming
                    # distance between the original semantic pattern for a given semantic patter and the activation
                    # pattern retrieved from the test.
            n_features_meaning_1 = np.multiply(sem_meaning_1[item, :], activation[:, n_orth_units:n_units]).sum()
            n_features_meaning_2 = np.multiply(sem_meaning_2[item, :], activation[:, n_orth_units:n_units]).sum()

            # Check to see if 90% of the units are active for either of the meanings.
            # activation_threshold = (n_sem_active[item] * 50) / 100
            activation_threshold = int(n_sem_active[item] * 0.9)

            if (n_features_meaning_1 > activation_threshold) or (n_features_meaning_2 > activation_threshold):
                correct_items[item, repeat_test] = 1
                if n_features_meaning_1 > n_features_meaning_2:
                    meaning_selected[item, repeat_test] = 1
                elif n_features_meaning_2 > n_features_meaning_1:
                    meaning_selected[item, repeat_test] = 2
                else:
                    meaning_selected[
                        item, repeat_test] = 0  # This could nte nan, but not sure if that make computations
                    # downstream easier.
            else:
                correct_items[item, repeat_test] = 0

            settled_output[item, :, repeat_test] = activation[:, n_orth_units:n_units]

    """Save Output of meaning settlement"""
    # meaning_selected_df = pd.DataFrame(meaning_selected)
    # meaning_selected_df.to_csv('C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access '
    #                            'Model\\WMP\\replication\\rodd2004_dom_effect\\dom_effect_meaning_selected.csv')

    """### Saving the Proportion of Incorrect Items per Parameter Change### """
    # reshape_param = n_items * test_updates
    # correct_items_all = correct_items.reshape(reshape_param)
    # incorrect_records.append(correct_items_all.shape[0] - (correct_items_all.sum() / correct_items_all.shape[0]))

    # Plotting average timecourse for the model to settle for each word meaning condition.
    average_timecourse = np.zeros([rounded_update_cycle, n_word_type_conditions])

    for i in range(0, rounded_update_cycle):
        average_timecourse[i, 0] = timecourse[0:n_amb_words_per_condition, i, :].mean()
        average_timecourse[i, 1] = timecourse[n_amb_words_per_condition:n_items, i, :].mean()

    max_features_retrieved.append((average_timecourse[:, 0].max(), average_timecourse[:, 1].max()))

    ####### Plotting Timecourse #######
    # set up the variables to be plotted.
    x = []
    for i in range(0, 20):
        x.append(i + 1)
    # Getting errors to plot in errorbar.
    yerr = np.zeros([rounded_update_cycle, n_word_type_conditions])

    for i in range(0, 20):
        yerr[i, 0] = (np.divide(average_timecourse[i, 0], np.sqrt(n_amb_words_per_condition)))
        yerr[i, 1] = (np.divide(average_timecourse[i, 1], np.sqrt(n_amb_words_per_condition)))

    yerr1 = stats.sem(yerr[:, 0])
    yerr2 = stats.sem(yerr[:, 1])

    fig, (ax1) = plt.subplots(1, 1)

    ax1.errorbar(x, average_timecourse[:, 0], yerr=yerr1)
    ax1.errorbar(x, average_timecourse[:, 1], yerr=yerr2)
    ax1.set_xlabel('Update Cycle (x100)')
    ax1.set_ylabel('N Semantic Features')
    ax1.legend(('90-10', '50-50'), loc='lower right')

    """#### Meaning selected ######"""
    meaning_selected_array = np.array(meaning_selected)

    ninety_meaning_selected = (m.fsum(i == 1 for i in meaning_selected_array[0:n_amb_words_per_condition, :].reshape(
        n_amb_words_per_condition * 100)) / n_amb_words_per_condition * 100)
    ten_meaning_selected = (m.fsum(i == 2 for i in meaning_selected_array[0:n_amb_words_per_condition, :].reshape(
        n_amb_words_per_condition * 100)) / n_amb_words_per_condition * 100)
    ninety_ten_unsettled = (m.fsum(i == 0 for i in meaning_selected_array[0:n_amb_words_per_condition, :].reshape(
        n_amb_words_per_condition * 100)) / n_amb_words_per_condition * 100)

    fifty_first_meaning_selected = (m.fsum(i == 1 for i in meaning_selected_array[
                                                           n_amb_words_per_condition:n_items, :].reshape(
        n_amb_words_per_condition * 100)) / n_amb_words_per_condition * 100)
    fifty_second_meaning_selected = (m.fsum(i == 2 for i in meaning_selected_array[
                                                            n_amb_words_per_condition:n_items, :].reshape(
        n_amb_words_per_condition * 100)) / n_amb_words_per_condition * 100)
    fifty_fifty_unsettled = (m.fsum(i == 0 for i in meaning_selected_array[
                                                    n_amb_words_per_condition:n_items, :].reshape(
        n_amb_words_per_condition * 100)) / n_amb_words_per_condition * 100)

    ### Group bar chart prep ###
    meaning_one_selected = [ninety_meaning_selected, fifty_first_meaning_selected]
    meaning_two_selected = [ten_meaning_selected, fifty_second_meaning_selected]
    error_settling = [ninety_ten_unsettled, fifty_fifty_unsettled]

    labels = ['Ninety-ten', 'fifty-fifty']
    x = np.arange(len(labels))
    width = 0.25

    n_sem_units += 100
    learning_rate -= int(learning_rate/2)
    n_epochs += int(n_epochs/2)

    ## Plotting bar chart to show meaning settlement patterns.
    fig2, ax2 = plt.subplots()
    meaning_one = ax2.bar(x - width / 2, meaning_one_selected, width)
    meaning_two = ax2.bar(x + width / 2, meaning_two_selected, width)

    ax2.set_xlabel('Proportion of training split between semantic representations per word')
    ax2.set_ylabel('Number of times settled on meaning attractor basin')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(['Attractor basin 1', 'Attractor basin 2'])

    fig2.tight_layout()

    plt.show()

# fig3 = plt.subplots()
# ax3 = plt.axes(projection='3d')
# ax3.plot3D(learning_rate_list, training_epochs_list, max_features_retrieved)
#
# plt.show()
