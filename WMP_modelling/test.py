import pdb
import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from WMP_modelling.wmp_functions import *

sns.set()
from tqdm import tqdm

"""############################ Input files ##############################"""

words_synsets = pd.read_csv(
    'C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access Model\\semantic '
    'representation\\words_synsets.csv')
sem_features = pd.read_csv(
    'C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access Model\\semantic '
    'representation\\semantic_representations.csv')

sem_features = sem_features.iloc[:, 1:]

"""########################### Model/network parameters #########################"""

n_orth_units = 100
n_sem_units = len(list(sem_features.iloc[:, 1]))
n_units = n_orth_units + n_sem_units
n_items = len(list(words_synsets.iloc[:, 0]))


"""###################### Setting up Semantic and Orthographic Representations ##########################"""
n_orth_active = int(n_orth_units * 0.1)

# Generating orthographic representations
orth = np.zeros([n_items, n_orth_units])
for item in range(0, n_items):
    orth[item, 0:n_orth_active] = 1
    np.random.shuffle(orth)

# Generating semantic representations
sem_meaning_1 = np.zeros([n_items, n_sem_units])
sem_meaning_2 = np.zeros([n_items, n_sem_units])
for item in range(0, n_items):
    sem_meaning_1[item, :] = np.array(sem_features.iloc[:, item])
    sem_meaning_2[item, :] = np.array(sem_features.iloc[:, item + 1])

n_update_cycles = 2000
rounded_update_cycle = int((n_update_cycles / 100) + 1)

n_word_type_conditions = 4

"""Testing out the model"""

weights = pd.read_csv(
    'C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access Model\\WMP\\Semantic represenation '
    'simulations\\weights_512epoch_noise.csv')
weights = np.array(weights.iloc[:, 1:])

test_epochs = 100
# timecourse = np.zeros([n_items, rounded_update_cycle, test_epochs])
# meaning_timecourse = np.zeros([n_items, 10, test_epochs])
correct_items = np.zeros([n_items, test_epochs])
# meaning_selected = np.zeros([n_items, test_epochs])
settled_output = np.zeros([n_items, n_sem_units, test_epochs])

# for repeat_test in range(0, test_epochs):
#     print("for Repeat Test: ", repeat_test)
#
#     item_numbers = np.random.permutation(n_items)
#
#     for i, item in enumerate(item_numbers):
#
#         activation = np.zeros([1, n_units])
#         activation[:, 0:n_orth_units] = orth[item, :]
#
#         for update_cycle in range(0, n_update_cycles):
#
#             unit = np.random.randint(n_orth_units, n_units)
#
#             net_input = np.dot(activation[:, :], weights[:, unit])
#
#             # Weights test update
#             if net_input > 0.5:
#                 activation[:, unit] = 1
#             else:
#                 activation[:, unit] = 0
#
#             if update_cycle % 99 == 0:
#                 n_active_sem_units = activation[:, n_orth_units:n_units].sum()
#                 rounded_update_cycle = int(update_cycle / 99)
#                 timecourse[item, rounded_update_cycle, repeat_test] = n_active_sem_units
#
#                 # Calculating the number of features activated for each meaning. This is computed as the hamming distance between
#                 # the original semantic pattern for a given semantic patter and the activation pattern retreived from the test.
#         n_features_meaning_1 = np.multiply(sem_meaning_1[item, :], activation[:, n_orth_units:n_units]).sum()
#         n_features_meaning_2 = np.multiply(sem_meaning_2[item, :], activation[:, n_orth_units:n_units]).sum()
#
#         n_sem_active_1 = sem_meaning_1[item, :].sum()
#         n_sem_active_2 = sem_meaning_2[item, :].sum()
#         # Check to see if 75% of the units are active for either of the meanings.
#         activation_threshold_1 = (n_sem_active_1 * 0.75)
#         activation_threshold_2 = (n_sem_active_2 * 0.75)
#
#         if (n_features_meaning_1 > activation_threshold_1) or (n_features_meaning_2 > activation_threshold_2):
#             correct_items[item, repeat_test] = 1
#             if n_features_meaning_1 > n_features_meaning_2:
#                 meaning_selected[item, repeat_test] = 1
#             elif n_features_meaning_2 > n_features_meaning_1:
#                 meaning_selected[item, repeat_test] = 2
#             else:
#                 meaning_selected[item, repeat_test] = 0
#         # breakpoint()
#         settled_output[item, :, repeat_test] = activation[:, n_orth_units:n_units]

### Plotting average timecourse for the model to settle for each word meaning condition.

meaning_selected = np.array(pd.read_csv('test_meaning_selected.csv').iloc[:, 1:])

average_timecourse = np.zeros([rounded_update_cycle, n_word_type_conditions])

idx_cond_1 = int(n_items/4)
idx_cond_2 = int(n_items/2)
idx_cond_3 = int(n_items*(3/4))

mean_per_word = [item.mean() for item in meaning_selected[item, :]]

breakpoint()

# for i in range(0, rounded_update_cycle):
#     average_timecourse[i, 0] = timecourse[0:idx_cond_1, i, :].mean()
#     average_timecourse[i, 1] = timecourse[idx_cond_1:idx_cond_2, i, :].mean()
#     average_timecourse[i, 2] = timecourse[idx_cond_2:idx_cond_3, i, :].mean()
#     average_timecourse[i, 3] = timecourse[idx_cond_3:n_items, i, :].mean()
#
# ####### Plotting Timecourse #######
# # set up the variables to be plotted.
# x = []
# for i in range(0, 20):
#     x.append(i + 1)
# # Getting errors to plot in errorbar.
# yerr = np.zeros([rounded_update_cycle, n_word_type_conditions])
# breakpoint()
# for i in range(0, 20):
#     yerr[i, 0] = (np.divide(average_timecourse[i, 0], np.sqrt(idx_cond_1)))
#     yerr[i, 1] = (np.divide(average_timecourse[i, 1], np.sqrt(idx_cond_1)))
#     yerr[i, 2] = (np.divide(average_timecourse[i, 2], np.sqrt(idx_cond_1)))
#     yerr[i, 3] = (np.divide(average_timecourse[i, 3], np.sqrt(idx_cond_1)))
#
# yerr1 = stats.sem(yerr[:, 0])
# yerr2 = stats.sem(yerr[:, 1])
# yerr3 = stats.sem(yerr[:, 2])
# yerr4 = stats.sem(yerr[:, 3])


# """Collection of Test Outputs"""
# test_meaning_settled = pd.DataFrame(meaning_selected)
# test_meaning_settled.to_csv('test_meaning_selected.csv')
# test_timecourse = pd.DataFrame(timecourse)
# test_timecourse.to_csv('test_timecourse.csv')

"""###########################################################"""

# fig, ax1 = plt.subplots(1, 1)
#
# ax1.errorbar(x, average_timecourse[:, 0], yerr=yerr1)
# ax1.errorbar(x, average_timecourse[:, 1], yerr=yerr2)
# ax1.errorbar(x, average_timecourse[:, 2], yerr=yerr3)
# ax1.errorbar(x, average_timecourse[:, 3], yerr=yerr4)
# ax1.set_xlabel('Update Cycle (x100)')
# ax1.set_ylabel('N Semantic Features')
# ax1.legend(('Non-ambiguous', '90-10', '75-25', '50-50'), loc='lower right')
# breakpoint()
#### Meaning selected ######
meaning_selected_array = meaning_selected

ninety_meaning_selected = (m.fsum(i == 1 for i in meaning_selected_array[0:idx_cond_1, :].reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100
ten_meaning_selected = (m.fsum(i == 2 for i in meaning_selected_array[0:idx_cond_1, :].reshape(idx_cond_1*test_epochs)) / idx_cond_1*test_epochs) * 100
ninety_ten_unsettled = (m.fsum(i == 0 for i in meaning_selected_array[0:idx_cond_1, :].reshape(idx_cond_1*test_epochs)) / idx_cond_1*test_epochs) * 100

seventyfive_meaning_selected = (m.fsum(i == 1 for i in meaning_selected_array[idx_cond_1:idx_cond_2, :].
                                       reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100
twentyfive_meaning_selected = (m.fsum(i == 2 for i in meaning_selected_array[idx_cond_1:idx_cond_2, :].
                                      reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100
seventyfive_twentyfive_unsettled = (m.fsum(i == 0 for i in meaning_selected_array[idx_cond_1:idx_cond_2, :].
                                           reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100

sixty_meaning_selected = (m.fsum(i == 1 for i in meaning_selected_array[idx_cond_2:idx_cond_3]
                                 .reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100
forty_meaning_selected = (m.fsum(i == 2 for i in meaning_selected_array[idx_cond_2:idx_cond_3]
                                 .reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100
sixtyforty_unsetteled = (m.fsum(i == 0 for i in meaning_selected_array[idx_cond_2:idx_cond_3]
                                .reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100


fifty_first_meaning_selected = (m.fsum(i == 1 for i in meaning_selected_array[idx_cond_3:n_items, :]
                                       .reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100
fifty_second_meaning_selected = (m.fsum(i == 2 for i in meaning_selected_array[idx_cond_3:n_items, :]
                                        .reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100
fifty_fifty_unsettled = (m.fsum(i == 0 for i in meaning_selected_array[idx_cond_3:n_items, :]
                                .reshape(idx_cond_1*test_epochs))/idx_cond_1*test_epochs) * 100

breakpoint()
#### Group bar chart prep ###
meaning_one_selected = [ninety_meaning_selected, seventyfive_meaning_selected, sixty_meaning_selected, fifty_first_meaning_selected]
meaning_two_selected = [ten_meaning_selected, twentyfive_meaning_selected, forty_meaning_selected, fifty_second_meaning_selected]
error_settling = [ninety_ten_unsettled, seventyfive_twentyfive_unsettled, sixtyforty_unsetteled, fifty_fifty_unsettled]

labels = ['Ninety-ten', 'Seventyfive-twentyfive','sixty-forty', 'fifty-fifty']
x = np.arange(len(labels))
width = 0.25

# breakpoint()
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

breakpoint()
