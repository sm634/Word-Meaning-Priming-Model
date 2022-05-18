import numpy as np
import matplotlib.pyplot as plt
import math as m
import pdb
import pandas as pd

n_models = 100

meanings_settled_per_model = pd.read_csv('unprimed_semantic_patterns.csv')
primed_meanings_settled_per_model = pd.read_csv('Primed_semantic_pattern.csv')

meanings_settled_per_model = meanings_settled_per_model.set_index('Unnamed: 0')
primed_meanings_settled_per_model = primed_meanings_settled_per_model.set_index('Unnamed: 0')

meanings_settled_per_model = np.array(meanings_settled_per_model).reshape((n_models,3,3))
primed_meanings_settled_per_model = np.array(primed_meanings_settled_per_model).reshape((n_models,3,3))

pdb.set_trace()

#### Group bar chart prep ###
meaning_one_selected = np.mean(meanings_settled_per_model[:,0,:],axis=0)
meaning_two_selected = np.mean(meanings_settled_per_model[:,1,:],axis=0)
error_settled = np.mean(meanings_settled_per_model[:,2,:],axis=0)

error_bar_m1 = np.std(meanings_settled_per_model[:,0,:],axis=0)/m.sqrt(n_models)
error_bar_m2 = np.std(meanings_settled_per_model[:,1,:], axis=0)/m.sqrt(n_models)
error_bar_error = np.std(meanings_settled_per_model[:,2,:], axis=0)/m.sqrt(n_models)

primed_meaning_one_selected = np.mean(primed_meanings_settled_per_model[:,0,:],axis=0)
primed_meaning_two_selected = np.mean(primed_meanings_settled_per_model[:,1,:],axis=0)
primed_error_settled = np.mean(primed_meanings_settled_per_model[:,2,:],axis=0)

primed_error_bar_m1 = np.std(primed_meanings_settled_per_model[:,0,:], axis=0)/m.sqrt(n_models)
primed_error_bar_m2 = np.std(primed_meanings_settled_per_model[:,1,:], axis=0)/m.sqrt(n_models)
primed_error_bar_error = np.std(primed_meanings_settled_per_model[:,2,:], axis=0)/m.sqrt(n_models)

labels = ['Ninety-ten', 'Seventyfive-twentyfive','fifty-fifty']
x = np.arange(len(labels))
width = 0.15

## Plotting bar chart to show meaning settlement patterns.
fig2, ax2 = plt.subplots()
meaning_one = ax2.bar(x - 2*width, meaning_one_selected, yerr=error_bar_m1, width=width)
meaning_one_primed = ax2.bar(x - width,primed_meaning_one_selected, yerr=primed_error_bar_m1,width=width)

meaning_two = ax2.bar(x + width, meaning_two_selected, yerr=error_bar_m2, width=width)
meaning_two_primed = ax2.bar(x + 2*width, primed_meaning_two_selected, yerr=primed_error_bar_m2,width=width)

# error = ax2.bar(x + width,error_settled, yerr=error_bar_error,width=width)
# error_primed = ax2.bar(x + 2*width,primed_error_settled, yerr=primed_error_bar_error, width=width)

ax2.set_title('Proportion of trials settled on across semantic-dominance and primed vs un-primed training conditions')
ax2.set_xlabel('Proportion of Training trials split on one of two possible semantic patterns per word')
ax2.set_ylabel('Percentage of test trials')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(['Dominant Semantic Pattern (Un-primed trial)', 'Dominant Semantic Pattern (primed trial)',
            'Subordinate Semantic Pattern (Un-primed trial)', 'Subordinate Semantic Pattern (primed trial)'])
            # ,'error (unprimed)', 'error (primed)'])

fig2.tight_layout()

plt.show()

