import matplotlib.pyplot as plt
import numpy as np
import csv
import statistics
import plotly.graph_objects as go

scores = []
with open('pos_scores.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        scores.append(float(row[0]))
np_scores = np.asarray(scores)
med = statistics.median(scores)
threshold = np.percentile(np_scores, 85)
print('median', med)
filtered_med = np_scores[(np_scores > med)]
print(len(np_scores[(np_scores > med)]))
print(np.sum(np_scores > med))
print(np.argwhere(np_scores > med))
filtered_ninetieth = np_scores[(np_scores > threshold)]
print('len filtered', len(filtered_med))
print('len filtered', len(filtered_ninetieth))
a = np.argwhere(np_scores > med)
b = a.tolist()
print('type of elements', type(b))
print('inside element', type(b[0][0]))

print()
print(filtered_med)
print(filtered_ninetieth)
x1 = np.linspace(0,1,8085)
y1 = scores
plt.style.use('ggplot')
plt.hist(scores, range=(0.8,1), bins = 200)
plt.hist(filtered_med, range=(0.8,1), bins = 200)
plt.hist(filtered_ninetieth, range=(0.8,1), bins = 200)
# plt.show()

import stanza.models.config as config
# config.indices = filtered
# print('ban', config.indices)
# plt.plot(x1, y1)
# plt.show()

# with open('final_scores_similar.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         scores_similar.append(float(row[0]))
#
# print(scores_similar)
# x2 = np.linspace(0,1,3332)
# y2 = scores_similar
# plt.plot(x2, y2)
# # plt.ylim(0,1)
# plt.show()

fig = go.Figure(go.Histogram(
    name= "All scores",
    x=np_scores,
    bingroup=1))

fig.add_trace(go.Histogram(
    name='Median threshold',
    x=filtered_med,
    bingroup=1))

fig.add_trace(go.Histogram(
    name='85th percentile threshold',
    x=filtered_ninetieth,
    bingroup=1))


fig.update_layout(
    barmode="overlay",
    bargap=0.1)

fig.update_layout(
    xaxis_title_text='Sentence scores', # xaxis label
    yaxis_title_text='Frequency', # yaxis label
#     bargap=0.1, # gap between bars of adjacent location coordinates
#     bargroupgap=0.1 # gap between bars of the same location coordinates
)

fig.show()