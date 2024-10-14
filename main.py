import dataloader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# test init
test_id = 'my_data'
config = dict()
config['mnist'] = {'data': 'mnist',
                   'label': [1, 6, 9],
                   'legend': [1, 6, 9],
                   'method': 'pca',
                   'n_components': 2}

config['my_data'] = {'data': 'my_data',
                     'frame_width': 17,
                     'target': 'long',
                     'label': [0, 1],
                     'legend': ['short is better', 'long is better'],
                     'method': 'pca',
                     'n_components': 5}

data, label, model = None, None, None
label_idx = list()
color_list = ['blue', 'red', 'green', 'orange', 'purple']

# data load
if config[test_id]['data'] == 'mnist':
    dataset = dataloader.MNIST()
    data, label = dataset.load_digits(config[test_id]['label'])
elif config[test_id]['data'] == 'my_data':
    dataset = dataloader.MyData(config[test_id]['frame_width'], config[test_id]['target'])
    data, label= dataset.data_load()

for tag in config[test_id]['label']:
    label_idx.append(np.where(label == tag))

# model fit
if config[test_id]['method'] == 'pca':
    model = PCA(n_components=config[test_id]['n_components'])
elif config[test_id]['method'] == 'tsne':
    model = TSNE(n_components=config[test_id]['n_components'])

res_model = model.fit_transform(data)

for i in range(len(config[test_id]['label'])):
    plt.scatter(res_model[label_idx[i], 0], res_model[label_idx[i], 1], c=color_list[i], s=2, label=config[test_id]['legend'][i], alpha=0.3)

plt.legend(loc='upper right', fontsize=8)
plt.show()
plt.close()
