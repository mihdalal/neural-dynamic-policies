import os
import numpy as np
import matplotlib.pyplot as plt

for d in os.listdir('data/'):
    try:
        data = np.load(os.path.join('data/', d, 'epoch_data.npy'), allow_pickle=True).item()
        plt.plot(data['success_train_sample'], label='train')
        plt.plot(data['success_test_sample'], label='test')
        plt.legend()
        plt.savefig(os.path.join('data/', d, 'success.png'))
        plt.clf()
    except:
        pass

