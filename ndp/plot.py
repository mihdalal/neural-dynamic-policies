import os
import numpy as np
import matplotlib.pyplot as plt
prefix = 'data/'
for d in os.listdir(prefix):
    try:
        data = np.load(os.path.join(prefix, d, 'epoch_data.npy'), allow_pickle=True).item()
        plt.plot(data['success_train_det'], label='train')
        plt.plot(data['success_test_det'], label='test')
        plt.legend()
        plt.savefig(os.path.join(prefix, d, 'success.png'))
        plt.clf()
    except:
        pass

