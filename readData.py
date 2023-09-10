import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# read the dataset using the compression zip
test_df = pd.read_csv('test.csv.zip',compression='zip')
train_df = pd.read_csv('train.csv.zip',compression='zip')
# display dataset
# label = train_df.loc[27000][0]
# print(label)
# img = np.reshape(np.array(train_df.loc[27000][1:]), (28, 28))
# plt.imshow(img)
# print(img)
# plt.show()

print(test_df.values.shape)