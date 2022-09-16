import numpy as np
import torch
import matplotlib.pyplot as plt

# Load all train data
center_train = np.load('../dataSet/new_train/Center.npy', allow_pickle=True) # 3006
donut_train = np.load('../dataSet/new_train/Donut.npy', allow_pickle=True) # 3112
edge_loc_train = np.load('../dataSet/new_train/Edge_Loc.npy', allow_pickle=True) # 3632
edge_ring_train = np.load('../dataSet/new_train/Edge_Ring.npy', allow_pickle=True) # 6776
loc_train = np.load('../dataSet/new_train/Loc.npy', allow_pickle=True) # 5030
random_train = np.load('../dataSet/new_train/Random.npy', allow_pickle=True) # 2423
scratch_train = np.load('../dataSet/new_train/Scratch.npy', allow_pickle=True) # 3340
near_full_train = np.load('../dataSet/new_train/Near_full.npy', allow_pickle=True) # 832
none_train = np.load('../dataSet/new_train/none.npy', allow_pickle=True) # 7000
none_train = none_train[:14000]
print('center_train:',center_train.shape[0],'donut_train:',donut_train.shape[0],
      'edge_loc_train:',edge_loc_train.shape[0],'edge_ring_train:',edge_ring_train.shape[0],'loc_train:',loc_train.shape[0],
      'random_train:',random_train.shape[0],'scratch_train:',scratch_train.shape[0],
      'near_full_train:',near_full_train.shape[0],'none_train:',none_train.shape[0])

#variables = [Center,Donut,Edge_Loc,Edge_Ring,Loc,Random,Scratch,Near_full,none]

# Load all test data
center_test = np.load('../dataSet/new_test/Center.npy', allow_pickle=True) # 1288
donut_test = np.load('../dataSet/new_test/Donut.npy', allow_pickle=True) # 1328
edge_loc_test = np.load('../dataSet/new_test/Edge_Loc.npy', allow_pickle=True) # 1557
edge_ring_test = np.load('../dataSet/new_test/Edge_Ring.npy', allow_pickle=True) # 2904
loc_test = np.load('../dataSet/new_test/Loc.npy', allow_pickle=True) # 2156
random_test = np.load('../dataSet/new_test/Random.npy', allow_pickle=True) # 1040
scratch_test = np.load('../dataSet/new_test/Scratch.npy', allow_pickle=True) # 1432
near_full_test = np.load('../dataSet/new_test/Near_full.npy', allow_pickle=True) # 360
none_test = np.load('../dataSet/new_test/none.npy', allow_pickle=True) # 3000
none_test = none_test[:6000]
print('center_test:',center_test.shape[0],'donut_test:',donut_test.shape[0],'edge_loc_test:',edge_loc_test.shape[0],
      'edge_ring_test:',edge_ring_test.shape[0],
      'loc_test:',loc_test.shape[0],
      'random_test:',random_test.shape[0],'scratch_test:',scratch_test.shape[0],
      'near_full_test:',near_full_test.shape[0],'none_test:',none_test.shape[0])



# Concatenate all the data together
train_data = np.concatenate((center_train, donut_train, edge_loc_train, edge_ring_train, loc_train, random_train,
                          scratch_train, near_full_train, none_train))
val_data = np.concatenate((center_test, donut_test, edge_loc_test, edge_ring_test, loc_test, random_test,
                          scratch_test, near_full_test, none_test))

print('train_data shape:', train_data.shape)
print('val_data shape:', val_data.shape)

# Shuffle the dataSet
np.random.seed(12321)  # for reproducibility
torch.manual_seed(12321)  # for reproducibility
np.random.shuffle(train_data)
np.random.shuffle(val_data)

# Extract labels
train_labels = np.array([])
for data in train_data:
    train_labels = np.append(train_labels, data[0])

val_labels = np.array([])
for data in val_data:
    val_labels = np.append(val_labels, data[0])

# print(train_labels,train_labels.shape)
# print(train_data[0])
# print(val_labels,val_labels.shape)
# print(val_data[0])

# Delete first element of dataset
train_data = np.delete(train_data, 0, 1)
train_data = np.squeeze(train_data, 1)
train_set = []
for data in train_data:
    train_set.append(data)
train_set = np.array(train_set)
train_set = np.moveaxis(train_set, 3, 1)

print(train_set.shape)
# train_data = train_data.reshape((8000,1,-1))

val_data = np.delete(val_data, 0, 1)
val_data = np.squeeze(val_data, 1)
val_set = []
for data in val_data:
    val_set.append(data)
val_set = np.array(val_set)
val_set = np.moveaxis(val_set, 3, 1)
print(val_set.shape)

# # visualization
# example = train_set[10000]
# plt.imshow(example)
# plt.show()

np.save('../dataSet/new_train_set32.npy', train_set)
np.save('../dataSet/new_test_set32.npy', val_set)

np.save('../dataSet/new_train_labels32.npy', train_labels)
np.save('../dataSet/new_test_labels32.npy', val_labels)