import pandas as pd
import numpy as np
import copy
import pickle

train = pd.read_csv('./train_old.csv', delimiter=",", header=-1, dtype=np.int32)
train.drop(2, axis=1, inplace=True)
train[0] = train[0] - 1
train[1] = train[1] - 1
train.rename(columns={0: 'uid', 1: 'iid'}, inplace=True)
train.to_csv('./train.csv', index=False)

test = pd.read_csv('./test_old.csv', delimiter=",", header=-1, dtype=np.int32)
test.drop(2, axis=1, inplace=True)
test[0] = test[0] - 1
test[1] = test[1] - 1
test.rename(columns={0: 'uid', 1: 'iid'}, inplace=True)

train_uids = train['uid'].values
train_iids = train['iid'].values
test_uids = test['uid'].values
test_iids = test['iid'].values
num_user = len(set(train_uids).union(set(test_uids)))
num_item = len(set(train_iids).union(set(test_iids)))
info = {'num_user': num_user, 'num_item': num_item}
with open('./info.pkl', 'wb') as f:
    pickle.dump(info, f)

vali_iids = list(np.random.choice(list(set(test_iids)), int(len(set(test_iids)) * 0.3), replace=False))
test_iids = list(set(test_iids) - set(vali_iids))

test_df = copy.copy(test[test['iid'].isin(test_iids)])
vali_df = copy.copy(test[test['iid'].isin(vali_iids)])
test_df.reset_index(drop=True, inplace=True)
vali_df.reset_index(drop=True, inplace=True)
test_df.to_csv('./test.csv', index=False)
vali_df.to_csv('./vali.csv', index=False)


