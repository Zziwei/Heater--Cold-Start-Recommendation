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

test_item = pd.read_csv('./test_cold_item_old.csv', delimiter=",", header=-1, dtype=np.int32)
test_item.drop(2, axis=1, inplace=True)
test_item[0] = test_item[0] - 1
test_item[1] = test_item[1] - 1
test_item.rename(columns={0: 'uid', 1: 'iid'}, inplace=True)

test_user = pd.read_csv('./test_cold_user_old.csv', delimiter=",", header=-1, dtype=np.int32)
test_user.drop(2, axis=1, inplace=True)
test_user[0] = test_user[0] - 1
test_user[1] = test_user[1] - 1
test_user.rename(columns={0: 'uid', 1: 'iid'}, inplace=True)

train_uids = train['uid'].values
train_iids = train['iid'].values
test_item_uids = test_item['uid'].values
test_item_iids = test_item['iid'].values
test_user_uids = test_user['uid'].values
test_user_iids = test_user['iid'].values
# num_user = len(set(train_uids).union(set(test_item_uids)).union(set(test_user_uids)))
# num_item = len(set(train_iids).union(set(test_item_iids)).union(set(test_user_iids)))
info = {'num_user': 1497020, 'num_item': 1306054}
with open('./info.pkl', 'wb') as f:
    pickle.dump(info, f)

vali_item_iids = list(np.random.choice(list(set(test_item_iids)), int(len(set(test_item_iids)) * 0.3), replace=False))
test_item_iids = list(set(test_item_iids) - set(vali_item_iids))

vali_user_uids = list(np.random.choice(list(set(test_user_uids)), int(len(set(test_user_uids)) * 0.3), replace=False))
test_user_uids = list(set(test_user_uids) - set(vali_user_uids))

test_item_df = copy.copy(test_item[test_item['iid'].isin(test_item_iids)])
vali_item_df = copy.copy(test_item[test_item['iid'].isin(vali_item_iids)])

test_user_df = copy.copy(test_user[test_user['uid'].isin(test_user_uids)])
vali_user_df = copy.copy(test_user[test_user['uid'].isin(vali_user_uids)])

test_item_df.reset_index(drop=True, inplace=True)
vali_item_df.reset_index(drop=True, inplace=True)
test_user_df.reset_index(drop=True, inplace=True)
vali_user_df.reset_index(drop=True, inplace=True)

test_item_df.to_csv('./test_item.csv', index=False)
vali_item_df.to_csv('./vali_item.csv', index=False)

test_user_df.to_csv('./test_user.csv', index=False)
vali_user_df.to_csv('./vali_user.csv', index=False)


