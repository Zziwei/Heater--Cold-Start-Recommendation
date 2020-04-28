import numpy as np
import tensorflow as tf
import scipy.sparse
import utils
import pandas as pd

"""
This module contains class and methods related to data used in Heater  
"""


def load_eval_data(test_file, cold_user=False, test_item_ids=None):
    timer = utils.timer()
    test = pd.read_csv(test_file, dtype=np.int32)
    if not cold_user:
        test_item_ids = list(set(test['iid'].values))
    test_data = test.values.ravel().view(dtype=[('uid', np.int32), ('iid', np.int32)])

    timer.toc('read %s triplets' % test_data.shape[0]).tic()
    eval_data = EvalData(
        test_data,
        test_item_ids)
    print(eval_data.get_stats_string())
    return eval_data


class EvalData:
    """
    EvalData:
        EvalData packages test triplet (user, item, score) into appropriate formats for evaluation

        Compact Indices:
            Specifically, this builds compact indices and stores mapping between original and compact indices.
            Compact indices only contains:
                1) items in test set
                2) users who interacted with such test items
            These compact indices speed up testing significantly by ignoring irrelevant users or items

        Args:
            test_triplets(int triplets): user-item-interaction_value triplet to build the test data
            train(int triplets): user-item-interaction_value triplet from train data

        Attributes:
            is_cold(boolean): whether test data is used for cold start problem
            test_item_ids(list of int): maps compressed item ids to original item ids (via position)
            test_item_ids_map(dictionary of int->int): maps original item ids to compressed item ids
            test_user_ids(list of int): maps compressed user ids to original user ids (via position)
            test_user_ids_map(dictionary of int->int): maps original user ids to compressed user ids
            R_test_inf(scipy lil matrix): pre-built compressed test matrix
            R_train_inf(scipy lil matrix): pre-built compressed train matrix for testing

            other relevant input/output exposed from tensorflow graph

    """

    def __init__(self, test_triplets, test_item_ids):
        # build map both-ways between compact and original indices
        # compact indices only contains:
        #  1) items in test set
        #  2) users who interacted with such test items

        self.test_item_ids = test_item_ids
        # test_item_ids_map
        self.test_item_ids_map = {iid: i for i, iid in enumerate(self.test_item_ids)}

        _test_ij_for_inf = [(t[0], t[1]) for t in test_triplets if t[1] in self.test_item_ids_map]
        # test_user_ids
        self.test_user_ids = np.unique(test_triplets['uid'])
        # test_user_ids_map
        self.test_user_ids_map = {user_id: i for i, user_id in enumerate(self.test_user_ids)}

        _test_i_for_inf = [self.test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
        _test_j_for_inf = [self.test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]
        self.R_test_inf = scipy.sparse.coo_matrix(
            (np.ones(len(_test_i_for_inf)),
             (_test_i_for_inf, _test_j_for_inf)),
            shape=[len(self.test_user_ids), len(self.test_item_ids)]
        ).tolil(copy=False)

        # allocate fields
        self.U_pref_test = None
        self.V_pref_test = None
        self.V_content_test = None
        self.U_content_test = None
        self.tf_eval_train = None
        self.tf_eval_test = None
        self.eval_batch = None

    def init_tf(self, user_factors, item_factors, user_content, item_content, eval_run_batchsize,
                cold_user=False, cold_item=False):
        self.U_pref_test = user_factors[self.test_user_ids, :]
        self.V_pref_test = item_factors[self.test_item_ids, :]
        if cold_user:
            self.U_content_test = user_content[self.test_user_ids, :]
            if scipy.sparse.issparse(self.U_content_test):
                self.U_content_test = self.U_content_test.todense()
        if cold_item:
            self.V_content_test = item_content[self.test_item_ids, :]
            if scipy.sparse.issparse(self.V_content_test):
                self.V_content_test = self.V_content_test.todense()
        eval_l = self.R_test_inf.shape[0]
        self.eval_batch = [(x, min(x + eval_run_batchsize, eval_l)) for x
                           in range(0, eval_l, eval_run_batchsize)]

        self.tf_eval_train = []
        self.tf_eval_test = []

    def get_stats_string(self):
        return ('\tn_test_users:[%d]\n\tn_test_items:[%d]' % (len(self.test_user_ids), len(self.test_item_ids))
                + '\n\tR_train_inf: %s' % (
                    'no R_train_inf for cold'
                )
                + '\n\tR_test_inf: shape=%s nnz=[%d]' % (
                    str(self.R_test_inf.shape), len(self.R_test_inf.nonzero()[0])
                ))
