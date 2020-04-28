import time
import datetime
import numpy as np
import scipy
import tensorflow as tf
from sklearn import preprocessing as prep
import pandas as pd


class timer(object):
    def __init__(self, name='default'):
        """
        timer object to record running time of functions, not for micro-benchmarking
        usage is:
            $ timer = utils.timer('name').tic()
            $ timer.toc('process A').tic()


        :param name: label for the timer
        """
        self._start_time = None
        self._name = name
        self.tic()

    def tic(self):
        self._start_time = time.time()
        return self

    def toc(self, message):
        elapsed = time.time() - self._start_time
        message = '' if message is None else message
        print('[{0:s}] {1:s} elapsed [{2:s}]'.format(self._name, message, timer._format(elapsed)))
        return self

    def reset(self):
        self._start_time = None
        return self

    @staticmethod
    def _format(s):
        delta = datetime.timedelta(seconds=s)
        d = datetime.datetime(1, 1, 1) + delta
        s = ''
        if (d.day - 1) > 0:
            s = s + '{:d} days'.format(d.day - 1)
        if d.hour > 0:
            s = s + '{:d} hr'.format(d.hour)
        if d.minute > 0:
            s = s + '{:d} min'.format(d.minute)
        s = s + '{:d} s'.format(d.second)
        return s


def batch(iterable, _n=1, drop=True):
    """
    returns batched version of some iterable
    :param iterable: iterable object as input
    :param _n: batch size
    :param drop: if true, drop extra if batch size does not divide evenly,
        otherwise keep them (last batch might be shorter)
    :return: batched version of iterable
    """
    it_len = len(iterable)
    for ndx in range(0, it_len, _n):
        if ndx + _n < it_len:
            yield iterable[ndx:ndx + _n]
        elif drop is False:
            yield iterable[ndx:it_len]


def tfidf(x):
    """
    compute tfidf of numpy array x
    :param x: input array, document by terms
    :return:
    """
    x_idf = np.log(x.shape[0] - 1) - np.log(1 + np.asarray(np.sum(x > 0, axis=0)).ravel())
    x_idf = np.asarray(x_idf)
    x_idf_diag = scipy.sparse.lil_matrix((len(x_idf), len(x_idf)))
    x_idf_diag.setdiag(x_idf)
    x_tf = x.tocsr()
    x_tf.data = np.log(x_tf.data + 1)
    x_tfidf = x_tf * x_idf_diag
    return x_tfidf


def standardize(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 5 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def standardize_2(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 5 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 1] = 1
    x_scaled[x_scaled < -1] = -1
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def standardize_3(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 5 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_nzrow, :] /= 2.
    x_scaled[x_scaled > 1] = 1
    x_scaled[x_scaled < -1] = -1
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def prep_standardize_dense(x):
    """
    takes dense input and compute standardized version

    Note:
        cap at 5 std

    :param x: 2D numpy data array to standardize (column-wise)
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    scaler = prep.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


idcg_array = np.arange(100) + 1
idcg_array = 1 / np.log2(idcg_array + 1)
idcg_table = np.zeros(100)
for i in range(100):
    idcg_table[i] = np.sum(idcg_array[:(i + 1)])


def batch_eval_recall(_sess, tf_eval, eval_feed_dict, recall_k, eval_data):
    """
    given EvalData and DropoutNet compute graph in TensorFlow, runs batch evaluation

    :param _sess: tf session
    :param tf_eval: the evaluate output symbol in tf
    :param eval_feed_dict: method to parse tf, pick from EvalData method
    :param recall_k: list of thresholds to compute recall at (information retrieval recall)
    :param eval_data: EvalData instance
    :return: recall array at thresholds matching recall_k
    """
    tf_eval_preds_batch = []
    for (batch, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        tf_eval_preds = _sess.run(tf_eval,
                                  feed_dict=eval_feed_dict(
                                      batch, eval_start, eval_stop, eval_data))
        tf_eval_preds_batch.append(tf_eval_preds)
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)
    tf.local_variables_initializer().run()

    # filter non-zero targets
    y_nz = [len(x) > 0 for x in eval_data.R_test_inf.rows]
    y_nz = np.arange(len(eval_data.R_test_inf.rows))[y_nz]

    preds_all = tf_eval_preds[y_nz, :]

    recall = []
    precision = []
    ndcg = []
    for at_k in recall_k:
        preds_k = preds_all[:, :at_k]
        y = eval_data.R_test_inf[y_nz, :]

        x = scipy.sparse.lil_matrix(y.shape)
        x.rows = preds_k
        x.data = np.ones_like(preds_k)

        z = y.multiply(x)
        recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1))))
        precision.append(np.mean(np.sum(z, 1) / at_k))

        x_coo = x.tocoo()
        rows = x_coo.row
        cols = x_coo.col
        y_csr = y.tocsr()
        dcg_array = y_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
        dcg = np.sum(dcg_array * idcg_array[:at_k].reshape((1, -1)), axis=1)
        idcg = np.sum(y, axis=1) - 1
        idcg[np.where(idcg >= at_k)] = at_k-1
        idcg = idcg_table[idcg.astype(int)]
        ndcg.append(np.mean(dcg / idcg))

    return recall, precision, ndcg


def batch_eval_store(_sess, tf_eval, eval_feed_dict, eval_data):
    """
    given EvalData and DropoutNet compute graph in TensorFlow, runs batch evaluation

    :param _sess: tf session
    :param tf_eval: the evaluate output symbol in tf
    :param eval_feed_dict: method to parse tf, pick from EvalData method
    :param recall_k: list of thresholds to compute recall at (information retrieval recall)
    :param eval_data: EvalData instance
    :return: recall array at thresholds matching recall_k
    """
    tf_eval_preds_batch = []
    for (batch, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        tf_eval_preds = _sess.run(tf_eval,
                                  feed_dict=eval_feed_dict(
                                      batch, eval_start, eval_stop, eval_data))
        tf_eval_preds_batch.append(tf_eval_preds)
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)
    tf.local_variables_initializer().run()

    np.save('./data/pred_R.npy', tf_eval_preds)


def negative_sampling(pos_user_array, pos_item_array, neg, item_warm):
    neg = int(neg)
    user_pos = pos_user_array.reshape((-1))
    user_neg = np.tile(pos_user_array, neg).reshape((-1))
    pos = pos_item_array.reshape((-1))
    neg = np.random.choice(item_warm, size=(neg * pos_user_array.shape[0]), replace=True).reshape((-1))
    target_pos = np.ones_like(pos)
    target_neg = np.zeros_like(neg)
    return np.concatenate((user_pos, user_neg)), np.concatenate((pos, neg)), \
           np.concatenate((target_pos, target_neg))


idcg_array = np.arange(100) + 1
idcg_array = 1 / np.log2(idcg_array + 1)
idcg_table = np.zeros(100)
for i in range(100):
    idcg_table[i] = np.sum(idcg_array[:(i + 1)])


def evaluate(_sess, tf_eval, eval_feed_dict, eval_data, like, filters, recall_k, test_file, cold_user=False, test_item_ids=None):
    tf_eval_preds_batch = []
    for (batch, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        tf_eval_preds = _sess.run(tf_eval,
                                  feed_dict=eval_feed_dict(
                                      batch, eval_start, eval_stop, eval_data))
        tf_eval_preds_batch.append(tf_eval_preds)
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)
    tf.local_variables_initializer().run()

    test = pd.read_csv(test_file, dtype=np.int32)

    if not cold_user:
        test_item_ids = list(set(test['iid'].values))

    test_data = test.values.ravel().view(dtype=[('uid', np.int32), ('iid', np.int32)])

    item_old2new_list = np.zeros(np.max(test_item_ids) + 1)
    test_item_ids_map = dict()
    for i, iid in enumerate(test_item_ids):
        test_item_ids_map[iid] = i
        item_old2new_list[iid] = i

    _test_ij_for_inf = [(t[0], t[1]) for t in test_data if t[1] in test_item_ids_map]
    test_user_ids = np.unique(test_data['uid'])

    user_old2new_list = np.zeros(np.max(test_user_ids) + 1)
    test_user_ids_map = dict()
    for i, uid in enumerate(test_user_ids):
        test_user_ids_map[uid] = i
        user_old2new_list[uid] = i

    _test_i_for_inf = [test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
    _test_j_for_inf = [test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]
    R_test_inf = scipy.sparse.coo_matrix(
        (np.ones(len(_test_i_for_inf)),
         (_test_i_for_inf, _test_j_for_inf)),
        shape=[len(test_user_ids), len(test_item_ids)]
    ).tolil(copy=False)

    # filter non-zero targets
    y_nz = [len(x) > 0 for x in R_test_inf.rows]
    y_nz = np.arange(len(R_test_inf.rows))[y_nz]

    preds_all = tf_eval_preds[y_nz, :]

    recall = []
    precision = []
    ndcg = []
    for at_k in recall_k:
        preds_k = preds_all[:, :at_k]
        y = R_test_inf[y_nz, :]

        x = scipy.sparse.lil_matrix(y.shape)
        x.rows = preds_k
        x.data = np.ones_like(preds_k)

        z = y.multiply(x)
        recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1))))
        precision.append(np.mean(np.sum(z, 1) / at_k))

        x_coo = x.tocoo()
        rows = x_coo.row
        cols = x_coo.col
        y_csr = y.tocsr()
        dcg_array = y_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
        dcg = np.sum(dcg_array * idcg_array[:at_k].reshape((1, -1)), axis=1)
        idcg = np.sum(y, axis=1) - 1
        idcg[np.where(idcg >= at_k)] = at_k - 1
        idcg = idcg_table[idcg.astype(int)]
        ndcg.append(np.mean(dcg / idcg))

    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[
        0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[
        1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[
        2] == 0 else 0
    f_score = [f_measure_1, f_measure_5, f_measure_10]

    print('\t\t' + '\t '.join([('@' + str(i)).ljust(6) for i in recall_k]))
    print('recall\t\t%s' % (
        ' '.join(['%.6f' % i for i in recall]),
    ))
    print('precision\t%s' % (
        ' '.join(['%.6f' % i for i in precision]),
    ))
    print('F1 score\t%s' % (
        ' '.join(['%.6f' % i for i in f_score]),
    ))
    print('NDCG\t\t%s' % (
        ' '.join(['%.6f' % i for i in ndcg]),
    ))

    return precision, recall, f_score, ndcg
