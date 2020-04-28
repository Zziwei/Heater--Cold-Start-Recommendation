import tensorflow as tf
import numpy as np


def l2_norm(para):
    return tf.reduce_sum(tf.square(para))


def dense_batch_fc_tanh(x, units, is_training, scope, do_norm=False):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init)
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        if do_norm:
            h2 = tf.contrib.layers.batch_norm(
                h1,
                decay=0.9,
                center=True,
                scale=True,
                is_training=is_training,
                scope=scope + '_bn')
            return tf.nn.tanh(h2, scope + '_tanh'), l2_norm(h1_w) + l2_norm(h1_b)
        else:
            return tf.nn.tanh(h1, scope + '_tanh'), l2_norm(h1_w) + l2_norm(h1_b)


def dense_fc(x, units, scope):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init)
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        return h1, l2_norm(h1_w) + l2_norm(h1_b)


class Heater:
    def __init__(self, latent_rank_in, user_content_rank, item_content_rank,
                 model_select, rank_out, reg, alpha, dim):

        self.rank_in = latent_rank_in  # input embedding dimension
        self.phi_u_dim = user_content_rank  # user content dimension
        self.phi_v_dim = item_content_rank  # item content dimension
        self.model_select = model_select  # model architecture
        self.rank_out = rank_out  # output dimension
        self.reg = reg
        self.alpha = alpha
        self.dim = dim

        # inputs
        self.Uin = None  # input user embedding
        self.Vin = None  # input item embedding
        self.Ucontent = None  # input user content
        self.Vcontent = None  # input item content
        self.is_training = None
        self.target = None  # input training target

        self.eval_trainR = None  # input training rating matrix for evaluation
        self.U_pref_tf = None
        self.V_pref_tf = None
        self.rand_target_ui = None

        # outputs in the model
        self.preds = None  # output of the model, the predicted scores
        self.optimizer = None  # the optimizer
        self.loss = None

        self.U_embedding = None  # new user embedding
        self.V_embedding = None  # new item embedding

        self.lr_placeholder = None  # learning rate

        # predictor
        self.tf_topk_vals = None
        self.tf_topk_inds = None
        self.preds_random = None
        self.tf_latent_topk_cold = None
        self.tf_latent_topk_warm = None
        self.eval_preds_warm = None  # the top-k predicted indices for warm evaluation
        self.eval_preds_cold = None  # the top-k predicted indices for cold evaluation

    def build_model(self):
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[], name='learn_rate')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')

        self.Uin = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='U_in_raw')
        self.Vin = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='V_in_raw')

        dim = self.dim
        self.reg_loss = 0.

        if self.phi_v_dim > 0:
            self.Vcontent = tf.placeholder(tf.float32, shape=[None, self.phi_v_dim], name='V_content')
            self.dropout_item_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='dropout_item_indicator')

            vcontent_gate, vcontent_gate_reg = dense_fc(self.Vcontent, dim,
                                                        'vcontent_gate_layer')  # size: batch_size X dim
            vcontent_gate = tf.nn.tanh(vcontent_gate)

            self.reg_loss += vcontent_gate_reg

            vcontent_expert_list = []
            for i in range(dim):
                tmp_expert = self.Vcontent
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'Vexpert_' + str(ihid) + '_' + str(i))
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                vcontent_expert_list.append(tf.reshape(tmp_expert, [-1, 1, self.rank_out]))

            vcontent_expert_concat = tf.concat(vcontent_expert_list, 1)  # size: batch_size X dim X self.rank_out

            vcontent_expert_concat = tf.linalg.matmul(tf.reshape(vcontent_gate, [-1, 1, dim]),
                                                      vcontent_expert_concat)
            Vcontent_last = tf.reshape(tf.nn.tanh(vcontent_expert_concat), [-1, self.rank_out])  # size: batch_size X self.rank_out

            self.Vin_filter = 1 - self.dropout_item_indicator

            diff_item_loss = self.alpha \
                             * (tf.reduce_sum(tf.reduce_sum(tf.square(Vcontent_last - self.Vin),
                                                            axis=1, keepdims=True)))
            v_last = (self.Vin * self.Vin_filter + Vcontent_last * (1 - self.Vin_filter))
        else:
            v_last = self.Vin
            diff_item_loss = 0

        if self.phi_u_dim > 0:
            self.Ucontent = tf.placeholder(tf.float32, shape=[None, self.phi_u_dim], name='U_content')
            self.dropout_user_indicator = tf.placeholder(tf.float32, shape=[None, 1], name='dropout_user_indicator')

            ucontent_gate, ucontent_gate_reg = dense_fc(self.Ucontent, dim,
                                                        'ucontent_gate_layer')  # size: batch_size X dim
            ucontent_gate = tf.nn.tanh(ucontent_gate)

            self.reg_loss += ucontent_gate_reg

            ucontent_expert_list = []
            for i in range(dim):
                tmp_expert = self.Ucontent
                for ihid, hid in enumerate(self.model_select):
                    tmp_expert, tmp_reg = dense_fc(tmp_expert, hid, 'Uexpert_' + str(ihid) + '_' + str(i))
                    tmp_expert = tf.nn.tanh(tmp_expert)
                    self.reg_loss += tmp_reg
                ucontent_expert_list.append(tf.reshape(tmp_expert, [-1, 1, self.rank_out]))

            ucontent_expert_concat = tf.concat(ucontent_expert_list, 1)  # size: batch_size X dim X self.rank_out

            ucontent_expert_concat = tf.linalg.matmul(tf.reshape(ucontent_gate, [-1, 1, dim]),
                                                      ucontent_expert_concat)
            Ucontent_last = tf.reshape(tf.nn.tanh(ucontent_expert_concat), [-1, self.rank_out])  # size: batch_size X self.rank_out

            self.Uin_filter = 1 - self.dropout_user_indicator

            diff_user_loss = self.alpha \
                             * (tf.reduce_sum(tf.reduce_sum(tf.square(Ucontent_last - self.Uin),
                                                            axis=1, keepdims=True)))
            u_last = (self.Uin * self.Uin_filter + Ucontent_last * (1 - self.Uin_filter))
        else:
            u_last = self.Uin
            diff_user_loss = 0

        for ihid, hid in enumerate([self.rank_out]):
            u_last, u_reg = dense_batch_fc_tanh(u_last, hid, self.is_training, 'user_layer_%d'%ihid,
                                                do_norm=True)
            v_last, v_reg = dense_batch_fc_tanh(v_last, hid, self.is_training, 'item_layer_%d'%ihid,
                                                do_norm=True)
            self.reg_loss += u_reg
            self.reg_loss += v_reg

        with tf.variable_scope("U_embedding"):
            u_emb_w = tf.Variable(tf.truncated_normal([u_last.get_shape().as_list()[1], self.rank_out], stddev=0.01),
                                  name='u_emb_w')
            u_emb_b = tf.Variable(tf.zeros([1, self.rank_out]), name='u_emb_b')
            self.U_embedding = tf.matmul(u_last, u_emb_w) + u_emb_b

        with tf.variable_scope("V_embedding"):
            v_emb_w = tf.Variable(tf.truncated_normal([v_last.get_shape().as_list()[1], self.rank_out], stddev=0.01),
                                  name='v_emb_w')
            v_emb_b = tf.Variable(tf.zeros([1, self.rank_out]), name='v_emb_b')
            self.V_embedding = tf.matmul(v_last, v_emb_w) + v_emb_b

        self.reg_loss += (l2_norm(v_emb_w) + l2_norm(v_emb_b) + l2_norm(u_emb_w) + l2_norm(u_emb_b))
        self.reg_loss *= self.reg

        with tf.variable_scope("loss"):
            preds = tf.multiply(self.U_embedding, self.V_embedding)
            self.preds = tf.reduce_sum(preds, 1)  # output of the model, the predicted scores
            self.diff_loss = diff_item_loss + diff_user_loss
            self.loss = tf.reduce_mean(tf.squared_difference(self.preds, self.target)) + self.reg_loss + self.diff_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.optimizer = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss)

    def build_predictor(self, recall_at):
        self.eval_trainR = tf.sparse_placeholder(
            dtype=tf.float32, shape=[None, None], name='trainR_sparse')

        with tf.variable_scope("eval"):
            embedding_prod_cold = tf.matmul(self.U_embedding, self.V_embedding, transpose_b=True, name='pred_all_items')
            embedding_prod_warm = tf.sparse_add(embedding_prod_cold, self.eval_trainR)
            _, self.eval_preds_cold = tf.nn.top_k(embedding_prod_cold, k=recall_at[-1], sorted=True,
                                                  name='topK_net_cold')
            _, self.eval_preds_warm = tf.nn.top_k(embedding_prod_warm, k=recall_at[-1], sorted=True,
                                                  name='topK_net_warm')

    def get_eval_dict(self, _i, _eval_start, _eval_finish, eval_data):
        _eval_dict = {
            self.Uin: eval_data.U_pref_test[_eval_start:_eval_finish, :],
            self.Vin: eval_data.V_pref_test,
            self.is_training: False
        }

        if self.phi_v_dim > 0:
            zero_index = np.where(np.sum(eval_data.V_pref_test, axis=1) == 0)[0]
            dropout_item_indicator = np.zeros((len(eval_data.test_item_ids), 1))
            dropout_item_indicator[zero_index] = 1
            _eval_dict[self.dropout_item_indicator] = dropout_item_indicator
            _eval_dict[self.Vcontent] = eval_data.V_content_test
        if self.phi_u_dim > 0:
            zero_index = np.where(np.sum(eval_data.U_pref_test[_eval_start:_eval_finish, :], axis=1) == 0)[0]
            dropout_user_indicator = np.zeros((_eval_finish - _eval_start, 1))
            dropout_user_indicator[zero_index] = 1
            _eval_dict[self.dropout_user_indicator] = dropout_user_indicator
            _eval_dict[self.Ucontent] = eval_data.U_content_test[_eval_start:_eval_finish, :]
        return _eval_dict

    def get_eval_dict_latent(self, _i, _eval_start, _eval_finish, eval_data, u_pref, v_pref):
        _eval_dict = {
            self.U_pref_tf: u_pref[eval_data.test_user_ids[_eval_start:_eval_finish], :],
            self.V_pref_tf: v_pref[eval_data.test_item_ids, :]
        }
        if not eval_data.is_cold:
            _eval_dict[self.eval_trainR] = eval_data.tf_eval_train[_i]
        return _eval_dict
