""" Experiment runner for the model with knowledge graph attached to interaction data """
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import time
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import sys
import json

from preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
    load_data_monti, load_official_trainvaltest_split, normalize_features
from model import RecommenderGAE, RecommenderSideInfoGAE
from utils import construct_feed_dict, get_averaged_weights

# Set random seed
# seed = 123 # use only for unit testing
seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)

# python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --testing
# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="ml_100k",
                choices=['ml_100k', 'ml_1m', 'ml_10m', 'douban', 'yahoo_music'])

ap.add_argument("-lr", "--learning_rate", type=float, default=0.01)

# ap.add_argument("-e", "--epochs", type=int, default=2500)
ap.add_argument("-e", "--epochs", type=int, default=200)

ap.add_argument("-hi", "--hidden", type=int, nargs=2, default=[500, 75],
                help="Number hidden units in 1st and 2nd layer")

ap.add_argument("-fhi", "--feat_hidden", type=int, default=64,
                help="Number hidden units in the dense layer for features")

ap.add_argument("-ac", "--accumulation", type=str, default="sum", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")

ap.add_argument("-do", "--dropout", type=float, default=0.7,
                help="Dropout fraction")

ap.add_argument("-nb", "--num_basis_functions", type=int, default=2,
                help="Number of basis functions for Mixture Model GCN.")

ap.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="""Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324).
                     Only used for ml_1m and ml_10m datasets. """)

ap.add_argument("-sdir", "--summaries_dir", type=str, default='logs/' + str(datetime.datetime.now()).replace(' ', '_'),
                help="Directory for saving tensorflow summaries.")




# Boolean flags
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-nsym', '--norm_symmetric', dest='norm_symmetric',
                help="Option to turn on symmetric global normalization", action='store_true')

fp.add_argument('-nleft', '--norm_left', dest='norm_symmetric',
                help="Option to turn on left global normalization", action='store_false')
ap.set_defaults(norm_symmetric=True)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-f', '--features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_true')
fp.add_argument('-no_f', '--no_features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_false')
ap.set_defaults(features=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-ws', '--write_summary', dest='write_summary',
                help="Option to turn on summary writing", action='store_true')
fp.add_argument('-no_ws', '--no_write_summary', dest='write_summary',
                help="Option to turn off summary writing", action='store_false')
ap.set_defaults(write_summary=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-t', '--testing', dest='testing',
                help="Option to turn on test set evaluation", action='store_true')
fp.add_argument('-v', '--validation', dest='testing',
                help="Option to only use validation set evaluation", action='store_false')
ap.set_defaults(testing=False)


args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')





# Define parameters
DATASET = args['dataset']
DATASEED = args['data_seed']
NB_EPOCH = args['epochs']
DO = args['dropout']
HIDDEN = args['hidden']
FEATHIDDEN = args['feat_hidden']
BASES = args['num_basis_functions']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = args['features']
SYM = args['norm_symmetric']
TESTING = args['testing']
ACCUM = args['accumulation']

SELFCONNECTIONS = False
SPLITFROMFILE = True
VERBOSE = True

if DATASET == 'ml_1m' or DATASET == 'ml_100k' or DATASET == 'douban':
    NUMCLASSES = 5
elif DATASET == 'ml_10m':
    NUMCLASSES = 10
    print('\n WARNING: this might run out of RAM, consider using train_minibatch.py for dataset %s' % DATASET)
    print('If you want to proceed with this option anyway, uncomment this.\n')
    sys.exit(1)
elif DATASET == 'yahoo_music':
    NUMCLASSES = 71
    if ACCUM == 'sum':
        print('\n WARNING: combining DATASET=%s with ACCUM=%s can cause memory issues due to large number of classes.')
        print('Consider using "--accum stack" as an option for this dataset.')
        print('If you want to proceed with this option anyway, uncomment this.\n')
        sys.exit(1)



if DATASET == 'ml_1m' or DATASET == 'ml_10m':
    if FEATURES:
        datasplit_path = 'data/' + DATASET + '/withfeatures_split_seed' + str(DATASEED) + '.pickle'
    else:
        datasplit_path = 'data/' + DATASET + '/split_seed' + str(DATASEED) + '.pickle'
elif FEATURES:
    datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
else:
    datasplit_path = 'data/' + DATASET + '/nofeatures.pickle'
    print("============datasplit_path:"+datasplit_path)


#  duqu     DATASET == 'ml_100k':
print('-------------------DATASET == ml_100k')
u_features1, v_features1, adj_train1, train_labels1, train_u_indices1, train_v_indices1, \
        val_labels1, val_u_indices1, val_v_indices1, test_labels1, \
        test_u_indices1, test_v_indices1, class_values1 = load_official_trainvaltest_split(DATASET, TESTING, 1)

u_features2, v_features2, adj_train2, train_labels2, train_u_indices2, train_v_indices2, \
        val_labels2, val_u_indices2, val_v_indices2, test_labels2, \
        test_u_indices2, test_v_indices2, class_values2 = load_official_trainvaltest_split(DATASET, TESTING, 2)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")



num_users1, num_items1 = adj_train1.shape
num_users2, num_items2 = adj_train1.shape



num_side_features = 0

# feature loading
if not FEATURES:
    print("========not FEATURES")
    u_features1 = sp.identity(num_users1, format='csr')
    v_features1 = sp.identity(num_items1, format='csr')

    u_features2 = sp.identity(num_users2, format='csr')
    v_features2 = sp.identity(num_items2, format='csr')

    u_features1, v_features1 = preprocess_user_item_features(u_features1, v_features1)
    u_features2, v_features2 = preprocess_user_item_features(u_features2, v_features2)


# global normalization
support1 = []
support_t1 = []
support2 = []
support_t2 = []
adj_train_int1 = sp.csr_matrix(adj_train1, dtype=np.int32)
adj_train_int2 = sp.csr_matrix(adj_train2, dtype=np.int32)


for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    support_unnormalized1 = sp.csr_matrix(adj_train_int1 == i + 1, dtype=np.float32)
    support_unnormalized2 = sp.csr_matrix(adj_train_int2 == i + 1, dtype=np.float32)


    if support_unnormalized1.nnz == 0 and DATASET != 'yahoo_music':
        # yahoo music has dataset split with not all ratings types present in training set.
        # this produces empty adjacency matrices for these ratings.
        sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

    support_unnormalized_transpose1 = support_unnormalized1.T
    support1.append(support_unnormalized1)
    support_t1.append(support_unnormalized_transpose1)

    support_unnormalized_transpose2 = support_unnormalized2.T
    support2.append(support_unnormalized2)
    support_t2.append(support_unnormalized_transpose2)


support1 = globally_normalize_bipartite_adjacency(support1, symmetric=SYM)
support_t1 = globally_normalize_bipartite_adjacency(support_t1, symmetric=SYM)
support2 = globally_normalize_bipartite_adjacency(support2, symmetric=SYM)
support_t2 = globally_normalize_bipartite_adjacency(support_t2, symmetric=SYM)

if SELFCONNECTIONS:
    support1.append(sp.identity(u_features1.shape[0], format='csr'))
    support_t1.append(sp.identity(v_features1.shape[0], format='csr'))
    support2.append(sp.identity(u_features2.shape[0], format='csr'))
    support_t2.append(sp.identity(v_features2.shape[0], format='csr'))

num_support1 = len(support1)
support1 = sp.hstack(support1, format='csr')
support_t1 = sp.hstack(support_t1, format='csr')
num_support2 = len(support2)
support2 = sp.hstack(support2, format='csr')
support_t2 = sp.hstack(support_t2, format='csr')






# Collect all user and item nodes for test set
test_u1 = list(set(test_u_indices1))
test_v1 = list(set(test_v_indices1))
test_u2 = list(set(test_u_indices2))
test_v2 = list(set(test_v_indices2))

test_u_dict1 = {n: i for i, n in enumerate(test_u1)}
test_v_dict1 = {n: i for i, n in enumerate(test_v1)}
test_u_dict2 = {n: i for i, n in enumerate(test_u2)}
test_v_dict2 = {n: i for i, n in enumerate(test_v2)}

test_u_indices1 = np.array([test_u_dict1[o] for o in test_u_indices1])
test_v_indices1 = np.array([test_v_dict1[o] for o in test_v_indices1])
test_u_indices2 = np.array([test_u_dict2[o] for o in test_u_indices2])
test_v_indices2 = np.array([test_v_dict2[o] for o in test_v_indices2])




test_support1 = support1[np.array(test_u1)]
test_support_t1 = support_t1[np.array(test_v1)]
test_support2 = support2[np.array(test_u2)]
test_support_t2 = support_t2[np.array(test_v2)]

# Collect all user and item nodes for validation set
val_u1 = list(set(val_u_indices1))
val_v1 = list(set(val_v_indices1))
val_u2 = list(set(val_u_indices2))
val_v2 = list(set(val_v_indices2))

val_u_dict1 = {n: i for i, n in enumerate(val_u1)}
val_v_dict1 = {n: i for i, n in enumerate(val_v1)}
val_u_dict2 = {n: i for i, n in enumerate(val_u2)}
val_v_dict2 = {n: i for i, n in enumerate(val_v2)}

val_u_indices1 = np.array([val_u_dict1[o] for o in val_u_indices1])
val_v_indices1 = np.array([val_v_dict1[o] for o in val_v_indices1])
val_u_indices2 = np.array([val_u_dict2[o] for o in val_u_indices2])
val_v_indices2 = np.array([val_v_dict2[o] for o in val_v_indices2])


val_support1 = support1[np.array(val_u1)]
val_support_t1 = support_t1[np.array(val_v1)]
val_support2 = support2[np.array(val_u2)]
val_support_t2 = support_t2[np.array(val_v2)]


# Collect all user and item nodes for train set
train_u1 = list(set(train_u_indices1))
train_v1 = list(set(train_v_indices1))
train_u2 = list(set(train_u_indices2))
train_v2 = list(set(train_v_indices2))

train_u_dict1 = {n: i for i, n in enumerate(train_u1)}
train_v_dict1 = {n: i for i, n in enumerate(train_v1)}
train_u_dict2 = {n: i for i, n in enumerate(train_u2)}
train_v_dict2 = {n: i for i, n in enumerate(train_v2)}

train_u_indices1 = np.array([train_u_dict1[o] for o in train_u_indices1])
train_v_indices1 = np.array([train_v_dict1[o] for o in train_v_indices1])
train_u_indices2 = np.array([train_u_dict2[o] for o in train_u_indices2])
train_v_indices2 = np.array([train_v_dict2[o] for o in train_v_indices2])


train_support1 = support1[np.array(train_u1)]
train_support_t1 = support_t1[np.array(train_v1)]
train_support2 = support2[np.array(train_u2)]
train_support_t2 = support_t2[np.array(train_v2)]

# features as side info
print("=======no features")
test_u_features_side1 = None
test_v_features_side1 = None
val_u_features_side1 = None
val_v_features_side1 = None
train_u_features_side1 = None
train_v_features_side1 = None

test_u_features_side2 = None
test_v_features_side2 = None
val_u_features_side2 = None
val_v_features_side2 = None
train_u_features_side2 = None
train_v_features_side2 = None
# -----------------------------------------

placeholders1 = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features1.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features1.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),

    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values1.shape),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

placeholders2 = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features2.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features2.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),

    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values2.shape),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

# create model

print('=======model===== no  features')
model_1 = RecommenderGAE(placeholders1,
                        input_dim=u_features1.shape[1],
                        num_classes=NUMCLASSES,
                        num_support=num_support1,
                        self_connections=SELFCONNECTIONS,
                        num_basis_functions=BASES,
                        hidden=HIDDEN,
                        num_users=num_users1,
                        num_items=num_items1,
                        accum=ACCUM,
                        learning_rate=LR,
                        logging=True)


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1')
model_2 = RecommenderGAE(placeholders2,
                        input_dim=u_features2.shape[1],
                        num_classes=NUMCLASSES,
                        num_support=num_support2,
                        self_connections=SELFCONNECTIONS,
                        num_basis_functions=BASES,
                        hidden=HIDDEN,
                        num_users=num_users2,
                        num_items=num_items2,
                        accum=ACCUM,
                        learning_rate=LR,
                        logging=True)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2')




# Convert sparse placeholders to tuples to construct feed_dict
test_support1 = sparse_to_tuple(test_support1)
test_support_t1 = sparse_to_tuple(test_support_t1)
test_support2 = sparse_to_tuple(test_support2)
test_support_t2 = sparse_to_tuple(test_support_t2)

val_support1 = sparse_to_tuple(val_support1)
val_support_t1 = sparse_to_tuple(val_support_t1)
val_support2 = sparse_to_tuple(val_support2)
val_support_t2 = sparse_to_tuple(val_support_t2)

train_support1 = sparse_to_tuple(train_support1)
train_support_t1 = sparse_to_tuple(train_support_t1)
train_support2 = sparse_to_tuple(train_support2)
train_support_t2 = sparse_to_tuple(train_support_t2)

u_features1 = sparse_to_tuple(u_features1)
v_features1 = sparse_to_tuple(v_features1)
u_features2 = sparse_to_tuple(u_features2)
v_features2 = sparse_to_tuple(v_features2)





assert u_features1[2][1] == v_features1[2][1], 'Number of features of users and items must be the same!'
assert u_features2[2][1] == v_features2[2][1], 'Number of features of users and items must be the same!'

num_features1 = u_features1[2][1]
u_features_nonzero1 = u_features1[1].shape[0]
v_features_nonzero1 = v_features1[1].shape[0]
num_features2 = u_features2[2][1]
u_features_nonzero2 = u_features2[1].shape[0]
v_features_nonzero2 = v_features2[1].shape[0]

# Feed_dicts for validation and test set stay constant over different update steps
#validation  and test de Feed_dicts     zai butong genxin zhong  baochi  bubian
train_feed_dict1 = construct_feed_dict(placeholders1, u_features1, v_features1, u_features_nonzero1,
                                      v_features_nonzero1, train_support1, train_support_t1,
                                      train_labels1, train_u_indices1, train_v_indices1, class_values1, DO,
                                      train_u_features_side1, train_v_features_side1)
train_feed_dict2 = construct_feed_dict(placeholders2, u_features2, v_features2, u_features_nonzero2,
                                      v_features_nonzero2, train_support2, train_support_t2,
                                      train_labels2, train_u_indices2, train_v_indices2, class_values2, DO,
                                      train_u_features_side2, train_v_features_side2)
# No dropout for validation and test runs
val_feed_dict1 = construct_feed_dict(placeholders1, u_features1, v_features1, u_features_nonzero1,
                                    v_features_nonzero1, val_support1, val_support_t1,
                                    val_labels1, val_u_indices1, val_v_indices1, class_values1, 0.,
                                    val_u_features_side1, val_v_features_side1)
val_feed_dict2 = construct_feed_dict(placeholders2, u_features2, v_features2, u_features_nonzero2,
                                    v_features_nonzero2, val_support2, val_support_t2,
                                    val_labels2, val_u_indices2, val_v_indices2, class_values2, 0.,
                                    val_u_features_side2, val_v_features_side2)

test_feed_dict1 = construct_feed_dict(placeholders1, u_features1, v_features1, u_features_nonzero1,
                                     v_features_nonzero1, test_support1, test_support_t1,
                                     test_labels1, test_u_indices1, test_v_indices1, class_values1, 0.,
                                     test_u_features_side1, test_v_features_side1)
test_feed_dict2 = construct_feed_dict(placeholders2, u_features2, v_features2, u_features_nonzero2,
                                     v_features_nonzero2, test_support2, test_support_t2,
                                     test_labels2, test_u_indices2, test_v_indices2, class_values2, 0.,
                                     test_u_features_side2, test_v_features_side2)

# Collect all variables to be logged into summary
merged_summary = tf.summary.merge_all()

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

sess_1 = tf.Session()
sess_1.run(tf.global_variables_initializer())

sess_2 = tf.Session()
sess_2.run(tf.global_variables_initializer())

if WRITESUMMARY:
    train_summary_writer_1 = tf.summary.FileWriter(SUMMARIESDIR + '/train', sess_1.graph)
    train_summary_writer_2 = tf.summary.FileWriter(SUMMARIESDIR + '/train', sess_2.graph)

    val_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/val')
else:
    train_summary_writer = None
    val_summary_writer = None

best_val_score_1 = np.inf
best_val_score_2 = np.inf

best_val_loss_1 = np.inf
best_val_loss_2 = np.inf

best_epoch_1 = 0
best_epoch_2 = 0

wait = 0

print('Training...')
for epoch in range(NB_EPOCH):

    t = time.time()

    # Run single weight update
    # outs = sess.run([model.opt_op, model.loss, model.rmse], feed_dict=train_feed_dict)
    # with exponential moving averages


    outs_1 = sess_1.run([model_1.training_op, model_1.loss, model_1.rmse], feed_dict=train_feed_dict1)
    outs_2 = sess_2.run([model_2.training_op, model_2.loss, model_2.rmse], feed_dict=train_feed_dict2)

    train_avg_loss_1 = outs_1[1]
    train_rmse_1 = outs_1[2]
    train_avg_loss_2 = outs_2[1]
    train_rmse_2 = outs_2[2]

    val_avg_loss_1, val_rmse_1 = sess_1.run([model_1.loss, model_1.rmse], feed_dict=val_feed_dict1)
    val_avg_loss_2, val_rmse_2 = sess_2.run([model_2.loss, model_2.rmse], feed_dict=val_feed_dict2)

    if VERBOSE:
        print("[*] Epoch:", '%04d' % (epoch + 1),
              "train_loss_1=", "{:.5f}".format(train_avg_loss_1),
              "train_rmse_1=", "{:.5f}".format(train_rmse_1),
              "val_loss_1=", "{:.5f}".format(val_avg_loss_1),
              "val_rmse_1=", "{:.5f}".format(val_rmse_1),
              '\n',
              "train_loss_2=", "{:.5f}".format(train_avg_loss_2),
              "train_rmse_2=", "{:.5f}".format(train_rmse_2),
              "val_loss_2=", "{:.5f}".format(val_avg_loss_2),
              "val_rmse_2=", "{:.5f}".format(val_rmse_2),
              "\t\ttime=", "{:.5f}".format(time.time() - t))


    # -----------------------------------------------------------
    # get_averaged_weights(model_1,model_2)

    if val_rmse_1 < best_val_score_1:
        best_val_score_1 = val_rmse_1
        best_epoch_1 = epoch
    if val_rmse_2 < best_val_score_2:
        best_val_score_2 = val_rmse_2
        best_epoch_2 = epoch
print('----------------------------Training  End-----------')




# store model including exponential moving averages
saver = tf.train.Saver()

save_path_1 = saver.save(sess_1, "tmp/%s.ckpt" % model_1.name, global_step=model_1.global_step)
save_path_2 = saver.save(sess_2, "tmp/%s.ckpt" % model_2.name, global_step=model_2.global_step)


if VERBOSE:
    print("\nOptimization Finished!")
    print('best validation score_1 =', best_val_score_1, 'at iteration', best_epoch_1)
    print('best validation score_2 =', best_val_score_2, 'at iteration', best_epoch_2)



if TESTING:
    test_avg_loss_1, test_rmse_1 = sess_1.run([model_1.loss, model_1.rmse], feed_dict=test_feed_dict1)
    test_avg_loss_2, test_rmse_2 = sess_2.run([model_2.loss, model_2.rmse], feed_dict=test_feed_dict2)

    print('test loss1 = ', test_avg_loss_1,'  test loss2 = ', test_avg_loss_2)
    print('test rmse1 = ', test_rmse_1,'  test rmse2 = ', test_rmse_2)

    # restore with polyak averages of parameters
    variables_to_restore_1 = model_1.variable_averages.variables_to_restore()
    variables_to_restore_2 = model_2.variable_averages.variables_to_restore()
    saver1 = tf.train.Saver(variables_to_restore_1)
    saver1.restore(sess_1, save_path_1)
    saver2 = tf.train.Saver(variables_to_restore_2)
    saver2.restore(sess_2, save_path_2)

    test_avg_loss_1, test_rmse_1 = sess_1.run([model_1.loss, model_1.rmse], feed_dict=test_feed_dict1)
    test_avg_loss_2, test_rmse_2 = sess_2.run([model_2.loss, model_2.rmse], feed_dict=test_feed_dict2)
    print('polyak test loss_1 = ', test_avg_loss_1,'  polyak test loss_2 = ', test_avg_loss_2)
    print('polyak test rmse_1 = ', test_rmse_1    ,'  polyak test rmse_2 = ', test_rmse_2)



print('\nSETTINGS:\n')
for key, val in sorted(vars(ap.parse_args()).iteritems()):
    print(key, val)

print('global seed = ', seed)

# For parsing results from file
results = vars(ap.parse_args()).copy()
# results.update({'best_val_score': float(best_val_score_1), 'best_epoch': best_epoch})
print(json.dumps(results))


sess_1.close()
sess_2.close()
