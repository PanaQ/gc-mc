from __future__ import division
from __future__ import print_function


def construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero, v_features_nonzero,
                        support, support_t, labels, u_indices, v_indices, class_values,
                        dropout, u_features_side=None, v_features_side=None):
    """
    Function that creates feed dictionary when running tensorflow sessions.
    """

    feed_dict = dict()
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})
    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['support_t']: support_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['class_values']: class_values})

    if (u_features_side is not None) and (v_features_side is not None):
        feed_dict.update({placeholders['u_features_side']: u_features_side})
        feed_dict.update({placeholders['v_features_side']: v_features_side})

    return feed_dict



# get_averge
def get_averaged_weights(model_1,model_2):
    print('get_averaged_weights')
    # print(model_1.layers.layer1.weights_u)
    # print(layer2.vars['weights_u'])






    # -----------------------------------------------------------  GCN,Dense,BilineaMixtures
    # print('-----------layer--------')
    # # print(model.layers[0].weights_u)  # print(model.layers[0].weights_v)
    # # [ < tf.Tensor  'recommendergae/split:0' shape = (2551, 100) dtype = float32 >,
    # #   < tf.Tensor 'recommendergae/split:1'  shape = (2551, 100) dtype = float32 >,
    # #   < tf.Tensor 'recommendergae/split:2' shape = (2551, 100)  dtype = float32 >,
    # #   < tf.Tensor 'recommendergae/split:3'  shape = (2551, 100) dtype = float32 >,
    # #   < tf.Tensor 'recommendergae/split:4'  shape = (2551, 100) dtype = float32 >]
    #
    #
    # # print(model.layers[1].vars['weights_u'])
    # print(model.layers[1].vars['weights_u']+model.layers[1].vars['weights_u'])

    # < tf.Variable  'recommendergae/dense_1_vars/weights:0' shape = (500, 75) dtype = float32_ref >






    # print(model.layers[2]._multiply_inputs_weights)

