# -*- coding: utf-8 -*-
# 分离后端函数，主要是为了同时兼容原生keras和tf.keras
# 通过设置环境变量TF_KERAS=1来切换tf.keras

import os, sys
from distutils.util import strtobool
import numpy as np
import tensorflow as tf

# 判断是tf.keras还是纯keras的标记
# is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

is_tf_keras = True

# 只有 2.0 版本
# if is_tf_keras:
#     import tensorflow.keras as keras
#     import tensorflow.keras.backend as K
#     sys.modules['keras'] = keras
# else:
#     import keras
#     import keras.backend as K


import tensorflow.keras as keras
import tensorflow.keras.backend as K

# 判断是否启用动态图模式，仅在TF 2.x下可用。
# 建议关闭。珍惜生命，远离eager。
is_tf_eager = strtobool(os.environ.get('TF_EAGER', '0'))

if tf.__version__.startswith('2.') and is_tf_keras:
    if not is_tf_eager:
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.experimental.output_all_intermediates(True)


def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    """基于Tanh近似计算的gelu函数
    """
    cdf = 0.5 * (
        1.0 + K.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))
    )
    return x * cdf


def set_gelu(version):
    """设置gelu版本
    """
    version = version.lower()
    assert version in ['erf', 'tanh'], 'gelu version must be erf or tanh'
    if version == 'erf':
        keras.utils.get_custom_objects()['gelu'] = gelu_erf
    else:
        keras.utils.get_custom_objects()['gelu'] = gelu_tanh


def piecewise_linear(t, schedule):
    """分段线性函数
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示 t ∈ [0, 1000]时，输出从0均匀增加至1，而
    t ∈ [1000, 2000]时，输出从1均匀降低到0.1，最后
    t > 2000时，保持0.1不变。
    """
    schedule = sorted(schedule.items())
    if schedule[0][0] != 0:
        schedule = [(0, 0.0)] + schedule

    x = K.constant(schedule[0][1], dtype=K.floatx())
    t = K.cast(t, K.floatx())
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1.0 * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = K.constant(schedule[i][1], dtype=K.floatx())
        x = K.switch(t >= t_begin, x, x_begin)

    return x


def search_layer(inputs, name, exclude_from=None):
    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
         如果找不到，那就返回None。
    """
    if exclude_from is None:
        exclude_from = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer, keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude_from)
                if layer is not None:
                    return layer


def sequence_masking(x, mask, mode=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    mode: 如果是0，则直接乘以mask；
          如果是1，则在padding部分减去一个大正数。
    axis: 序列所在轴，默认为1；
    """
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if axis is None:
            axis = 1
        if axis == -1:
            axis = K.ndim(x) - 1
        assert axis > 0, 'axis muse be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12


def batch_gather(params, indices):
    """同tf旧版本的batch_gather
    """
    try:
        return tf.gather(params, indices, batch_dims=-1)
    except Exception as e1:
        try:
            return tf.batch_gather(params, indices)
        except Exception as e2:
            raise ValueError('%s\n%s\n' % (e1.message, e2.message))


def pool1d(
    x,
    pool_size,
    strides=1,
    padding='valid',
    data_format=None,
    pool_mode='max'
):
    """向量序列的pool函数
    """
    x = K.expand_dims(x, 1)
    x = K.pool2d(
        x,
        pool_size=(1, pool_size),
        strides=(1, strides),
        padding=padding,
        data_format=data_format,
        pool_mode=pool_mode
    )
    return x[:, 0]


def divisible_temporal_padding(x, n):
    """将一维向量序列右padding到长度能被n整除
    """
    r_len = K.shape(x)[1] % n
    p_len = K.switch(r_len > 0, n - r_len, 0)
    return K.temporal_padding(x, (0, p_len))


def swish(x):
    """swish函数（这样封装过后才有 __name__ 属性）
    """
    return tf.nn.swish(x)


def leaky_relu(x, alpha=0.2):
    """leaky relu函数（这样封装过后才有 __name__ 属性）
    """
    return tf.nn.leaky_relu(x, alpha=alpha)


def symbolic(f):
    """恒等装饰器（兼容旧版本keras用）
    """
    return f


# 给旧版本keras新增symbolic方法（装饰器），
# 以便兼容optimizers.py中的代码
K.symbolic = getattr(K, 'symbolic', None) or symbolic

custom_objects = {
    'gelu_erf': gelu_erf,
    'gelu_tanh': gelu_tanh,
    'gelu': gelu_erf,
    'swish': swish,
    'leaky_relu': leaky_relu,
}

keras.utils.get_custom_objects().update(custom_objects)
