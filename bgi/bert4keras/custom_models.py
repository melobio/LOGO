#! -*- coding: utf-8 -*-
# 主要模型

import json

# 只有 Tensorflow 2.0版本
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Lambda, Dense

from bgi.bert4keras.layers import *
from bgi.bert4keras.snippets import delete_arguments
from bgi.bert4keras.models import Transformer, BERT, ALBERT




class Variable_Length_ALBERT(Transformer):
    """构建BERT模型
    """

    def __init__(
            self,
            max_position,  # 序列最大长度
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            custom_position_ids=False,  # 是否自行传入位置id
            custom_masked_sequence=False,
            multi_inputs:list = [],
            **kwargs  # 其余参数
    ):
        super(Variable_Length_ALBERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids
        self.custom_masked_sequence = custom_masked_sequence
        self.multi_inputs = multi_inputs

    def get_inputs(self):
        """BERT的输入是token_ids和segment_ids
        （但允许自行传入位置id，以实现一些特殊需求）
        """
        inputs = []
        if len(self.multi_inputs) > 0:
            for ii in range(len(self.multi_inputs)):
                t_in = keras.layers.Input(shape=(self.sequence_length,), name='Input-Token-{}'.format(ii))
                inputs.append(t_in)
            return inputs
        return inputs

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和
        """
        #x, s = inputs[:2]
        z = self.layer_norm_conds[0]
        # if self.custom_position_ids:
        #     p = inputs[2]
        # else:
        p = None

        # if self.type_vocab_size > 0:
        #     t = inputs[2]
        # else:

        t = None

        embedding_inputs = []
        for x in inputs:
            x = self.apply(
                inputs=x,
                layer=Embedding,
                input_dim=self.vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                mask_zero=True,
                name='Embedding-Token'
            )
            embedding_inputs.append(x)
        # s = self.apply(
        #     inputs=s,
        #     layer=Embedding,
        #     input_dim=2,
        #     output_dim=self.embedding_size,
        #     embeddings_initializer=self.initializer,
        #     name='Embedding-Segment'
        # )

        #embedding_inputs = [x, s]

        # if self.type_vocab_size > 0:
        #     t = self.apply(
        #         inputs=t,
        #         layer=Embedding,
        #         input_dim=self.type_vocab_size,
        #         output_dim=self.embedding_size,
        #         embeddings_initializer=self.initializer,
        #         name='Embedding-Token-Type'
        #     )
        #     embedding_inputs.append(t)

        x = self.apply(inputs=embedding_inputs, layer=keras.layers.Add, name='Embedding-Token-Segment')
        x = self.apply(
            inputs=self.simplify([x, p]),
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=keras.layers.Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=keras.layers.Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """ALBERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-MultiHeadSelfAttention'
        feed_forward_name = 'Transformer-FeedForward'
        attention_mask = self.compute_attention_mask(0)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_mask': None}
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=keras.layers.Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=keras.layers.Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=keras.layers.Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=keras.layers.Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        x = inputs
        z = self.layer_norm_conds[0]
        outputs = [x]

        if self.with_pool or self.with_nsp:
            # Pooler部分（提取CLS向量）
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=keras.layers.Lambda,
                function=lambda x: x[:, 0],
                name='Pooler'
            )
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool
            x = self.apply(
                inputs=x,
                layer=keras.layers.Dense,
                units=self.hidden_size,
                activation=pool_activation,
                kernel_initializer=self.initializer,
                name='Pooler-Dense'
            )
            if self.with_nsp:
                # Next Sentence Prediction部分
                x = self.apply(
                    inputs=x,
                    layer=keras.layers.Dense,
                    units=2,
                    activation='softmax',
                    kernel_initializer=self.initializer,
                    name='NSP-Proba'
                )
            outputs.append(x)

        if self.with_mlm:
            # Masked Language Model部分
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=keras.layers.Dense,
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='MLM-Dense'
            )
            x = self.apply(
                inputs=self.simplify([x, z]),
                layer=LayerNormalization,
                conditional=(z is not None),
                hidden_units=self.layer_norm_conds[1],
                hidden_activation=self.layer_norm_conds[2],
                hidden_initializer=self.initializer,
                name='MLM-Norm'
            )
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            x = self.apply(inputs=x, layer=BiasAdd, name='MLM-Bias')
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=keras.layers.Activation,
                activation=mlm_activation,
                name='MLM-Activation'
            )
            outputs.append(x)

        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs


def build_transformer_model(
        config_path: str = None,
        configs: dict = None,
        checkpoint_path: str = None,
        model='bert',
        application='encoder',
        return_keras_model=True,
        **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """

    if config_path is not None:
        configs = {}
        configs.update(json.load(open(config_path)))
        configs.update(kwargs)

    if configs is None:
        configs = {}

    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings')
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')

    model, application = model.lower(), application.lower()

    models = {
        'bert': BERT,
        'albert': ALBERT,
        'albert_variable_length': Variable_Length_ALBERT,
        # 'nezha': NEZHA,
        # 'electra': ELECTRA,
        # 'gpt2_ml': GPT2_ML,
        # 't5': T5,
        # 'multi_inputs_bert': Multi_Inputs_BERT,
        # 'multi_inputs_alt_bert': Multi_Inputs_Alt_BERT,
        # 'albert_hierarchy': ALBERT_Hierarchy
    }
    MODEL = models[model]

    # if model != 't5':
    #     if application == 'lm':
    #         MODEL = extend_with_language_model(MODEL)
    #     elif application == 'unilm':
    #         MODEL = extend_with_unified_language_model(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model
    else:
        return transformer

if __name__ == '__main__':

    # Distributed Training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync

    multi_inputs = [1] * 10

    embedding_size = 128
    model_dim = 256
    num_heads = 2
    max_depth = 2
    vocab_size = 138
    num_classes = 1
    with strategy.scope():
        # 模型配置
        config = {
            "max_position": 512,
            "attention_probs_dropout_prob": 0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0,
            "embedding_size": embedding_size,
            "hidden_size": model_dim,
            "initializer_range": 0.02,
            "intermediate_size": model_dim * 4,
            # "max_position_embeddings": 512,
            "num_attention_heads": num_heads,
            "num_hidden_layers": max_depth,
            "num_hidden_groups": 1,
            "net_structure_type": 0,
            "gap_size": 0,
            "num_memory_blocks": 0,
            "inner_group_num": 1,
            "down_scale_factor": 1,
            "type_vocab_size": 0,
            "vocab_size": vocab_size,
            "custom_masked_sequence": False,
            "sequence_length": 100,
            "multi_inputs": multi_inputs,

        }
        bert = build_transformer_model(
            configs=config,
            # checkpoint_path=checkpoint_path,
            model='albert_variable_length',
            return_keras_model=False,
        )

        output = tf.reduce_mean(bert.model.output, axis=1)
        output = Dense(
            name='CLS-Activation',
            units=num_classes,
            activation='sigmoid',
            kernel_initializer=bert.initializer
        )(output)

        albert = tf.keras.models.Model(bert.model.input, output)
        albert.summary()
        albert.compile(optimizer='adam', loss=[tf.keras.losses.BinaryCrossentropy()],
                       metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])


    x = np.ones((1000, 100), dtype=np.int)
    y = np.ones((1000, 1), dtype=np.int)
    albert.fit([x,x], y, batch_size=32, epochs=1, steps_per_epoch=10)

