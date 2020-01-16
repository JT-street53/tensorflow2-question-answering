import flags, copy, 
import tensorflow as tf

FLAGS = flags.FLAGS

#############################################
# https://github.com/kamalkraj/ALBERT-TF2.0 #
#############################################


class ALBertQALayer(tf.keras.layers.Layer):
    """Layer computing position and is_possible for question answering task."""

    def __init__(self, hidden_size, start_n_top, end_n_top, initializer, dropout, **kwargs):
        """Constructs Summarization layer.
        Args:
          hidden_size: Int, the hidden size.
          start_n_top: Beam size for span start.
          end_n_top: Beam size for span end.
          initializer: Initializer used for parameters.
          dropout: float, dropout rate.
          **kwargs: Other parameters.
        """
        super(ALBertQALayer, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.initializer = initializer
        self.dropout = dropout

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.start_logits_proj_layer = tf.keras.layers.Dense(
            units=1, kernel_initializer=self.initializer, name='start_logits/dense')
        self.end_logits_proj_layer0 = tf.keras.layers.Dense(
            units=self.hidden_size,
            kernel_initializer=self.initializer,
            activation=tf.nn.tanh,
            name='end_logits/dense_0')
        self.end_logits_proj_layer1 = tf.keras.layers.Dense(
            units=1, kernel_initializer=self.initializer, name='end_logits/dense_1')
        self.end_logits_layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name='end_logits/LayerNorm')
        self.answer_class_proj_layer0 = tf.keras.layers.Dense(
            units=self.hidden_size,
            kernel_initializer=self.initializer,
            activation=tf.nn.tanh,
            name='answer_class/dense_0')
        self.answer_class_proj_layer1 = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=self.initializer,
            use_bias=False,
            name='answer_class/dense_1')
        self.ans_feature_dropout = tf.keras.layers.Dropout(rate=self.dropout)
        super(ALBertQALayer, self).build(unused_input_shapes)

    def __call__(self,
                 sequence_output,
                 p_mask,
                 cls_index,
                 start_positions=None,
                 **kwargs):
        inputs = tf_utils.pack_inputs(
            [sequence_output, p_mask, cls_index, start_positions])
        return super(ALBertQALayer, self).__call__(inputs, **kwargs)


    def call(self, inputs, **kwargs):
        """Implements call() for the layer."""
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        sequence_output = unpacked_inputs[0]
        p_mask = unpacked_inputs[1]
        cls_index = unpacked_inputs[2]
        start_positions = unpacked_inputs[3]

        _, seq_len, _ = sequence_output.shape.as_list()
        sequence_output = tf.transpose(sequence_output, [1, 0, 2])

        start_logits = self.start_logits_proj_layer(sequence_output)
        start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
        start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
        start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)

        if kwargs.get("training", False):
            # during training, compute the end logits based on the
            # ground truth of the start position
            start_positions = tf.reshape(start_positions, [-1])
            start_index = tf.one_hot(start_positions, depth=seq_len, axis=-1,
                                     dtype=tf.float32)
            start_features = tf.einsum(
                'lbh,bl->bh', sequence_output, start_index)
            start_features = tf.tile(start_features[None], [seq_len, 1, 1])
            end_logits = self.end_logits_proj_layer0(
                tf.concat([sequence_output, start_features], axis=-1))

            end_logits = self.end_logits_layer_norm(end_logits)

            end_logits = self.end_logits_proj_layer1(end_logits)
            end_logits = tf.transpose(tf.squeeze(end_logits, -1), [1, 0])
            end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
            end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
        else:
            start_top_log_probs, start_top_index = tf.nn.top_k(
                start_log_probs, k=self.start_n_top)
            start_index = tf.one_hot(
                start_top_index, depth=seq_len, axis=-1, dtype=tf.float32)
            start_features = tf.einsum(
                'lbh,bkl->bkh', sequence_output, start_index)
            end_input = tf.tile(sequence_output[:, :, None], [
                                1, 1, self.start_n_top, 1])
            start_features = tf.tile(start_features[None], [seq_len, 1, 1, 1])
            end_input = tf.concat([end_input, start_features], axis=-1)
            end_logits = self.end_logits_proj_layer0(end_input)
            end_logits = tf.reshape(end_logits, [seq_len, -1, self.hidden_size])
            end_logits = self.end_logits_layer_norm(end_logits)

            end_logits = tf.reshape(end_logits,
                                    [seq_len, -1, self.start_n_top, self.hidden_size])

            end_logits = self.end_logits_proj_layer1(end_logits)
            end_logits = tf.reshape(
                end_logits, [seq_len, -1, self.start_n_top])
            end_logits = tf.transpose(end_logits, [1, 2, 0])
            end_logits_masked = end_logits * (
                1 - p_mask[:, None]) - 1e30 * p_mask[:, None]
            end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
            end_top_log_probs, end_top_index = tf.nn.top_k(
                end_log_probs, k=self.end_n_top)
            end_top_log_probs = tf.reshape(end_top_log_probs,
                                           [-1, self.start_n_top * self.end_n_top])
            end_top_index = tf.reshape(end_top_index,
                                       [-1, self.start_n_top * self.end_n_top])

        # an additional layer to predict answerability

        # get the representation of CLS
        cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=tf.float32)
        cls_feature = tf.einsum('lbh,bl->bh', sequence_output, cls_index)

        # get the representation of START
        start_p = tf.nn.softmax(start_logits_masked,
                                axis=-1, name='softmax_start')
        start_feature = tf.einsum('lbh,bl->bh', sequence_output, start_p)

        ans_feature = tf.concat([start_feature, cls_feature], -1)
        ans_feature = self.answer_class_proj_layer0(ans_feature)
        ans_feature = self.ans_feature_dropout(
            ans_feature, training=kwargs.get('training', False))
        cls_logits = self.answer_class_proj_layer1(ans_feature)
        cls_logits = tf.squeeze(cls_logits, -1)

        if kwargs.get("training", False):
            return (start_log_probs, end_log_probs, cls_logits)
        else:
            return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)


class ALBertQAModel(tf.keras.Model):

    def __init__(self, albert_config, max_seq_length, init_checkpoint, start_n_top, end_n_top, dropout=0.1, **kwargs):
        super(ALBertQAModel, self).__init__(**kwargs)
        self.albert_config = copy.deepcopy(albert_config)
        self.initializer = tf.keras.initializers.TruncatedNormal(
            stddev=self.albert_config.initializer_range)
        float_type = tf.float32

        input_word_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

        albert_layer = AlbertModel(config=albert_config, float_type=float_type)

        _, sequence_output = albert_layer(
            input_word_ids, input_mask, input_type_ids)

        self.albert_model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                           outputs=[sequence_output])
        if init_checkpoint != None:
            self.albert_model.load_weights(init_checkpoint)

        self.qalayer = ALBertQALayer(self.albert_config.hidden_size, start_n_top, end_n_top,
                                     self.initializer, dropout)

    def call(self, inputs, **kwargs):
        # unpacked_inputs = tf_utils.unpack_inputs(inputs)
        unique_ids = inputs["unique_ids"]
        input_word_ids = inputs["input_ids"]
        input_mask = inputs["input_mask"]
        segment_ids = inputs["segment_ids"]
        cls_index = tf.reshape(inputs["cls_index"], [-1])
        p_mask = inputs["p_mask"]
        if kwargs.get('training',False):
            start_positions = inputs["start_positions"]
        else:
            start_positions = None
        sequence_output = self.albert_model(
            [input_word_ids, input_mask, segment_ids], **kwargs)
        output = self.qalayer(
            sequence_output, p_mask, cls_index, start_positions, **kwargs)
        return (unique_ids,) + output
    
    