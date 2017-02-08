import tensorflow as tf
import numpy as np


OPTIMIZERS = {"rmsprop": tf.train.RMSPropOptimizer,
              "adadelta": tf.train.AdadeltaOptimizer,
              "adagrad": tf.train.AdagradOptimizer,
              "momentum": tf.train.MomentumOptimizer}

ACTIVATIONS = {"sigmoid": tf.sigmoid,
               "relu": tf.nn.relu,
               "tanh": tf.nn.tanh,
               "none": lambda x: x}


# TODO make all batch_size tf variables for 1-batch prediction

def bilstm(input_ops, n_hidden, sequence_lengths, activation, dropout_rate=0.0):
    """
    Creates intermediate biLSTM layer with specified properties. In Tensorflow
    lingo, it creates the ops needed to run input through this layer (especially
    the `outputs` and `dropouts` ops
    :param input_ops:
    :param n_hidden:
    :param sequence_lengths:
    :param activation:
    :param dropout_rate:
    :return:
    """
    timesteps, batch_size = tf.pack(input_ops).get_shape()[:2]
    # First job: split order 3 tensor (timesteps * batch_size * input_len)
    # to list of $timesteps order 2 tensors with shape (batch_size * input_len)
    # This is what tf.nn.bidirectional_rnn wants
    input_ops = tf.split(0, timesteps, input_ops)
    # tf.split turns tensor of shape (x, y, z) into list of x tensors with
    # shape (1, y, z). tf.unpack lets us get rid of the bogus 1st order so the
    # list has now x tensors of shape (y, z), which is what the biLSTM expects
    input_ops = [tf.unpack(x, name="inputs")[0] for x in input_ops]
    # make forward and backward cells:
    with tf.variable_scope('forward'):
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
    with tf.variable_scope('backward'):
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
    # make initial states (Xavier initialization)
    cell_shape = (batch_size.value, n_hidden)
    # init_fw = tf.get_variable("init_fw", shape=cell_shape,
    #                           initializer=xavier_init(
    #                               cell_shape[0], cell_shape[1])
    #                           )
    # init_bw = tf.get_variable("init_bw", shape=cell_shape,
    #                           initializer=xavier_init(
    #                               cell_shape[0], cell_shape[1])
    #                           )
    # get layer output op
    outputs, _, _ = tf.nn.bidirectional_rnn(cell_fw, cell_bw, input_ops,
                                            # initial_state_fw=init_fw,
                                            # initial_state_bw=init_bw,
                                            dtype=tf.float32,
                                            sequence_length=sequence_lengths)
    # apply activation function by each timestep and batch
    activations = [[activation(tf.unpack(b)[0])
                   for b in tf.split(0, batch_size, ts)]
                   for ts in outputs]
    dropouts = activations
    # wrap dropout operation around
    if 0 < dropout_rate < 1:
        # TODO make dropout_rate a placeholder so it can be switched on during
        # training and off during testing (https://www.tensorflow.org/versions/
        # r0.11/tutorials/mnist/pros/index.html#densely-connected-layer)
        dropouts = tf.nn.dropout(activations, keep_prob=1-dropout_rate,
                                 name="dropouts")
    return outputs, activations, dropouts


def projection(input_ops, y_task, n_hidden, sequence_lengths, class_weights,
               optmzr, batch_size=1):
    """
    Used for sequence labeling tasks, makes a dense projection of inputs to
    labels
    :param input_ops:
    :param y_task:
    :param n_hidden:
    :param sequence_lengths:
    :param class_weights:
    :param optmzr:
    :param batch_size:
    :return:
    """
    timesteps = len(input_ops)
    n_classes = len(class_weights)
    # n_classes = class_weights
    # output projection, projects from hidden states to outputs
    # it's 2*n_hidden because input comes from biLSTM, doubles hidden states
    # TODO make more robust, let projection() figure out n_hidden and batch_size
    w = tf.get_variable("weights", [2*n_hidden, n_classes],
                        initializer=xavier_init(2*n_hidden, n_classes))
    b = tf.get_variable("biases", [n_classes],
                        initializer=xavier_init(1, n_classes))
    # make input_ops one tensor timesteps * batch_size * n_classes
    # NB: first $timesteps rows in this tensor are all from 1st TS, not batch.
    # that's not necessarily a problem, but refactoring here makes life easier
    # later on, when reading off outputs
    _input_ops = tf.concat(0, input_ops)
    # so, swap batch_size/timesteps axes, so that adjacent rows come from same
    # batch but different TS (requires reshaping first and then re-reshaping)
    _input_ops = tf.reshape(_input_ops, [timesteps, batch_size, 2*n_hidden])
    _input_ops = tf.transpose(_input_ops, [1, 0, 2])  # swap axes
    # _input_ops is now indexed by [batch, timestep, 2*n_hidden] - perfect!
    # now we just flatten for faster multiplication
    _input_ops = tf.reshape(_input_ops, [timesteps * batch_size, 2*n_hidden])

    # Generate predictions
    # preds = tf.add(tf.matmul(_input_ops, w), b, name="preds")
    # Computing masked loss
    y_flat = tf.reshape(y_task, [-1], name="y_flat")
    y_onehot = tf.one_hot(y_flat, depth=n_classes)
    # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, y_flat)

    # class_weights = tf.constant(1.0, dtype=tf.float32, shape=[n_classes])
    preds, losses = tf.contrib.learn.ops.softmax_classifier(
        _input_ops, y_onehot, w, b, class_weight=class_weights)

    mask = tf.sign(tf.to_float(y_flat))  # 0 for PAD (PAD = 0), 1 for all others
    masked_losses = mask * losses  # make loss for all PAD zero
    # Bring back to [batch_size, timesteps] shape
    masked_losses = tf.reshape(masked_losses, [batch_size, timesteps],
                               name="masked_losses")
    # Calculate mean loss per example in batch
    mean_loss_by_example = tf.truediv(  # divide timestep-loss sum by seq_lens
        tf.reduce_sum(masked_losses, reduction_indices=1),
        tf.cast(sequence_lengths, tf.float32), name="mean_loss_by_ex"
    )
    # Finally, average loss over examples in batch and create optimizer
    loss = tf.reduce_mean(mean_loss_by_example, name="batch_mean_loss")
    opt = optmzr.minimize(loss)
    return preds, loss, opt


def encdec(encoder_inputs, y_task, n_hidden, n_classes, sequence_lengths,
           optmzr, timesteps, batch_size=1, attention=True):
    """
    Encoder-decoder, is used for seq2seq tasks
    :param encoder_inputs:
    :param y_task:
    :param n_hidden:
    :param n_classes: target vocab size
    :param sequence_lengths:
    :param optmzr:
    :param timesteps:
    :param batch_size:
    :param attention:
    :return:
    """
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
    decoder_inputs = ([tf.zeros_like(
        encoder_inputs[0], dtype=np.float32, name="GO")] + encoder_inputs[:-1])

    if attention:
        dec_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
        enc_out, enc_state = tf.nn.rnn(cell, encoder_inputs, dtype=np.float32,
                                       sequence_length=sequence_lengths)
        top_states = [tf.python.array_ops.reshape(e, [-1, 1, cell.output_size])
                      for e in enc_out]
        attention_states = tf.python.array_ops.concat(1, top_states)

        _outputs, _ = tf.nn.seq2seq.attention_decoder(
            decoder_inputs, enc_state, attention_states, dec_cell)
    else:
        _outputs, _ = tf.nn.seq2seq.basic_rnn_seq2seq(
            encoder_inputs, decoder_inputs, cell)

    # Reshape _outputs such that batch/timestep indexing is correct
    # (see prediction())
    _outputs = tf.concat(0, _outputs)
    _outputs = tf.reshape(_outputs, [timesteps, batch_size, n_hidden])
    _outputs = tf.transpose(_outputs, [1, 0, 2])
    _outputs = tf.reshape(_outputs, [-1, n_hidden])

    w = tf.get_variable("weights", [n_hidden, n_classes],
                        initializer=xavier_init(n_hidden, n_classes))
    w_t = tf.transpose(w)  # used for sampled softmax (see sampled_loss() below)
    b = tf.get_variable("biases", [n_classes],
                        initializer=xavier_init(1, n_classes))

    preds = tf.add(tf.matmul(_outputs, w), b)
    preds = tf.reshape(preds, [timesteps * batch_size, n_classes], name="preds")
    y_flat = tf.reshape(y_task, [-1], name="y_flat")

    def sampled_loss(inputs, lbls):
        with tf.device("/cpu:0"):
            # TODO see what happens when remove tf.device
            return tf.nn.sampled_softmax_loss(
                w_t, b, inputs, lbls, batch_size, n_classes)

    # loss = sampled_loss(tf.concat(0, _outputs), y_flat)
    #    using _outputs, projection happens inside sampled_loss()
    # TODO masking
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(preds, y_flat))

    opt = optmzr.minimize(loss)
    return preds, loss, opt


def xavier_init(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    Taken from https://github.com/google/prettytensor/blob/
    a69f13998258165d6682a47a931108d974bab05e/prettytensor/layers.py
    :param n_inputs: The number of input nodes into each output.
    :param n_outputs: The number of output nodes for each input.
    :param uniform: If true use a uniform distribution, otherwise use a normal.
    :return: An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


class SeqMtlModel:
    """
    Model for sequence multitask learning.
    """
    def __init__(self, tasks, input_size, num_layers, hid_dims, optimizer,
                 actfunc, timesteps=50, dropout_rate=0.1, learning_rate=0.1,
                 batch_size=64, embed_size=0, pretrained_embeddings=None,
                 weighted_costs=False):
        """
        :param tasks:
        :param input_size: only relevant if not embed
        :param num_layers:
        :param hid_dims:
        :param optimizer:
        :param actfunc:
        :param timesteps:
        :param dropout_rate:
        :param learning_rate:
        :param batch_size:
        :param embed_size:
        """
        # The input placeholder, expects either IDs (if embed==True) or
        # vectors (pre-trained embeddings or one-hots)
        if embed_size > 0 or pretrained_embeddings is not None:
            self.X_in = tf.placeholder(tf.int32, (timesteps, batch_size),
                                       name="X_in")
        else:
            self.X_in = tf.placeholder(
                tf.float32, (timesteps, batch_size, input_size), name="X_in")
        # The following are registries for different operations per task
        # Each dict maps tasks to appropriate tf operations as registered below
        self.preds = {}
        self.losses = {}
        self.ys = {}

        # Some utilities
        self.optimizers = {}
        self.task2layer = {}
        self.task2tasktype = {}
        self.layer_outputs = []  # holds output ops for each intermediate layer
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.sequence_lengths = tf.placeholder(tf.int32, batch_size,
                                               name="sequence_lengths")

        print("=====================================")
        # # # INPUT LAYER # # #
        if pretrained_embeddings is not None:
            print("Creating trainable embedding layer initialized with "
                  "provided pretrained embeddings.")
            emb_matrix = tf.get_variable(
                name="embedding",
                shape=pretrained_embeddings.shape,
                initializer=tf.constant_initializer(pretrained_embeddings),
                trainable=True)  # allow updating embeddings task-specifically
            layer_input = tf.nn.embedding_lookup(emb_matrix, self.X_in)
        elif embed_size > 0:  # learn embed matrix from indices to embeddings
            print("Creating embedding layer mapping inputs to embeddings "
                  "of length {}.".format(embed_size))
            emb_matrix = tf.get_variable(
                name="embedding",
                shape=[input_size, embed_size], dtype=tf.float32,
                initializer=xavier_init(input_size, embed_size))
            layer_input = tf.nn.embedding_lookup(emb_matrix, self.X_in)
        else:  # use provided vectors (pre-trained embeddings or one-hots)
            layer_input = self.X_in

        # # # INTERMEDIATE LAYERS # # #
        print("Creating stacked biLSTM of {} layers with {} hidden units."
              .format(num_layers, hid_dims))
        for i in range(num_layers):
            activation = ACTIVATIONS.get(actfunc)
            with tf.variable_scope("layer{}".format(i), reuse=None):
                outputs, _, dropouts = bilstm(layer_input, hid_dims,
                                              self.sequence_lengths,
                                              activation, dropout_rate)
            self.layer_outputs.append(outputs)
            # dropouts from previous layer are input to next layer
            layer_input = dropouts
            # layer_input = outputs

        # # # TASK OUTPUTS # # #
        # Register tasks at layers
        for task_id, task in tasks.items():
            layer = task.layer
            if layer > num_layers:
                print("Warning! Task was to be registered at layer {}, but "
                      "model only has {} layers. Using top layer instead.".
                      format(layer, num_layers))
                layer = num_layers
            print("Registering task '{}' (type '{}', {} classes) at LSTM "
                  "layer {}.".format(task_id, task.task_type,
                                     task.get_num_labels(), layer))
            try:
                outputs = self.layer_outputs[layer-1]
            except IndexError:
                import sys
                sys.stderr.write("Error: Trying trying to register task, not "
                                 "enough hidden layers (expected at least {}, "
                                 "only {} present.)".format(layer, num_layers))
            layer_y, pred, loss, opt = None, None, None, None
            optmzr = OPTIMIZERS.get(optimizer)(learning_rate=learning_rate)
            n_classes = task.get_num_labels()
            with tf.variable_scope("output_{}".format(task_id), reuse=None):
                if task.task_type == "lbl":
                    class_weights = task.get_class_weights() if weighted_costs \
                        else np.ones(n_classes)
                    print("  Class weights: ", class_weights)
                    layer_y = tf.placeholder(tf.int32,
                                             (batch_size, timesteps),
                                             name="labels")
                    pred, loss, opt = projection(outputs, layer_y, hid_dims,
                                                 self.sequence_lengths,
                                                 class_weights, optmzr,
                                                 batch_size)
                elif task.task_type == "seq":
                    layer_y = tf.placeholder(tf.int32,
                                             (batch_size, timesteps),
                                             name="labels")
                    pred, loss, opt = encdec(outputs, layer_y, hid_dims,
                                             n_classes, self.sequence_lengths,
                                             optmzr, timesteps, batch_size)
                else:
                    raise ValueError("Illegal task type '{}'. Only 'lbl' and "
                                     "'seq' supported.".format(task.task_type))
            # Register operations for tasks
            self.ys[task_id] = layer_y
            self.preds[task_id] = pred
            self.losses[task_id] = loss
            self.optimizers[task_id] = opt
        print("=====================================")
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, task, x, input_lengths, y=None, mode="train"):
        """
        Performs a model step. Performs different sequence of operations
        (computing predictions/loss, optimizing) depending on mode.
        :param session:
        :param task:
        :param x:
        :param input_lengths:
        :param y:
        :param mode: any of 'train', 'eval', 'decode'.
        :return:
        """
        pred = self.preds[task.task_id]
        optimizer = self.optimizers[task.task_id]
        loss = self.losses[task.task_id]
        gold = self.ys[task.task_id]

        l = None
        if mode == "decode":
            feed = {self.X_in: x, self.sequence_lengths: input_lengths}
            p = session.run(pred, feed_dict=feed)
        elif mode == "eval":
            assert (y is not None), "Need to provide gold sequence y " \
                                    "when evaluating"
            feed = {self.X_in: x, gold: y, self.sequence_lengths: input_lengths}
            p, l = session.run([pred, loss], feed_dict=feed)
        elif mode == "train":
            assert (y is not None), "Need to provide gold sequence y " \
                                    "when training"
            feed = {self.X_in: x, gold: y, self.sequence_lengths: input_lengths}
            p, l, _ = session.run([pred, loss, optimizer], feed_dict=feed)
        else:
            raise ValueError("Illegal step mode '{}'. Valid modes are 'train', "
                             "'decode' and 'eval'.")
        return p, l
