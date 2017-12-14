import numpy as np
import tensorflow as tf
import reader

DATA_PATH = "../../data/simple-examples/data"
HIDDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = 10000

LEARNING_RATE = 1
TRAINING_BATCH_SIZE = 20
TRAINING_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable(
            "embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        outputs = []
        state = self.initial_state

        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_ouput, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_ouput)
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits], [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])

        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training:
            return

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(
            self.cost, trainable_variables), MAX_GRAD_NORM)

        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))


def my_run_epoch(session, model, data, train_op, output_log):

    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    epoch_size = len(data)

    for step in range(epoch_size):
        x, y = data[step]
        x, y = np.array(x), np.array(y)
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                     feed_dict={model.input_data: x,
                                                model.targets: y,
                                                model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("Afer %d steps,perplexity is %.3f " %
                  (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)


def my_ptb(raw_data, batch_size, num_steps):
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.reshape(
        raw_data[0:batch_size * batch_len], [batch_size, batch_len])
    epoch_size = int(np.round(batch_len / num_steps))
    result = []
    for i in range(epoch_size):
        x = data[0:batch_size, i * num_steps:(i + 1) * num_steps]
        y = data[0:batch_size, i * num_steps + 1:(i + 1) * num_steps + 1]
        x_shape = np.shape(x)
        y_shape = np.shape(y)
        if x_shape == y_shape:
            result.append((x, y))
    return result


def main():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    data_train = my_ptb(train_data, TRAINING_BATCH_SIZE, TRAINING_NUM_STEP)
    data_valid = my_ptb(valid_data, EVAL_BATCH_SIZE, EVAL_NUM_STEP)
    data_test = my_ptb(test_data, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAINING_BATCH_SIZE, TRAINING_NUM_STEP)

    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        for i in range(NUM_EPOCH):
            print("In iteration :%d" % (i + 1))
            my_run_epoch(session, train_model, data_train,
                         train_model.train_op, True)
            valid_perplexity = my_run_epoch(
                session, eval_model, data_valid,
                tf.no_op(), False)
            print("Epoch: %d  Validation Perplexity: %.3f" %
                  (i + 1, valid_perplexity))

        test_perplexity = my_run_epoch(
            session, eval_model, data_test,
            tf.no_op(), False)
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    main()
