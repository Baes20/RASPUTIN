from AI.Transformers.Transformer import *
from xgboost import XGBClassifier
from scipy.integrate import odeint
import math
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("D:/Projects/AI")


def get_lorenz_z_dataset():
    def lorenz(P, t):
        x, y, z = P
        dx = 10 * (y - x)
        dy = x * (28 - z) - y
        dz = x * y - (8 / 3) * z
        return [dx, dy, dz]

    time = np.arange((data_size * (seq_length + output_length)))

    P = odeint(lorenz, [-8, 7, 27], time)
    feature_dim = 3
    return P, feature_dim


def get_sine_dataset(resolution):
    def get_sine_waves(x):
        seasonality = np.sin(x) + np.sin(2 * x) + np.sin(3 * x)
        residual = np.cumsum(np.random.normal(0, 1, x.shape), axis=0)
        return seasonality + residual

    return get_sine_waves(np.arange(0, data_size * resolution * (seq_length + output_length), resolution)), 1


def get_linear_dataset():
    return np.arange((data_size * (seq_length + output_length))), 1


def get_expanding_random_sine_dataset(resolution):
    def get_expanding_sin(x):
        return np.sin(x) * np.sqrt(x) + np.sin(3 * x) + np.sin(7 * x) + np.random.normal(0, 1, x.shape)

    return get_expanding_sin(np.arange(0, data_size * resolution * (seq_length + output_length), resolution)), 1


def get_autoreg_dataset(resolution):
    def autoreg(Xs, t):  # Xs has 16 elements; Xs[0] = t, Xs[1] = t-1 ...
        Xs[1:] = Xs[:-1]
        Xs[0] = math.tanh(Xs[1]) + math.sin(2 * Xs[2]) + math.sin(3 * Xs[3]) + math.log(abs(Xs[4]))
        return Xs

    time = np.arange(0, data_size * resolution * (seq_length + output_length), resolution)
    P = odeint(autoreg, [1, -1, 0.5, 0.1, 0.3], time)
    return P[:, 0], 1


def get_random_dataset():
    def random_data(x):
        return np.cumsum(np.random.normal(0, 1, x.shape), axis=0)

    return random_data(np.arange(data_size * (seq_length + output_length))), 1

def mymetrics(Y, pred):
    Y_class = Y
    pred_class = pred
    ones = np.ones_like(Y_class)
    zeros = np.zeros_like(Y_class)
    tps = np.logical_and((Y_class == ones), (pred_class == ones))
    tns = np.logical_and((Y_class == zeros), (pred_class == zeros))
    fps = np.logical_and((Y_class == zeros), (pred_class == ones))
    fns = np.logical_and((Y_class == ones), (pred_class == zeros))
    tp = np.sum(np.logical_and((Y_class == ones), (pred_class == ones)), axis=0)
    tn = np.sum(np.logical_and((Y_class == zeros), (pred_class == zeros)), axis=0)
    fp = np.sum(np.logical_and((Y_class == zeros), (pred_class == ones)), axis=0)
    fn = np.sum(np.logical_and((Y_class == ones), (pred_class == zeros)), axis=0)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    p_precision = tp / (tp + fp)
    n_precision = tn / (tn + fn)
    recall = tp / (tp + fn)
    negative_recall = tn / (tn + fp)
    F1 = 2 * (recall * p_precision) / (recall + p_precision)

    print("acc", accuracy, "p_precision", p_precision, "n_precision", n_precision,
          "recall", recall, "neg_recall", negative_recall, "F1", F1)

#################################################
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
data_size = 100
EPOCHS = 100
seq_length = 64
output_length = 8
feature_dim = 3
# input_vocab_size = tokenizer_pt.vocab_size + 2
# target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.1
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#################################################

nasdaq = pd.read_csv("./Datasets/nasdaq100_padding.csv")
print(nasdaq.describe())

nasdaq_np = nasdaq.to_numpy()


# nasdaq_np = np.diff(nasdaq_np, axis=0, append=nasdaq_np[-1:])
# scaler = preprocessing.RobustScaler()
# scaler.fit(nasdaq_np)
# nasdaq_np = scaler.transform(nasdaq_np)


def chunk(data, chunk_length, preprocessor_func=None, window_step=1):
    temp = []
    for i in range(len(data) - chunk_length):
        if i % window_step == 0:
            if preprocessor_func is not None:
                temp.append(preprocessor_func(data[i:i + chunk_length]))
            else:
                temp.append(data[i:i + chunk_length])

    return np.array(temp)


nasdaq = chunk(nasdaq_np, 65, window_step=4)
nasdaq = nasdaq.transpose([0, 2, 1])
nasdaq = nasdaq.reshape((nasdaq.shape[0] * nasdaq.shape[1], nasdaq.shape[2]))


def apply_robust(data):
    temp = data.transpose([1, 0])
    return preprocessing.minmax_scale(temp).transpose([1, 0])


def test_XGBoost(nasdaq):
    nasdaq = apply_robust(nasdaq)

    new_nasdaq = []
    for timeseries in nasdaq:
        if not (timeseries[-1] > 0.99999 or timeseries[-1] < 0.00001):
            new_nasdaq.append(timeseries)
    nasdaq = np.array(new_nasdaq)

    X, Y = nasdaq[:, :-1], nasdaq[:, -2:]
    Y = np.diff(Y, axis=1)
    Y = np.sign(Y)
    Y = np.where(Y <= 0, Y * 0, Y)

    print(Y.shape)
    base_winrate = np.sum(Y) / len(Y)
    print(base_winrate)

    train_size = int(0.8 * len(X))
    trainX, trainY = X[train_size:], Y[train_size:, 0]
    validX, validY = X[:train_size], Y[:train_size, 0]

    print(trainX.shape, trainY.shape)

    model = XGBClassifier(max_depth=5, n_estimators=300)
    model.fit(trainX, trainY, eval_set=[(trainX, trainY), (validX, validY)], early_stopping_rounds=50)
    pred = model.predict(validX)

    plt.hist(validY)
    plt.hist(pred)
    plt.show()

    mymetrics(validY, pred)


def test_Transformer(nasdaq):
    nasdaq = apply_robust(nasdaq)
    new_nasdaq = []
    for timeseries in nasdaq:
        if not (timeseries[-1] > 0.9999 or timeseries[-1] < 0.0001):
            new_nasdaq.append(timeseries)
    nasdaq = np.array(new_nasdaq)

    X, Y = nasdaq[:, :-1], nasdaq[:, -2:]

    discretizer = preprocessing.KBinsDiscretizer(n_bins=63, encode='ordinal')
    discretizer.fit(X)
    X = discretizer.transform(X)

    Y = np.diff(Y, axis=1)
    Y = np.sign(Y)
    Y = np.where(Y <= 0, Y * 0, Y)

    print(Y.shape)
    base_winrate = np.sum(Y) / len(Y)
    print(base_winrate)

    train_size = int(0.8 * len(X))
    trainX, trainY = X[:train_size], Y[:train_size]
    validX, validY = X[train_size:], Y[train_size:]
    print(trainX.shape, trainY.shape)

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model = Transformer_Encoder(num_layers=6, d_model=64, num_heads=8, dff=128,
                                input_vocab_size=64, target_vocab_size=1, rate=dropout_rate)
    checkpoint_path = "./SavedModel/Transformer/Transformer_FXTM.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True)
    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
    model.build(input_shape=(None, 64))
    history = model.fit(trainX, trainY, validation_data=(validX, validY), batch_size=256,
                        epochs=EPOCHS, callbacks=[checkpoint])


test_XGBoost(nasdaq)


def RMSE(real, pred):
    return tf.sqrt(tf.losses.mean_squared_error(real, pred))


def diff_mda(true, pred):
    res = tf.equal(tf.sign(true), tf.sign(pred))
    return tf.reduce_mean(tf.cast(res, tf.float32), axis=[0, 1])


# model.compile(optimizer=optimizer, loss=loss_object, metrics=[RMSE, diff_mda])
# model.build(input_shape=[(None, None, feature_dim), (None, None, feature_dim)])
# history = model.fit([trainX, train_tar], trainY, validation_data=([validX, valid_tar], validY), batch_size=256,
#                     epochs=EPOCHS, callbacks=[checkpoint])
# model.load_weights(checkpoint_path)
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()


def evaluate(inp_sequence, max_length):
    # inp sequence is int64
    encoder_input = np.expand_dims(inp_sequence, 0)

    decoder_input = encoder_input[:, -1:]
    output = decoder_input

    for i in range(max_length):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = model([encoder_input, output])

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int64)

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predictions], axis=1)

    return tf.squeeze(output, axis=0)[1:]


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def predict(sequence, how_many, plot=''):
    result = evaluate(sequence, how_many)

    predicted_sequence = result

    print('Input: {}'.format(sequence))
    print('Predicted translation: {}'.format(predicted_sequence))

    return predicted_sequence


# pred = predict(validX[0], how_many=output_length, plot='decoder_layer4_block2')



