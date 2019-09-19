from AI.Transformers.RegressionTransformer import *

from scipy.integrate import odeint
import math
import random
from DataCookers.VAEdataset import VaeGen
from datetime import datetime
import MetaTrader5
from sklearn import preprocessing


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


# dataset = get_sine_dataset()
# dataset = ((np.reshape(dataset, (data_size, seq_length + output_length)) ** 2) * 10 + 1).astype(np.int64)
########################CommonParams##############################

train_ratio = 0.8
symbol = [
    "EURUSD"]  # , "EURGBP", "GBPUSD"]#, "USDCHF", "USDJPY", "USDCAD"]#, "NZDUSD", "CHFJPY", "EURAUD", "AUDJPY", "GBPCAD", "GBPJPY", "EURCAD"]
# symbol = ["EURUSD", "EURGBP", "GBPUSD"]
ohlc = ["close", "high", "low"]
ema = None
oscillator = None
ema_list = None
diff = True
preprocess = False
predict_count = 4
date = datetime(2019, 8, 9)
n_sample = 92160
timeframe = MetaTrader5.MT5_TIMEFRAME_M1

##################################################################

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
loss_object = tf.keras.losses.Huber()
#################################################

FXTM_gen = VaeGen(train_ratio, seq_length, output_length, symbol, ohlc, date, n_ema=ema, osc_list=oscillator,
                  ema_list=ema_list,
                  test_output_count=predict_count + 16,
                  window_step=4,
                  num_samples=n_sample,
                  diff=diff,
                  logtransform=False,
                  preprocess=preprocess,
                  timeframe=timeframe)

train = FXTM_gen.get_train_data()
valid = FXTM_gen.get_val_data()

train = np.squeeze(train)
valid = np.squeeze(valid)

train = np.cumsum(train, axis=1)
valid = np.cumsum(valid, axis=1)

# for i in range(len(train)):
#     train[i] = preprocessing.minmax_scale(train[i])
#
# for i in range(len(valid)):
#     valid[i] = preprocessing.minmax_scale(valid[i], axis=0)
for i in range(100):
    plt.plot(np.squeeze(train[i]))
plt.show()


trainX, trainY = FXTM_gen.get_X_and_Y(train)
validX, validY = FXTM_gen.get_X_and_Y(valid)


train_tar = np.concatenate([trainX[:, -1:], trainY[:, :-1]], axis=1)
valid_tar = np.concatenate([validX[:, -1:], validY[:, :-1]], axis=1)

print(trainX.shape, train_tar.shape, trainY.shape)

model = Transformer(num_layers, d_model, num_heads, dff, 256, 256, n_features=feature_dim, rate=dropout_rate)

checkpoint_path = "./SavedModel/Transformer/Transformer_FXTM.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True)


def RMSE(real, pred):
    return tf.sqrt(tf.losses.mean_squared_error(real, pred))

def diff_mda(true, pred):
    res = tf.equal(tf.sign(true), tf.sign(pred))
    return tf.reduce_mean(tf.cast(res, tf.float32), axis=[0, 1])

model.compile(optimizer=optimizer, loss=loss_object, metrics=[RMSE, diff_mda])
model.build(input_shape=[(None, None, feature_dim), (None, None, feature_dim)])
history = model.fit([trainX, train_tar], trainY, validation_data=([validX, valid_tar], validY), batch_size=256,
                    epochs=EPOCHS, callbacks=[checkpoint])
model.load_weights(checkpoint_path)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


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
def mymetrics(Y, pred):
    Y_class = np.sign(Y)
    pred_class = np.sign(pred)
    ones = np.ones_like(Y_class)
    tps = np.logical_and((Y_class == ones), (pred_class == ones))
    tns = np.logical_and((Y_class == -ones), (pred_class == -ones))
    fps = np.logical_and((Y_class == -ones), (pred_class == ones))
    fns = np.logical_and((Y_class == ones), (pred_class == -ones))
    tp = np.sum(np.logical_and((Y_class == ones), (pred_class == ones)), axis=0)
    tn = np.sum(np.logical_and((Y_class == -ones), (pred_class == -ones)), axis=0)
    fp = np.sum(np.logical_and((Y_class == -ones), (pred_class == ones)), axis=0)
    fn = np.sum(np.logical_and((Y_class == ones), (pred_class == -ones)), axis=0)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    p_precision = tp / (tp + fp)
    n_precision = tn / (tn + fn)
    recall = tp / (tp + fn)
    negative_recall = tn / (tn + fp)
    F1 = 2 * (recall * p_precision) / (recall + p_precision)

    count_uu = 0
    count_ud = 0
    count_du = 0
    count_dd = 0
    total = 0
    for i in range(len(tps)):  # for all timesteps
        if pred_class[i, 1] == -1 and pred_class[i, 2] == -1:  # when both H and L are predicted to go up
            if Y_class[i, 1] == 1 and Y_class[i, 2] == 1:  # both goes up as predicted; the best
                count_uu += 1 # 33%
            if Y_class[i, 1] == 1 and Y_class[i, 2] == -1:  # high up, low down, second best
                count_ud += 1 # 30%
            if Y_class[i, 1] == -1 and Y_class[i, 2] == 1:  # high down, low up, third best
                count_du += 1 # 30%
            if Y_class[i, 1] == -1 and Y_class[i, 2] == -1: #both down, worst
                count_dd += 1
            total += 1

    count_th = 0
    total_th = 0
    for i in range(len(tps)):  # for all timesteps
        if pred[i, 1] < -0.0003 and pred[i, 2] < -0.0003:
            if Y_class[i, 1] == -1 and Y_class[i, 2] == -1: # checking accuracy when the pred value is higher by certain amount
                count_th += 1
            total_th += 1



    print("uu", count_uu/total, "ud", count_ud/total, "du", count_du/total, "dd", count_dd/total, "sumall",
           count_uu/total + count_ud/total + count_du/total + count_dd/total)
    print("threshold acc 0.0001", count_th/total_th)
    print("acc", accuracy, "p_precision", p_precision,
          "recall", recall, "neg_recall", negative_recall, "F1", F1)

pred = model.forecast(validX, 4)
pred = pred
X = validX
Y = validY

for i in range(1000):
    plt.clf()
    plt.plot(np.concatenate([X[i], Y[i]], axis=0))
    plt.plot(np.concatenate([X[i], pred[i]], axis=0))
    plt.draw()
    plt.pause(0.5)


mymetrics(Y, pred)
