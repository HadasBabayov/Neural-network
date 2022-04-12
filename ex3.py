# Hadas Babayov 322807629
import sys
import numpy as np

NUM_OF_PIXELS = 784
OUTPUT = 10
sigmoid = lambda x: 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x, axis=0))
    return e / np.sum(e, axis=0)


# Forward the input through the network.
def fprop(x, y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = (np.dot(W1, x) + b1) / 128
    h1 = sigmoid(z1)
    z2 = (np.dot(W2, h1) + b2) / 10
    h2 = softmax(z2)
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in params:
        ret[key] = params[key]
    return ret


# Compute the gradients w.r.t all the parameters.
def bprop(fprop_cache):
    x, y, z1, h1, z2, h2 = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2')]
    dz2 = (h2 - y)
    dW2 = np.dot(dz2, h1.T)
    db2 = dz2
    dz1 = np.dot(fprop_cache['W2'].T, (h2 - y)) * sigmoid(z1) * (1 - sigmoid(z1))
    dW1 = np.dot(dz1, x.T)
    db1 = dz1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def train(X, Y, epoch, eta):
    # Initialize the parameters for two layers.
    w1 = np.random.rand(128, NUM_OF_PIXELS)
    b1 = np.random.rand(128, 1)
    w2 = np.random.rand(OUTPUT, 128)
    b2 = np.random.rand(OUTPUT, 1)
    params = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}

    for i in range(epoch):
        # Shuffle the data.
        xy_train = list(zip(X, Y))
        np.random.shuffle(xy_train)
        X_shuffled, Y_shuffled = zip(*xy_train)
        for x, y in zip(X_shuffled, Y_shuffled):
            x.shape = (784, 1)
            y.shape = (10, 1)
            # Forward the input through thr network.
            fprop_cache = fprop(x, y, params)
            # Compute the gradients w.r.t all the parameters.
            bprop_cache = bprop(fprop_cache)
            # Update all parameters.
            for key in bprop_cache.keys():
                params[key] -= eta * bprop_cache[key]
    return params


def predict(test, params):
    y_hats = []
    for t in test:
        t = t.reshape(t.size, 1)
        fprop_cache = fprop(t, 0, params)
        y_hat = fprop_cache['h2']
        y_hats.append(np.argmax(y_hat))
    return y_hats


train_x, train_y, test_x = sys.argv[1:4]
# Normalize the data.
X = np.loadtxt(train_x) / 255
Y = np.loadtxt(train_y)
encoded = np.zeros((Y.shape[0], 10))
for i, y in enumerate(Y):
    encoded[i][int(y)] = 1
Y = encoded
test = np.loadtxt(test_x) / 255

# Shuffle the data.
xy_train = list(zip(X, Y))
np.random.shuffle(xy_train)
X_shuffled, Y_shuffled = zip(*xy_train)

params = train(X_shuffled, Y_shuffled, 13, 0.1)
predictions = predict(test, params)

# Write the predictions to the file.
with open("test_y", "w") as f:
    for i in range(len(test) - 1):
        f.write(str(predictions[i]) + '\n')
    f.write(str(predictions[len(test) - 1]))
