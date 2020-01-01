# -*- coding: utf-8 -*-

import numpy as np


class RNN(object):

    def __init__(self, input_dim, hidden_dim, output_dim, depth, lr=0.002):
        self.lr = lr
        self.depth = depth
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.U = xavier_init(input_dim, hidden_dim, fc=True)
        self.W = xavier_init(hidden_dim, hidden_dim, fc=True)
        self.V = xavier_init(hidden_dim, output_dim, fc=True)
        # tmp variable
        self.x = None
        self.H = None
        self.Alpha = None

    def forward_prop(self, x):
        batch_size = x.shape[1]
        self.x = x
        self.H = np.zeros((self.depth, batch_size, self.hidden_dim))
        self.Alpha = np.zeros((self.depth, batch_size, self.hidden_dim))
        h_prev = np.zeros((batch_size, self.hidden_dim))
        sigmoid_output = np.zeros((self.depth, batch_size, self.output_dim))
        for t in range(self.depth):
            self.Alpha[t] = self.x[t]@self.U + h_prev@self.W
            self.H[t] = self.relu(self.Alpha[t])
            o_t = self.H[t]@self.V
            y_t = self.sigmoid(o_t)
            sigmoid_output[t] = y_t
            h_prev = self.H[t]
        return sigmoid_output

    def backward_prop(self, sigmoid_output, output_label):
        batch_size = output_label.shape[1]
        dU = np.zeros(self.U.shape)
        dW = np.zeros(self.W.shape)
        dV = np.zeros(self.V.shape)
        dH_t_front = np.zeros((batch_size, self.hidden_dim))
        for t in range(self.depth-1, 0, -1):
            dY_t = sigmoid_output[t] - output_label[t]
            dO_t = dY_t * sigmoid_output[t] * (1 - sigmoid_output[t])
            dV += self.H[t].T @ dO_t
            dH_t = dO_t @ self.V.T + dH_t_front
            dAlpha_t = self.relu(self.Alpha[t], dH_t, deriv=True)
            dU += self.x[t].T @ dAlpha_t
            if t > 0:
                dW += self.H[t-1].T @ dAlpha_t
            dH_t_front = dAlpha_t @ self.W.T
        self.U -= self.lr * dU
        self.W -= self.lr * dW
        self.V -= self.lr * dV

    def relu(self, x, front_delta=None, deriv=False):
        if deriv == False:
            return x * (x > 0)
        else:
            back_delta = front_delta * 1. * (x > 0)
            return back_delta

    def sigmoid(self, x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))


def xavier_init(c1, c2, w=1, h=1, fc=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2 * np.random.random((c1, c2, w, h)) - 1)
    if fc:
        params = params.reshape(c1, c2)
    return params


# The X of this dataset is @data_size number of floats lies within (0, 1)
# The Y of this dataset is whether the current prefix sum of X has exceed the value @data_size/2.
# X (length, data_size, 1)
# Y (length, data_size, 1)
def generate_dataset(data_size, length, split_ratio):
    X = np.random.uniform(0, 1, (data_size, length, 1))
    Y = np.zeros((data_size, length, 1))
    threshold = length / 2.
    for i in range(data_size):
        prefix_sum = 0
        for j in range(length):
            prefix_sum += X[i][j][0]
            Y[i][j][0] = int(prefix_sum > threshold)
    split_point = int(data_size * split_ratio)
    train_x, test_x = X[:split_point], X[split_point:]
    train_y, test_y = Y[:split_point], Y[split_point:]
    return np.swapaxes(train_x, 0, 1), np.swapaxes(test_x, 0, 1), \
           np.swapaxes(train_y, 0, 1), np.swapaxes(test_y, 0, 1)


def main():
    length = 12
    data_size = 1000
    split_ratio = 0.9
    max_iter = 100
    iters_before_test = 10
    batch_size = 25
    train_x, test_x, train_y, test_y = generate_dataset(data_size, length, split_ratio)
    rnn = RNN(1, 10, 1, length)
    for iters in range(max_iter+1):
        st_idx = int(iters % ((split_ratio * length) / batch_size))
        ed_idx = int(st_idx + batch_size)
        sigmoid_output = rnn.forward_prop(train_x[:, st_idx:ed_idx, :])
        rnn.backward_prop(sigmoid_output, train_y[:, st_idx:ed_idx, :])
        loss = np.sum((sigmoid_output - train_y[:, st_idx:ed_idx, :]) ** 2)
        print("The loss on training data is %f" % loss)
        if iters % iters_before_test == 0:
            sigmoid_output = rnn.forward_prop(test_x)
            predict_label = sigmoid_output > 0.5
            accuracy = float(np.sum(predict_label == test_y.astype(bool))) / test_y.size
            print("The accuracy on testing data is %f" % accuracy)


if __name__ == '__main__':
    main()
