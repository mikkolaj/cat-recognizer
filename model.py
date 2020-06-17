import numpy as np
from PIL import Image
import os
from numba import jit


def load_validation(normalization=False):
    cur_dir = os.getcwd() + "\\data\\cats-validation\\"
    cur_dir2 = os.getcwd() + "\\data\\other-validation\\"
    num = 1024
    cur_pos = 0
    arr = np.empty(shape=(num, 128, 128, 3))
    for i in range(num // 2):
        img: Image.Image = Image.open(cur_dir + str(i) + ".jpg", "r")
        img = img.resize((128, 128))
        arr[i] = np.array(img, dtype="float32")

        if normalization:
            arr[i] = arr[i] / 255
            mean = arr[i].mean(axis=(0, 1), dtype="float64")
            std = arr[i].std(axis=(0, 1), dtype='float64')
            arr[i] = (arr[i] - mean) / std

        img.close()
    cur_pos -= num // 2
    for i in range(num // 2):
        img: Image.Image = Image.open(cur_dir2 + str(i) + ".jpg", "r")
        img = img.resize((128, 128))
        arr[num // 2 + i] = np.array(img, dtype="float32")

        if normalization:
            arr[num // 2 + i] = arr[num // 2 + i] / 255
            mean = arr[num // 2 + i].mean(axis=(0, 1), dtype="float64")
            std = arr[num // 2 + i].std(axis=(0, 1), dtype='float64')
            arr[num // 2 + i] = (arr[num // 2 + i] - mean) / std

        img.close()
    Y = np.ones(shape=(1, num))
    Y[0, num // 2:] = 0
    arr = arr.reshape(arr.shape[0], -1).T

    return arr, Y


def load_minibatches(num=128, normalization=False):
    cur_dir = os.getcwd() + "\\data\\cats-small\\"
    cur_dir2 = os.getcwd() + "\\data\\other-small\\"
    minibsX = []
    minibsY = []
    cur_pos = 0

    for i in range(70):
        arr = np.empty(shape=(num, 128, 128, 3))
        for i in range(num//2):
            img: Image.Image = Image.open(cur_dir + str(cur_pos) + ".jpg", "r")
            arr[i] = np.array(img, dtype="float32")

            if normalization:
                arr[i] = arr[i]/255
                mean = arr[i].mean(axis=(0, 1), dtype="float64")
                std = arr[i].std(axis=(0, 1), dtype='float64')
                arr[i] = (arr[i] - mean)/std
            img.close()
            cur_pos += 1
        cur_pos -= num//2
        for i in range(num//2):
            img: Image.Image = Image.open(cur_dir2 + str(cur_pos) + ".jpg", "r")
            arr[num//2+i] = np.array(img, dtype="float32")

            if normalization:
                arr[num//2+i] = arr[num//2+i]/255
                mean = arr[num//2+i].mean(axis=(0, 1), dtype="float64")
                std = arr[num//2+i].std(axis=(0, 1), dtype='float64')
                arr[num//2+i] = (arr[num//2+i] - mean)/std

            img.close()
            cur_pos += 1

        Y = np.ones(shape=(1, num))
        Y[0, num//2:] = 0
        arr = arr.reshape(arr.shape[0], -1).T
        minibsX.append(arr)
        minibsY.append(Y)

    return minibsX, minibsY


@jit(nopython=True)
def sigmoid(Z):
    cache = Z
    return 1/(1+np.exp(-Z)), cache


@jit(nopython=True)
def sigmoid_backward(dA, Z):
    s, cache = sigmoid(Z)
    return dA*s*(1-s)


@jit(nopython=True)
def relu(Z):
    cache = Z
    return np.maximum(0, Z), cache


def relu_backward(dA, Z):
    dA[Z <= 0] = 0
    return dA


def param_init_he(layers_dims):
    L = len(layers_dims)
    params = {}
    for l in range(1, L):
        he_factor = np.sqrt(2/layers_dims[l-1])
        params["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        params["b"+str(l)] = np.zeros((layers_dims[l], 1))
    return params


@jit(nopython=True)
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activations_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    caches = (linear_cache, activation_cache)
    return A, caches


def forward_prop(X, params):
    caches = []
    A = X
    L = len(params) // 2  # num of layers
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activations_forward(A_prev, params["W"+str(l)], params["b"+str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activations_forward(A, params["W"+str(L)], params["b"+str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y, params, lmbda, lay_num):
    m = Y.shape[1]
    cost = (np.dot(Y, np.log(AL.T))+np.dot((1-Y), np.log((1-AL).T)))/(-m)

    # calculating regularization term
    regularization = 0
    for i in range(1, lay_num):
        regularization += np.sum(np.square(params["W"+str(i)]))
    regularization *= lmbda/(2*m)

    cost = cost + regularization
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache, lmbda):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    # dW with respect to regualrization
    dW = np.dot(dZ, A_prev.T)/m + lmbda/m*W
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activations_backward(dA, caches, activation, lmbda):
    linear_cache, activation_cache = caches

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache, lmbda)

    return dA_prev, dW, db


def backward_prop(AL, Y, caches, lmbda):
    grads = {}
    L = len(caches) - 1
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    cur_caches = caches[L]
    dA_prev, dW, db = linear_activations_backward(dAL, cur_caches, 'sigmoid', lmbda)
    grads["dA" + str(L)] = dA_prev
    grads["dW" + str(L + 1)] = dW
    grads["db" + str(L + 1)] = db

    for l in range(L-1, -1, -1):
        cur_caches = caches[l]
        dA_prev, dW, db = linear_activations_backward(grads["dA" + str(l + 1)], cur_caches, 'relu', lmbda)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_params(params, grads, learning_rate):
    L = len(params) // 2

    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return params


def teach_the_model(X, Y, layers_dims, learning_rate, iterations, lmbda):
    costs = []

    params = param_init_he(layers_dims)

    for i in range(iterations):
        # forward propagation
        AL, caches = forward_prop(X, params)

        # compute cost for making a graph
        if i % 2 == 0:
            print("Iteration:", i)
            cost = compute_cost(AL, Y, lmbda)
            print(cost)
            costs.append(cost)

        # backward propagation
        grads = backward_prop(AL, Y, caches, lmbda)

        # param update
        params = update_params(params, grads, learning_rate)

    return params


def make_predictions(X, Y, params):
    m = X.shape[1]
    predicts, _ = forward_prop(X, params)
    predicts = predicts > 0.5
    predicts2 = np.array(predicts, copy=True)
    predicts3 = np.array(predicts, copy=True)
    predicts2[Y < 1] = 0
    predicts3[Y == 1] = 0
    print("{}% of non-cats labeled correctly.".format((m/2 - np.sum(predicts3))/(m/2)*100))
    print("{}% of cats were labeled correctly.".format(np.sum(predicts2)/(m/2)*100))
    print(predicts.shape, X.shape, Y.shape)
    print("Accuracy of the model: ", np.sum((predicts == Y))/m)
    return predicts


def shuffle(a, b):
    for i in range(len(a)):
        rng_state = np.random.get_state()
        a[i] = a[i].T
        np.random.shuffle(a[i])
        a[i] = a[i].T
        np.random.set_state(rng_state)
        b[i] = b[i].T
        np.random.shuffle(b[i])
        b[i] = b[i].T
    return a, b


def make_single_prediction():
    file = input("Put your image in the current directory and specify its name (with extension): ")
    directory = os.path.join(os.getcwd(), file)
    img: Image.Image = Image.open(directory, "r")
    img = img.resize((128, 128)).convert("RGB")
    arr = np.array(img, dtype="float64")
    arr.resize((49152, 1))
    params = np.load(os.path.join(os.getcwd(), "params\\params_current.npy"), allow_pickle=True)
    params = params.item()
    predict, _ = forward_prop(arr, params)
    predict = np.squeeze(predict)
    if predict > 0.5:
        print("Prediction: A cat is in the picture.")
    else:
        print("Prediction: There isn't a cat in the picture")


def process_minibatches(miniX, miniY, layers_dims, learning_rate, epochs, lmbda, new_params=True):
    if new_params:
        params = param_init_he(layers_dims)
    else:
        params = np.load(os.path.join(os.getcwd(), "params\\4_4_4_1\\small_reg\\params17.npy"), allow_pickle=True)
        params = params.item()
    counter = 1
    cost = 0

    for i in range(1, epochs + 1):
        for j, minX in enumerate(miniX):

            # forward propagation
            AL, caches = forward_prop(minX, params)

            # compute cost for making a graph
            if j % 20 == 0:
                print("Iteration:", j, "Epoch:", i)
                cost = compute_cost(AL, miniY[j], params, lmbda, len(layers_dims))
                print(cost)

            # backward propagation
            grads = backward_prop(AL, miniY[j], caches, lmbda)

            # param update
            params = update_params(params, grads, learning_rate)

            if i % 50 == 0 and j == 49:
                np.save("params\\4_4_4_1\\small_reg\\params" + str(counter) + ".npy", params)
                file = open("D:\\catrecognizer\\params\\4_4_4_1\\small_reg\\cost" + str(counter) + ".txt", "w")
                file.write(str(cost))
                file.close()
                counter += 1

    return params


if __name__ == "__main__":
    # minibsX, minibsY = load_minibatches(normalization=False)
    # minibsX, minibsY = shuffle(minibsX, minibsY)
    # layers_dims = [minibsX[0].shape[0], 4, 4, 4, 1]
    # params = process_minibatches(minibsX, minibsY, layers_dims, 0.0001, 1000, lmbda=0.0001, new_params=False)

    # if not os.path.exists(os.path.join(os.getcwd(), "params")):
    #     os.makedirs("params")
    # np.save("params/params2.npy", params)

    # validation run
    X, Y = load_validation(normalization=False)
    params = np.load(os.path.join(os.getcwd(), "params\\4_4_4_1\\small\\params_base3.npy"), allow_pickle=True)
    params = params.item()
    make_predictions(X, Y, params)
