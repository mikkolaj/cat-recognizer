import numpy as np
from PIL import Image
import os


def load_data(width=300, height=300, num=198):
    cur_dir = os.getcwd() + "\\cats-combined\\"
    cur_dir2 = os.getcwd() + "\\other\\"
    arr = np.empty(shape=(num, 300, 300, 3))
    for i in range(num//2):
        img: Image.Image = Image.open(cur_dir + str(i) + ".jpg", "r")
        arr[i] = np.array(img)
        img.close()
    for i in range(num//2, num):
        img: Image.Image = Image.open(cur_dir2 + str(i-num//2) + ".jpg", "r")
        arr[i] = np.array(img)
        img.close()
    nums = set()
    test_X = np.empty(shape=(20, 300, 300, 3))
    for i in range(10):
        numy = np.random.randint(100, 200)
        while numy in nums:
            numy = np.random.randint(100, 2000)
        nums.add(numy)
        img: Image.Image = Image.open(cur_dir + str(num) + ".jpg", "r")
        test_X[i] = np.array(img)
        img.close()
    nums = set()
    for i in range(10, 20):
        numy = np.random.randint(100, 200)
        while numy in nums:
            numy = np.random.randint(100, 2000)
        nums.add(numy)
        img: Image.Image = Image.open(cur_dir2 + str(i) + ".jpg", "r")
        test_X[i] = np.array(img)
        img.close()
    X = arr.reshape(arr.shape[0], -1).T
    test_X = test_X.reshape(test_X.shape[0], -1).T
    Y = np.ones(shape=(1, num))
    test_Y = np.ones(shape=(1, 20))
    Y[0, num//2:] = 0
    # print(Y)
    test_Y[0, 10:] = 0
    return X, Y, test_X, test_Y


def sigmoid(Z):
    cache = Z
    # print("Z", Z)
    # input()
    return 1/(1+np.exp(-Z)), cache


def sigmoid_backward(dA, Z):
    s, cache = sigmoid(Z)
    # print("sigmoid backward", s)
    return dA*s*(1-s)


def relu(Z):
    cache = Z
    return np.maximum(0, Z), cache


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    # print(dA.shape)
    # print(dZ.shape)
    dZ[Z <= 0] = 0
    return dZ


def param_init_he(layers_dims):
    L = len(layers_dims)
    params = {}
    for l in range(1, L):
        he_factor = np.sqrt(2/layers_dims[l-1])
        params["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        params["b"+str(l)] = np.zeros((layers_dims[l], 1))
    return params


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


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (np.dot(Y, np.log(AL.T))+np.dot((1-Y), np.log((1-AL).T)))/(-m)
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    # print("dZ, linear backward", dZ)
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)


    return dA_prev, dW, db


def linear_activations_backward(dA, caches, activation):
    linear_cache, activation_cache = caches

    # print("dA linear activations backward", dA)

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    # print("dZ linear activations backward", dZ)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backward_prop(AL, Y, caches):
    grads = {}
    L = len(caches) - 1
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    # print("dalszap backward prop", dAL.shape)
    cur_caches = caches[L]
    dA_prev, dW, db = linear_activations_backward(dAL, cur_caches, 'sigmoid')
    # print(dA_prev.shape)
    grads["dA" + str(L)] = dA_prev
    grads["dW" + str(L + 1)] = dW
    grads["db" + str(L + 1)] = db

    for l in range(L-1, -1, -1):
        cur_caches = caches[l]
        # print(grads["dA" + str(l + 1)].shape)
        dA_prev, dW, db = linear_activations_backward(grads["dA" + str(l + 1)], cur_caches, 'relu')
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_params(params, grads, learning_rate):
    L = len(params) // 2

    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    # print("params", params)
    # input()
    return params


def teach_the_model(X, Y, layers_dims, learning_rate, iterations):
    costs = []

    params = param_init_he(layers_dims)

    for i in range(iterations):
        # forward propagation
        AL, caches = forward_prop(X, params)
        # print("AL, params", AL, params)

        # compute cost for making a graph
        if i % 2 == 0:
            print("Iteration:", i)
            cost = compute_cost(AL, Y)
            print(cost)
            costs.append(cost)

        # backward propagation
        grads = backward_prop(AL, Y, caches)

        # param update
        params = update_params(params, grads, learning_rate)

    return params


def make_predictions(X, Y, params):
    m = X.shape[1]
    predicts, _ = forward_prop(X, params)
    predicts = predicts > 0.5
    print(predicts.shape, X.shape, Y.shape)
    print("Accuracy of the model: ", np.sum((predicts == Y))/m)
    return predicts


def shuffle(a, b):
    rng_state = np.random.get_state()
    a = a.T
    np.random.shuffle(a)
    a = a.T
    np.random.set_state(rng_state)
    b = b.T
    np.random.shuffle(b)
    b = b.T
    return a, b


def make_single_prediction():
    file = input("Put your image in the current directory and specify its name (with extension): ")
    directory = os.path.join(os.getcwd(), file)
    img: Image.Image = Image.open(directory, "r")
    img = img.resize((300, 300)).convert("RGB")
    arr = np.array(img)
    arr.resize((270000, 1))
    params = np.load(os.path.join(os.getcwd(), "params\\params.npy"), allow_pickle=True)
    params = params.item()
    predict, _ = forward_prop(arr, params)
    predict = np.squeeze(predict)
    if predict > 0.5:
        print("Prediction: A cat is in the picture.")
    else:
        print("Prediction: There isn't a cat in the picture")


if __name__ == "__main__":
    X, Y, test_X, test_Y = load_data()
    X, Y = shuffle(X, Y)
    layers_dims = [X.shape[0], 8, 4, 1]
    params = teach_the_model(X, Y, layers_dims, 0.00008, 3000)
    if not os.path.exists(os.path.join(os.getcwd(), "params")):
        os.makedirs("params")
    np.save("params/params.npy", params)
    make_predictions(test_X, test_Y, params)
