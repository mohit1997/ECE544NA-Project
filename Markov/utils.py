import numpy as np

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def generate_single_output_data(series,batch_size,time_steps):
    series = series.reshape(-1)
    series = series.copy()
    data = strided_app(series, time_steps+1, 1)
    l = int(len(data)/batch_size) * batch_size

    # data = data[:l]
    X = data[:, :-1]
    Y = data[:, -1:]

    return X,Y
