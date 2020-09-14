import numpy as np

def zero_pad(X, pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev*W
    Z = np.sum(s)
    Z = Z+np.float(b)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    n_H = (int((n_H_prev - f + (2 * pad)) / stride)) + 1
    n_W = (int((n_W_prev - f + (2 * pad)) / stride)) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev,pad)
    for i in range(m):              
        a_prev_pad = A_prev_pad[i,:,:,:]              
        for h in range(n_H):           
            vert_start = h * stride
            vert_end = h * stride + f
            for w in range(n_W):       
                horiz_start = w * stride
                horiz_end = w * stride + f
                for c in range(n_C):   
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    weights = W
                    biases = b
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights[:,:,:,c], biases[:,:,:,c])
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    return Z, cache

np.random.seed(1)
A_prev = np.random.randn(10,5,7,4)
W = np.random.randn(3,3,4,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 1,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

def pool_forward(A_prev, hparameters_pool, mode = "max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters_pool["f"]
    stride = hparameters_pool["stride"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))              
    for i in range(m):                         
        for h in range(n_H):
            vert_start = h * stride        
            vert_end = h * stride + f
            for w in range(n_W):                 
                horiz_start = w * stride
                horiz_end = w * stride + f
                
                for c in range (n_C):            
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache

hparameters_pool = {"stride" : 1, "f": 3}

A, cache = pool_forward(A_prev, hparameters_pool)
