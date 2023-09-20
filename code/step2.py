'''Step 2: Expands from a single layer neural network to a 2-layer with a hidden layer.
   Updated gradients, new activation functions, and data augmentation!'''
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

''' Loads datasets into numpy arrays'''
def LoadBatch(filename):
    import pickle
    with open('C:/KTH/numpy-neural-networks/code/Datasets/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.array(dict[b'data'])
    y = np.array(dict[b'labels'])
    Y = np.zeros((y.size, 10))
    Y[np.arange(y.size),y] = 1
    
    
    X = np.transpose(np.array(X).astype(float))
    Y = np.transpose(np.array(Y).astype(float))
    y = np.transpose(y)
    
    return X, Y, y

'''Fetch the necessary data and labels (y = labels, Y = one hot labels)'''
def GetAllData(big_batch=False):
    if(big_batch):
            train_X1, train_Y1, train_y1 = LoadBatch("data_batch_1")
            train_X2, train_Y2, train_y2 = LoadBatch("data_batch_2")
            train_X3, train_Y3, train_y3 = LoadBatch("data_batch_3")
            train_X4, train_Y4, train_y4 = LoadBatch("data_batch_4")
            train_X5, train_Y5, train_y5 = LoadBatch("data_batch_5")

            train_X = numpy.concatenate((train_X1, train_X2, train_X3, train_X4, train_X5[:,:9000]), 1)
            train_Y = numpy.concatenate((train_Y1, train_Y2, train_Y3, train_Y4, train_Y5[:,:9000]), 1)
            train_y = numpy.concatenate((train_y1, train_y2, train_y3, train_y4, train_y5[:9000]), 0)

            val_X = train_X5[:,9000:]
            val_Y = train_Y5[:,9000:]
            val_y = train_y5[9000:]

            test_X, test_Y, test_y = LoadBatch("test_batch")
    else:
        train_X, train_Y, train_y = LoadBatch("data_batch_1")
        val_X, val_Y, val_y = LoadBatch("data_batch_2")
        test_X, test_Y, test_y = LoadBatch("test_batch")

    '''Normalise all data based on training'''
    train_mean = np.mean(train_X, 1)
    train_std = np.std(train_X, 1)
    
    mean = np.transpose(np.matlib.repmat(train_mean, np.matlib.size(train_X, 1), 1))
    std = np.transpose(np.matlib.repmat(train_std, np.matlib.size(train_X, 1), 1))
    train_X = np.divide(np.subtract(train_X, mean), std)
    
    mean = np.transpose(np.matlib.repmat(train_mean, np.matlib.size(val_X, 1), 1))
    std = np.transpose(np.matlib.repmat(train_std, np.matlib.size(val_X, 1), 1))
    val_X = np.divide(np.subtract(val_X, mean), std)

    mean = np.transpose(np.matlib.repmat(train_mean, np.matlib.size(test_X, 1), 1))
    std = np.transpose(np.matlib.repmat(train_std, np.matlib.size(test_X, 1), 1))
    test_X = np.divide(np.subtract(test_X, mean), std)
    
    return train_X, train_Y, train_y, val_X, val_Y, val_y, test_X, test_Y, test_y

'''Display the image for each label in W'''
def montage(W):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[i*5+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j + 1))
            ax[i][j].axis('off')
    plt.show() 

''' Standard definition of the softmax function '''
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

'''Returns activations h and probabilities p of each'''
def EvaluateClassifier(X, W1, W2, b1, b2, return_activations = False):
    s1 = np.matmul(W1,X) + b1
    h = np.maximum(s1, 0) # ReLu activation function
    if(return_activations):
        return h, softmax(np.matmul(W2, h) + b2)
    
    return softmax(np.matmul(W2, h) + b2)

'''Calculates the cost function for a batch of predictions'''
def ComputeCost(X, Y, W1, W2, b1, b2, lam):
    P = EvaluateClassifier(X, W1, W2, b1, b2)
    N = np.size(P,1)
    loss_term = [np.dot(Y[:,i], np.log(P[:,i])) for i in range(N)]
    loss_term = - sum(loss_term) / N
    reg_term = (np.sum(np.square(W1)) + np.sum(np.square(W2)))* lam
    return loss_term + reg_term, loss_term    

def ComputeAccuracy(X, y, W1, W2, b1, b2):
    p = EvaluateClassifier(X, W1, W2, b1, b2)
    guesses = np.argmax(p, 0)
    corrects = np.sum((y==guesses).astype(float))
    return corrects / np.size(y, 0)

'''Analytically computes gradients of the parameters W1, W2, b1, b2'''
def ComputeGradients(X, Y, W1, W2, b1, b2, lam, dropout=False):
    nb = np.size(X, 1)
    keepProb = 0.5   # P = 1 - keepProb
    
    H_batch = np.maximum(np.matmul(W1, X) + b1, 0)
    
    if(dropout):
        # DROPOUT FORWARD
        D = np.random.rand(H_batch.shape[0], H_batch.shape[1])
        D = D < keepProb
        H_batch = np.multiply(H_batch, D)
        H_batch = H_batch/keepProb
        # DROPOUT FORWARD
    P_batch = softmax(np.matmul(W2, H_batch) + b2)
    G_batch = - (Y - P_batch)
    
    dL_dW2 = np.matmul(G_batch, np.transpose(H_batch)) / nb
    dJ_dW2 = dL_dW2 + 2*lam*W2
    dJ_db2 = np.matmul(G_batch, np.ones(((nb, 1)))) / nb
    
    G_batch = np.matmul(np.transpose(W2), G_batch)
    if(dropout):
        # DROPOUT BACKWARD
        G_batch = np.multiply(D, G_batch)
        G_batch = G_batch/keepProb
        # DROPOUT BACKWARD
    ind = (H_batch > 0).astype(int)
    G_batch = numpy.multiply(G_batch, ind)
    
    dL_dW1 = np.matmul(G_batch, np.transpose(X)) / nb
    dJ_dW1 = dL_dW1 + 2*lam*W1
    dJ_db1 = np.matmul(G_batch, np.ones(((nb, 1)))) / nb
    
    return dJ_dW1, dJ_dW2, dJ_db1, dJ_db2

''' Slow but accurate numerical computation to verify the analytical ComputeGradients(). Converted from Matlab code.'''
def ComputeGradsNumSlow(X, Y, W1, W2, b1, b2, lamda, h):
    no1 = W1.shape[0]
    no2 = W2.shape[0]
    d = X.shape[0]

    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros((no1, 1))
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros((no2, 1))

    c, _ = ComputeCost(X, Y, W1, W2, b1, b2, lamda);
    
    for i in range(len(b1)):
        b1_try = np.array(b1)
        b1_try[i] += h
        c2, _ = ComputeCost(X, Y, W1, W2, b1_try, b2, lamda)
        b1_try[i] -= 2*h
        c1, _ = ComputeCost(X, Y, W1, W2, b1_try, b2, lamda)
        grad_b1[i] = (c2-c1) / (2*h)
    
    for i in range(len(b2)):
        b2_try = np.array(b2)
        b2_try[i] += h
        c2, _ = ComputeCost(X, Y, W1, W2, b1, b2_try, lamda)
        b2_try[i] -= 2*h
        c1, _ = ComputeCost(X, Y, W1, W2, b1, b2_try, lamda)
        grad_b2[i] = (c2-c1) / (2*h)
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i,j] += h
            c2, _ = ComputeCost(X, Y, W1_try, W2, b1, b2, lamda)
            W1_try[i,j] -= 2*h
            c1, _ = ComputeCost(X, Y, W1_try, W2, b1, b2, lamda)
            grad_W1[i,j] = (c2-c1) / (2*h)
    
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i,j] += h
            c2, _ = ComputeCost(X, Y, W1, W2_try, b1, b2, lamda)
            W2_try[i,j] -= 2*h
            c1, _ = ComputeCost(X, Y, W1, W2_try, b1, b2, lamda)
            grad_W2[i,j] = (c2-c1) / (2*h)
    
    return grad_W1, grad_W2, grad_b1, grad_b2

'''Initialise W1, W2, b1, b2 randomly'''
def InitialiseParameters(K, d, m):
    # K = Num of labels
    # d = Size of image
    # m = Hidden nodes
    
    W1 = 0.01 * np.random.randn(m, d)
    W2 = 0.01 * np.random.randn(K, m)
    b1 = 0.01 * np.random.randn(m, 1)
    b2 = 0.01 * np.random.randn(K, 1)
    return W1, W2, b1, b2

'''Performs a learning episode using mini batches'''
def MiniBatchGD(X, Y, W1, W2, b1, b2, lam, n_batch, n_epochs):
    augment = True
    if augment:
        variants = DataAugmentation(X)
    N = np.size(Y, 1)
    num_per_epoch = int(N/n_batch)
    eta_min = 1e-5
    eta_max = 1e-1
    eta_t = eta_min
    n_s = 1000 #1000 in optimal
    t = 0
    T = 10*n_s
    
    train_costs = []
    train_loss = []
    val_costs = []
    val_loss = []
    train_acc = []
    val_acc = []
    res = [train_costs, train_loss, val_costs, val_loss, train_acc, val_acc]
   
    
    
    for i in range(n_epochs):
        
        #print(f'Epoch {i}')
        rand = np.random.permutation(N)
        for j in range(num_per_epoch):
            if(t % (2*n_s) < n_s): # Increases for n_s cycles, decreases for n_s, and so on
                eta_t += (eta_max - eta_min) / n_s
            else:
                eta_t -= (eta_max - eta_min) / n_s
                
            if(t % (2*n_s) == n_s):
                #eta_max starts at its max value and ends at eta_min
                print(f'Decreasing from eta = {eta_t} during epoch {i}')
            elif(t % (2*n_s) == 0):
                print(f'Increasing from eta = {eta_t} during epoch {i}')
                #eta_max = (T - t)/T * (eta_max - eta_min) + eta_min
            t += 1
            start = j*n_batch
            end = (j+1)*n_batch - 1
            indices = rand[start:end]
            if augment:
                X_batch = GetAugmentedBatch(variants, indices)
            else:
                X_batch = X[:,indices]
            Y_batch = Y[:,indices]
            
            dJ_dW1, dJ_dW2, dJ_db1, dJ_db2 = ComputeGradients(X_batch, Y_batch, W1, W2, b1, b2, lam)
            W1 -= eta_t * dJ_dW1
            b1 -= eta_t * dJ_db1
            W2 -= eta_t * dJ_dW2
            b2 -= eta_t * dJ_db2
            
            #'''
            if(t%100 == 1):
                #print(f't = {t} out of {4*n_s}')
                tcost, tloss = ComputeCost(train_X, train_Y, W1, W2, b1, b2, lam)
                vcost, vloss = ComputeCost(val_X, val_Y, W1, W2, b1, b2, lam)
                train_costs.append(tcost)
                train_loss.append(tloss)
                val_costs.append(vcost)
                val_loss.append(vloss)
            #'''
            if(t > T): # 2ns per cycle, 6ns in optimal
                return W1, W2, b1, b2, res  
        '''
        tcost, tloss = ComputeCost(train_X, train_Y, W1, W2, b1, b2, lam)
        vcost, vloss = ComputeCost(val_X, val_Y, W1, W2, b1, b2, lam)
        train_costs.append(tcost)
        train_loss.append(tloss)
        val_costs.append(vcost)
        val_loss.append(vloss)
        #'''
    return W1, W2, b1, b2, res

'''Creates and stores mirror and translated images for each training data. These will be called on at random during training.
Each image has 7*7*2 = 98 variants (7 tx, 7 ty, 2 for flipped states'''
def DataAugmentation(X):
    flip = [X, MirrorImages(X)]
    variants = []
    for i in range(2):
        for tx in range(-1, 2):
            for ty in range(-1, 2):
                variants.append(ShiftImages(flip[i], 3*tx, 3*ty))
    return np.array(variants)

'''Returns randomly flipped/translated versions of the batch defined by indices'''
def GetAugmentedBatch(variants, indices):
    versions = np.random.randint(0, 18, indices.shape)
    thing = np.transpose(variants[versions,:,indices])
    return thing
    
'''Creates vertically flipped image versions'''
def MirrorImages(X):
    aa_32 = [32*x for x in range(32)]
    bb = [31-x for x in range(32)]
    vv = np.matlib.repmat(aa_32, 32, 1)
    bb_mat = np.transpose(np.matlib.repmat(bb, 1, 32))
    vv = np.transpose(np.matrix((vv.flatten('F'))))
    ind_flip = vv + bb_mat
    inds_flip = np.concatenate((ind_flip, 1024+ind_flip, 2048+ind_flip))
    mirrored_X = numpy.squeeze(X[inds_flip])
    return mirrored_X

'''Shifts images translationally'''
def ShiftImages(X, tx, ty):
    aa_32 = [32*x for x in range(32)]
    if(tx < 0):
        vv = np.matlib.repmat(aa_32, 32+tx, 1)
        bb1 = [x for x in range(32+tx)] #Have done -1 for 0-index
        bb2 = [x for x in range(-tx, 32)]
    else:
        vv = np.matlib.repmat(aa_32, 32-tx, 1)
        bb1 = [x for x in range(tx+1-1, 32)] #Have done -1 for 0-index
        bb2 = [x for x in range(32-tx)]
    bb1_mat = np.transpose(np.matlib.repmat(bb1, 1, 32)) #swap transpose and repmat?
    bb2_mat = np.transpose(np.matlib.repmat(bb2, 1, 32))
    vv_dots = np.transpose(np.matrix(vv.flatten('F')))
    ind_fill = vv_dots + bb1_mat
    ind_xx = vv_dots + bb2_mat
    
    if(ty < 0):
        ii = numpy.where(ind_fill < 1024 + ty*32)[0]
        ii2 = numpy.where(ind_xx >= -ty*32)[0]
        ind_fill = ind_fill[:ii[ii.shape[0]-1]+1]
        ind_xx = ind_xx[ii2[0]:]
    else:
        ii = numpy.where(ind_fill >= ty*32)[0]
        ii2 = numpy.where(ind_xx <= 1024-ty*32-1)[0]
        ind_fill = ind_fill[ii[0]:]
        ind_xx = ind_xx[:ii2[ii2.shape[0]-1]+1]

    inds_fill = np.concatenate((ind_fill, 1024+ind_fill, 2048+ind_fill))
    inds_xx = np.concatenate((ind_xx, 1024+ind_xx, 2048+ind_xx))

    shifted_X = np.zeros(X.shape)
    shifted_X[inds_fill] = X[inds_xx]
    return shifted_X

'''Grid search for a good lambda value'''
def FindLambda(lambdas):
    train_X, train_Y, train_y, val_X, val_Y, val_y, test_X, test_Y, test_y = GetAllData(True)
    val_accs = []
    
    for lam in lambdas:
        d = np.size(train_X, 0)
        K = np.size(train_Y, 0)
        m = 50
        W1, W2, b1, b2 = InitialiseParameters(K, d, m)
        
        n_epochs = 100
        n_batch = 100
        
        W1_star, W2_star, b1_star, b2_star, res = MiniBatchGD(train_X, train_Y, W1, W2, b1, b2, lam, n_batch, n_epochs)
        val_acc = 100*ComputeAccuracy(val_X, val_y, W1_star, W2_star, b1_star, b2_star)
        print(f"Validation accuracy: {val_acc}% with λ = {lam}, batch of {n_batch}, {n_epochs} epochs")
        val_accs.append(val_acc)
    
    return val_accs

#%% Cell 1: Testing parameters and analytical gradient against numerical
train_X, train_Y, train_y, val_X, val_Y, val_y, test_X, test_Y, test_y = GetAllData(True)
lam = 0.0001

batch_X = train_X[0:20, 0:1]
batch_Y = train_Y[:, 0:1]
batch_y = train_y[0:1]
d = np.size(batch_X, 0)
K = np.size(batch_Y, 0)
m = 50

W1, W2, b1, b2 = InitialiseParameters(K, d, m)

grad_W1, grad_W2, grad_b1, grad_b2 = ComputeGradients(batch_X, batch_Y, W1, W2, b1, b2, lam)
num_grad_W1, num_grad_W2, num_grad_b1, num_grad_b2 = ComputeGradsNumSlow(batch_X, batch_Y, W1, W2, b1, b2, lam, 1e-5)

W1_diff = grad_W1 - num_grad_W1 
print(f"Largest absolute error in W1: {np.max(W1_diff)}")
b1_diff = grad_b1 - num_grad_b1
print(f"Largest absolute error in b1: {np.max(b1_diff)}")
W2_diff = grad_W2 - num_grad_W2 
print(f"Largest absolute error in W2: {np.max(W2_diff)}")
b2_diff = grad_b2 - num_grad_b2
print(f"Largest absolute error in b2: {np.max(b2_diff)}")
enum = np.abs(np.sum(grad_W1 - num_grad_W1))
denom = np.max((0.0001, np.abs(np.sum(grad_W1)) + np.abs(np.sum(num_grad_W1))))
print(f"Relative error in W1 gradient: {enum/denom}") # This should always return a small value 
enum = np.abs(np.sum(grad_b1 - num_grad_b1))
denom = np.max((0.0001, np.abs(np.sum(grad_b1)) + np.abs(np.sum(num_grad_b1))))
print(f"Relative error in b1 gradient: {enum/denom}")
enum = np.abs(np.sum(grad_W2 - num_grad_W2))
denom = np.max((0.0001, np.abs(np.sum(grad_W2)) + np.abs(np.sum(num_grad_W2))))
print(f"Relative error in W2 gradient: {enum/denom}") # This should always return a small value 
enum = np.abs(np.sum(grad_b2 - num_grad_b2))
denom = np.max((0.0001, np.abs(np.sum(grad_b2)) + np.abs(np.sum(num_grad_b2))))
print(f"Relative error in b2 gradient: {enum/denom}")

#%% Cell 2: Sanity check overfitting the loss

batch_X = train_X[:, 0:100]
batch_Y = train_Y[:, 0:100]
batch_y = train_y[0:100]
d = np.size(batch_X, 0)
K = np.size(batch_Y, 0)
m = 50

W1, W2, b1, b2 = InitialiseParameters(K, d, m)
lam = 0
n_epochs = 100
n_batch = 10
W1_star, W2_star, b1_star, b2_star, res = MiniBatchGD(batch_X, batch_Y, W1, W2, b1, b2, lam, n_batch, n_epochs, batch_y)

tl = res[1]
x = [i+1 for i in range(n_epochs)]

plt.plot(x, tl, label="Loss (Training)")
plt.xlabel("Epoch")
plt.xlim(0, n_epochs+1)
plt.legend()
plt.show()


#%% Cell 3: Perform a fine search for lambda

l_min = -4
l_max = -3
zerotoone = np.random.uniform(1.40, 1.90, 30)
lambdas = [val * np.power(10.0, -4) for val in zerotoone]
accs = FindLambda(lambdas)

plt.scatter(lambdas, accs)
plt.title("Change in validation accuracy for different lambdas")
plt.xlabel("Lambda")
plt.ylabel("Validation accuracy (%)")
plt.show()

#%% Cell 4: Sampling the data augmentation functions

train_X, train_Y, train_y, val_X, val_Y, val_y, test_X, test_Y, test_y = GetAllData(False)
variants = DataAugmentation(train_X)
#%%
images = np.array([100, 120, 2, 4, 1, 5, 7, 8, 9, 9])
thing = GetAugmentedBatch(variants, images)
montage(thing)

#%% Cell 5: Train a network!

train_X, train_Y, train_y, val_X, val_Y, val_y, test_X, test_Y, test_y = GetAllData(True)
train_X = train_X[:,:1000]
train_Y = train_Y[:,:1000]

d = np.size(train_X, 0)
K = np.size(train_Y, 0)
m = 500 # 50 in optimal
W1, W2, b1, b2 = InitialiseParameters(K, d, m)

lam = 0.0001
#lam = 0.000164 #Optimal with m = 50 according to search
n_epochs = 100
n_batch = 100

W1_star, W2_star, b1_star, b2_star, res = MiniBatchGD(train_X, train_Y, W1, W2, b1, b2, lam, n_batch, n_epochs)
test_acc = 100*ComputeAccuracy(test_X, test_y, W1_star, W2_star, b1_star, b2_star)
print(f"Test accuracy: {test_acc}% with λ = {lam}, batch of {n_batch}, {n_epochs} epochs, {m} hidden nodes")


tc = res[0]
tl = res[1]
vc = res[2]
vl = res[3]

x = [100*i for i in range(len(tc))]

plt.plot(x, tl, label="Loss (Training)")
plt.plot(x, vl, label="Loss (Validation)")
plt.title(f"Changes in loss over time \n (λ={lam}, {m} hidden nodes)")
plt.xlabel("Step")
plt.xlim(0, 100*len(tc)+1)
plt.legend()
plt.show()

'''
plt.plot(x, tc, label="Cost (Training)")
plt.plot(x, vc, label="Cost (Validation)")
plt.title(f"Changes in cost over time \n (λ={lam})")
plt.xlabel("Update step")
plt.xlim(0, 160*len(tc)+1)
plt.legend()
plt.show()

plt.plot(x, ta, label="Accuracy (Training)")
plt.plot(x, va, label="Accuracy (Validation)")
plt.title(f"Changes in accuracy over time \n (λ={lam})")
plt.xlabel("Update step")
plt.xlim(0, 160*len(tc)+1)
plt.legend()
plt.show()
'''

#montage(W1_star)
#montage(W2_star)
    

