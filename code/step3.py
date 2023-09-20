import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def LoadBatch(filename):
    import pickle
    """ Copied from the dataset website """
    with open('C:/KTH/Deep Learning/Datasets/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.array(dict[b'data'])
    y = np.array(dict[b'labels'])
    Y = np.zeros((y.size, 10))
    Y[np.arange(y.size),y] = 1
    
    
    X = np.transpose(np.array(X).astype(float))
    Y = np.transpose(np.array(Y).astype(float))
    y = np.transpose(y)
    
    return X, Y, y

def GetAllData(big_batch=False):
    '''Fetch the necessary data and labels (y = labels, Y = one hot labels)'''
    if(big_batch):
            train_X1, train_Y1, train_y1 = LoadBatch("data_batch_1")
            train_X2, train_Y2, train_y2 = LoadBatch("data_batch_2")
            train_X3, train_Y3, train_y3 = LoadBatch("data_batch_3")
            train_X4, train_Y4, train_y4 = LoadBatch("data_batch_4")
            train_X5, train_Y5, train_y5 = LoadBatch("data_batch_5")

            train_X = numpy.concatenate((train_X1, train_X2, train_X3, train_X4, train_X5[:,:5000]), 1)
            train_Y = numpy.concatenate((train_Y1, train_Y2, train_Y3, train_Y4, train_Y5[:,:5000]), 1)
            train_y = numpy.concatenate((train_y1, train_y2, train_y3, train_y4, train_y5[:5000]), 0)

            val_X = train_X5[:,5000:]
            val_Y = train_Y5[:,5000:]
            val_y = train_y5[5000:]

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


def montage(W):
    
    """ Display the image for each label in W """
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
    
def montage_images(W):
    
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[:,i*5+j].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j + 1))
            ax[i][j].axis('off')
    plt.show()    

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def BatchNormalise(S, provided=False, mu=None, var=None):
    if not provided:
        mu = np.mean(S, 1)[:, None]
        var = np.var(S, 1)[:, None] + 1e-15 # Add small value to prevent div by 0
    
    std = np.sqrt(var) 
    S_norm = np.divide(np.subtract(S, mu), std)
    return S_norm, mu, var


def EvaluateClassifier(X, parameters, all_scores = False, stat_params=None):
    '''Returns activations h and probabilities p of each'''
    Ws = parameters.get("W")
    bs = parameters.get("b")
    gammas = parameters.get("gamma")
    betas = parameters.get("beta")
    layers = (int) (len(Ws))
    scores = {}
    s_scores = []
    s_hat_scores = []
    x_scores = []
    means = []
    variances = []
    
    x_i = X
    for i in range(layers-1):
        s_i = np.matmul(Ws[i], x_i) + bs[i]
        s_scores.append(np.copy(s_i))
        
        if(stat_params == None):
            s_hat_i, mu, v = BatchNormalise(s_i)
        else:
            mu = stat_params.get("mean")[i]
            v = stat_params.get("variance")[i]
            s_hat_i, _, _ = BatchNormalise(s_i, True, mu, v)
        s_hat_scores.append(np.copy(s_hat_i))
        means.append(mu)
        variances.append(v)
        
        s_dash_i = np.multiply(gammas[i], s_hat_i) + betas[i]
        x_i = np.maximum(s_dash_i, 0) # ReLu activation function
        x_scores.append(np.copy(x_i))
    s_i =  np.matmul(Ws[(layers-1)], x_i) + bs[(layers-1)]
    p_i = softmax(s_i)
    if(all_scores):
        scores["x"] = x_scores
        scores["s"] = s_scores
        scores["s_hat"] = s_hat_scores
        scores["mean"] = means
        scores["variance"] = variances
        return p_i, scores
    return p_i

def ComputeCost(X, Y, parameters, lam, stats=None):
    '''Calculates the cost function for a batch of predictions'''
    Ws = parameters.get("W")
    layers = (int) (len(Ws))
    P = EvaluateClassifier(X, parameters, False, stats)
    N = np.size(P,1)
    loss_term = [np.dot(Y[:,i], np.log(P[:,i])) for i in range(N)]
    loss_term = - sum(loss_term) / N
    reg_term = 0
    for i in range(layers):
        reg_term += np.sum(np.square(Ws[i]))
    reg_term = lam * reg_term
    return loss_term + reg_term, loss_term    

def ComputeAccuracy(X, y, parameters, stats=None):
    p = EvaluateClassifier(X, parameters, False, stats)
    guesses = np.argmax(p, 0)
    corrects = np.sum((y==guesses).astype(float))
    return corrects / np.size(y, 0)

def BatchNormBackPass(G, s, mu, v, nb):
    v_e = v + 1e-15
    v_e_sqrt = np.sqrt(v_e)
    sig1 = np.reciprocal(v_e_sqrt)
    sig1_big = np.matmul(sig1, np.ones((1, nb)))
    G1 = np.multiply(G, sig1_big)
    
    v_e_3_2 = v_e * v_e_sqrt
    sig2 = np.reciprocal(v_e_3_2)
    sig2_big = np.matmul(sig2, np.ones((1, nb)))
    G2 = np.multiply(G, sig2_big)
    
    D = s - np.matmul(mu, np.ones((1, nb)))
    G2D = np.multiply(G2, D)
    c = np.matmul(G2D, np.ones((nb, 1)))
    
    G11 = np.matmul(G1, np.ones((nb, 1)))
    G111 = np.matmul(G11, np.ones((1, nb)))
    
    c1 = np.matmul(c, np.ones((1, nb)))
    Doc1 = np.multiply(D, c1)
    
    new_G = G1 - G111/nb - Doc1/nb
    
    return new_G

def ComputeGradients(X, Y, parameters, lam):
    '''Computes gradients of the parameters W_i, b_i'''
    Ws = parameters.get("W")
    gammas = parameters.get("gamma")
    layers = (int) (len(Ws))
    nb = np.size(X, 1)
    gradients = {}
    W_gradients = []
    b_gradients = []
    gamma_gradients = []
    beta_gradients = []
    
    P_batch, scores = EvaluateClassifier(X, parameters, True)
    x = scores.get("x")
    s = scores.get("s")
    s_hat = scores.get("s_hat")
    mu = scores.get("mean")
    v = scores.get("variance")
    
    G_batch = - (Y - P_batch)
    
    for i in range(layers-2, -1, -1):
        # W_grad and b_grad
        dJ_db_i = np.matmul(G_batch, np.ones(((nb, 1)))) / nb #b grad
        dJ_dW_i = np.matmul(G_batch, np.transpose(x[i]))/nb #W grad
        dJ_dW_i = dJ_dW_i + 2*lam*Ws[i+1] #W grad
        b_gradients.append(dJ_db_i)
        W_gradients.append(dJ_dW_i)
        
        #Prop to previous layer
        G_batch = np.matmul(np.transpose(Ws[i+1]), G_batch)
        ind = (x[i] > 0).astype(int)
        G_batch = np.multiply(G_batch, ind)
        
        #gamma_grad and beta_grad
        G_S_hat = np.multiply(G_batch, s_hat[i])
        dJ_dgamma_i = np.matmul(G_S_hat, np.ones(((nb, 1)))) / nb
        dJ_dbeta_i = np.matmul(G_batch, np.ones(((nb, 1)))) / nb
        gamma_gradients.append(dJ_dgamma_i)
        beta_gradients.append(dJ_dbeta_i)
        
        #Prop past scale/shift
        G_batch = np.multiply(G_batch, np.matmul(gammas[i], np.ones((1, nb))))
        
        #Prop through batch norm
        G_batch = BatchNormBackPass(G_batch, s[i], mu[i], v[i], nb)
        
    
    # Final W_grad and b_grad
    dJ_db_1 = np.matmul(G_batch, np.ones(((nb, 1)))) / nb
    dJ_dW_1 = np.matmul(G_batch, np.transpose(X)) / nb
    dJ_dW_1 = dJ_dW_1 + 2*lam*Ws[0]
    b_gradients.append(dJ_db_1)
    W_gradients.append(dJ_dW_1)
    
    b_gradients.reverse()
    W_gradients.reverse()
    gamma_gradients.reverse()
    beta_gradients.reverse()
    
    gradients["W"] = W_gradients
    gradients["b"] = b_gradients
    gradients["gamma"] = gamma_gradients
    gradients["beta"] = beta_gradients
    gradients["mean"] = np.copy(mu)
    gradients["variance"] = np.copy(v)
    gradients =  {"W": W_gradients, "b" : b_gradients, 
                  "gamma" : gamma_gradients, "beta" : beta_gradients,
                  "mean" : np.copy(mu), "variance" : np.copy(v)}
    return gradients

def ComputeGradsNumSlow(X, Y, parameters, lamda, h):
    '''Numerical gradient estimation (Very slow for large data)'''
    Ws = parameters.get("W")
    bs = parameters.get("b")
    gammas = parameters.get("gamma")
    betas = parameters.get("beta")
    layers = (int) (len(Ws))
    numerical_grads = {}
    W_gradients = []
    b_gradients = []
    gamma_gradients = []
    beta_gradients = []
    
    c, _ = ComputeCost(X, Y, parameters, lamda);
    
    for i in range(layers):
        Wi = Ws[i]
        bi = bs[i]
        grad_Wi = np.zeros(Wi.shape)
        grad_bi = np.zeros((Wi.shape[0], 1))  
        
        for i in range(Wi.shape[0]):
            for j in range(Wi.shape[1]):
                Wi[i,j] += h
                c2, _ = ComputeCost(X, Y, parameters, lamda)
                Wi[i,j] -= 2*h
                c1, _ = ComputeCost(X, Y, parameters, lamda)
                Wi[i,j] += h #Reset value for next calculation
                grad_Wi[i,j] = (c2-c1) / (2*h)
        W_gradients.append(np.copy(grad_Wi))
        
        for i in range(len(bi)):
            bi[i] += h
            c2, _ = ComputeCost(X, Y, parameters, lamda)
            bi[i] -= 2*h
            c1, _ = ComputeCost(X, Y, parameters, lamda)
            bi[i] += h
            grad_bi[i] = (c2-c1) / (2*h)
        b_gradients.append(np.copy(grad_bi))
    
    for i in range(layers-1):
        gamma_i = gammas[i]
        beta_i = betas[i]
        grad_gamma = np.zeros((gamma_i.shape[0], 1)) 
        grad_beta = np.zeros((beta_i.shape[0], 1)) 
        
        for i in range(len(gamma_i)):
            gamma_i[i] += h
            c2, _ = ComputeCost(X, Y, parameters, lamda)
            gamma_i[i] -= 2*h
            c1, _ = ComputeCost(X, Y, parameters, lamda)
            gamma_i[i] += h
            grad_gamma[i] = (c2-c1) / (2*h)
        gamma_gradients.append(np.copy(grad_gamma))    
        
        for i in range(len(beta_i)):
            beta_i[i] += h
            c2, _ = ComputeCost(X, Y, parameters, lamda)
            beta_i[i] -= 2*h
            c1, _ = ComputeCost(X, Y, parameters, lamda)
            beta_i[i] += h
            grad_beta[i] = (c2-c1) / (2*h)
        beta_gradients.append(np.copy(grad_beta))
        
        
    numerical_grads["W"] = W_gradients
    numerical_grads["b"] = b_gradients
    numerical_grads["gamma"] = gamma_gradients
    numerical_grads["beta"] = beta_gradients
    return numerical_grads

def InitialiseParameters(dims, mode="Standard"):
    '''Initialise weights and biases'''
    print(f'{mode} initialisation')
    Ws = []
    bs = []
    gammas = []
    betas = []
    sig = 0.0001
    for i in range(dims.shape[0]-1):
        if mode=="He":
            W_i = np.sqrt(2/dims[i]) * np.random.randn(dims[i+1], dims[i])
            b_i = 0 * np.sqrt(2/dims[i]) * np.random.randn(dims[i+1], 1) # 0 for BN
        elif mode=="Xavier":
            W_i = np.sqrt(1/dims[i]) * np.random.randn(dims[i+1], dims[i])
            b_i = np.sqrt(1/dims[i]) * np.random.randn(dims[i+1], 1)
        else:
            W_i = sig * np.random.randn(dims[i+1], dims[i])
            b_i = sig * np.random.randn(dims[i+1], 1)
        
        Ws.append(W_i)
        bs.append(b_i)
        
    for i in range(dims.shape[0]-2): # gamma/beta should not change the shape
        #gamma_i = np.sqrt(2/dims[i]) * np.random.randn(dims[i+1], 1)
        #beta_i = np.sqrt(2/dims[i]) * np.random.randn(dims[i+1], 1)
        gamma_i = sig * np.random.randn(dims[i+1], 1)
        beta_i = sig * np.random.randn(dims[i+1], 1)
        gammas.append(gamma_i)
        betas.append(beta_i)
        
    parameters = {"W": Ws, "b" : bs, "gamma" : gammas, "beta" : betas}
    return parameters

def MiniBatchGD(X, Y, parameters, lam, n_batch, n_epochs):
    augment = True
    if augment:
        variants = DataAugmentation(X)
    N = np.size(Y, 1)
    num_per_epoch = int(N/n_batch)
    eta_min = 1e-5
    eta_max = 1e-1
    eta_t = eta_min
    n_s = 6 * 45000 / n_batch #1000 in optimal
    t = 0
    T = 4*n_s
    sliding_mean = 0
    sliding_variance = 0
    alpha = 0.5
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
                print(f'Decreasing from eta = {eta_t} during epoch {i} / step {t}')
            elif(t % (2*n_s) == 0):
                print(f'Increasing from eta = {eta_t} during epoch {i} / step {t}')
                #eta_max = (T - t)/T * (eta_max - eta_min) + eta_min
            t += 1
            start = j*n_batch
            end = (j+1)*n_batch
            indices = rand[start:end]
            if augment:
                X_batch = GetAugmentedBatch(variants, indices)
            else:
                X_batch = X[:,indices]
            Y_batch = Y[:,indices]
            
            gradients = ComputeGradients(X_batch, Y_batch, parameters, lam)
                
            # Update sliding average
            if t == 1:
                sliding_mean = gradients.get("mean")
                sliding_variance = gradients.get("variance")
            else:
                new_mean = gradients.get("mean")
                new_var = gradients.get("variance")
                sliding_mean = alpha * sliding_mean + (1 - alpha) * new_mean
                sliding_variance = alpha * sliding_variance + (1 - alpha) * new_var
                
            # Update parameters
            for string in ["W", "b", "gamma", "beta"]:
                param = parameters.get(string)
                for num, gradient in enumerate(gradients.get(string)):
                    param[num] -= eta_t * gradient
            #'''
            if(t%100 == 1):
                '''print(f't = {t} out of {T}')
                b_stuff = parameters.get("b")[1]
                b_grad = gradients.get("b")[1]
                print(f'the average entry in b2 has value {np.mean(b_stuff)}')
                print(f'the average entry in b2_grad has value {np.mean(b_grad)}')
                '''
                stats = {"mean" : sliding_mean, "variance" : sliding_variance}
                tcost, tloss = ComputeCost(train_X, train_Y, parameters, lam, stats)
                vcost, vloss = ComputeCost(val_X, val_Y, parameters, lam, stats)
                train_costs.append(tcost)
                train_loss.append(tloss)
                val_costs.append(vcost)
                val_loss.append(vloss)
            #'''
            if(t > T): # 2ns per cycle, 6ns in optimal
                return parameters, res, stats  
        
        
        '''
        tcost, tloss = ComputeCost(train_X, train_Y, W1, W2, b1, b2, lam)
        vcost, vloss = ComputeCost(val_X, val_Y, W1, W2, b1, b2, lam)
        train_costs.append(tcost)
        train_loss.append(tloss)
        val_costs.append(vcost)
        val_loss.append(vloss)
        #'''
    return parameters, res, stats

def DataAugmentation(X):
    '''Creates and stores mirror and translated images for each training data. These will be called on at random during training.'''
    '''Each image has 7*7*2 = 98 variants (7 tx, 7 ty, 2 for flipped states'''
    flip = [X, MirrorImages(X)]
    variants = []
    for i in range(2):
        for tx in range(-1, 2):
            for ty in range(-1, 2):
                variants.append(ShiftImages(flip[i], 3*tx, 3*ty))
    return np.array(variants)

def GetAugmentedBatch(variants, indices):
    '''Returns randomly flipped/translated versions of the batch defined by indices'''
    versions = np.random.randint(0, 18, indices.shape)
    thing = np.transpose(variants[versions,:,indices])
    return thing
    
def MirrorImages(X):
    '''Creates vertically flipped versions'''
    aa_32 = [32*x for x in range(32)]
    bb = [31-x for x in range(32)]
    vv = np.matlib.repmat(aa_32, 32, 1)
    bb_mat = np.transpose(np.matlib.repmat(bb, 1, 32))
    vv = np.transpose(np.matrix((vv.flatten('F'))))
    ind_flip = vv + bb_mat
    inds_flip = np.concatenate((ind_flip, 1024+ind_flip, 2048+ind_flip))
    mirrored_X = numpy.squeeze(X[inds_flip])
    return mirrored_X

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


#%%
'''Perform a fine search for lambda'''
'''
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
'''

#%%
'''Sampling the data augmentation functions'''
'''
train_X, train_Y, train_y, val_X, val_Y, val_y, test_X, test_Y, test_y = GetAllData(False)
variants = DataAugmentation(train_X)
#%%
images = np.array([100, 120, 2, 4, 1, 5, 7, 8, 9, 9])
thing = GetAugmentedBatch(variants, images)
montage_images(thing)
'''
#%%
'''Check gradient'''

'''
train_X, train_Y, train_y, val_X, val_Y, val_y, test_X, test_Y, test_y = GetAllData(True)
train_X = train_X[:30,:3]
train_Y = train_Y[:30,:3]

d = np.size(train_X, 0) #Input
K = np.size(train_Y, 0) #Output
dims = np.array([d, 50, 30, 20, 20, 10, 10, 10, 10, K])
parameters = InitialiseParameters(dims)


a = ComputeGradients(train_X, train_Y, parameters, 0.0)
b = ComputeGradsNumSlow(train_X, train_Y, parameters, 0.0, 1e-5)

print("Sampling 2 layer NN. Hidden layer of size 50.")
for string in ["W", "b", "gamma", "beta"]:
    print("")
    an = a.get(string)
    num = b.get(string)
    for i in range(len(an)):
        enum = np.abs(np.sum(an[i] - num[i]))
        denom = np.max((0.0001, np.abs(np.sum(an[i])) + np.abs(np.sum(num[i]))))
        print(f"Relative error for {string}_{i+1}: {enum/denom}")

#'''

#%%
'''Train a network'''

#'''
accs = []
for lam in [0.005]:
    train_X, train_Y, train_y, val_X, val_Y, val_y, test_X, test_Y, test_y = GetAllData(False)
    #train_X = train_X[:,:10000]
    #train_Y = train_Y[:,:10000]
    
    d = np.size(train_X, 0) #Input
    K = np.size(train_Y, 0) #Output
    dims = np.array([d, 50, 50, K])
    #dims = np.array([d, 50, 30, 20, 20, 10, 10, 10, 10, K])
    parameters = InitialiseParameters(dims)
    
    #lam = 0.005
    #lam = 0.000164 #Optimal with m = 50 according to search
    n_epochs = 100
    n_batch = 100
    
    star_parameters, res, stats = MiniBatchGD(train_X, train_Y, parameters, lam, n_batch, n_epochs)
    
    #%%
    test_acc = 100*ComputeAccuracy(test_X, test_y, star_parameters, stats)
    print(f"Test accuracy: {test_acc}% with λ = {lam}, batch of {n_batch}, {n_epochs} epochs")
    accs.append(test_acc)

tc = res[0]
tl = res[1]
vc = res[2]
vl = res[3]

x = [100*i for i in range(len(tc))]

plt.plot(x, tl, label="Loss (Training)")
plt.plot(x, vl, label="Loss (Validation)")
plt.title(f"Changes in loss over time (with BN)\n (sig = 0.0001)")
plt.xlabel("Step")
plt.xlim(0, 100*len(tc)+1)
plt.legend()
plt.show()
#'''

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
    

