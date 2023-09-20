'''Step 1 in learning to design neural networks from scratch. Introduces many essential functions for neural networks like '''

import numpy as np
import numpy.matlib 
import matplotlib.pyplot as plt

''' Loads datasets into numpy arrays'''
def LoadBatch(filename):
    import pickle
    with open('Datasets/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.array(dict[b'data'])
    y = np.array(dict[b'labels'])
    Y = np.zeros((y.size, 10))
    Y[np.arange(y.size),y] = 1
    
    
    X = np.transpose(np.array(X).astype(float))
    Y = np.transpose(np.array(Y).astype(float))
    y = np.transpose(y)
        
    return X, Y, y

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

'''Standard definition of the sigmoid function'''
def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1) 

'''Returns the probabilities p of each label'''
def EvaluateClassifier(X, W, b):
    return softmax(np.matmul(W,X) + b)

'''Calculates the cost function for a batch of predictions'''
def ComputeCost(X, Y, W, b, lam):
    P = EvaluateClassifier(X, W, b)
    N = np.size(P,1)
    loss_term = [np.dot(Y[:,i], np.log(P[:,i])) for i in range(N)]
    loss_term = - sum(loss_term) / N
    reg_term = np.sum(np.square(W)) * lam
    return loss_term + reg_term, loss_term    

'''Computes the accuracy of a set of predictions (between 0 and 1)'''
def ComputeAccuracy(X, y, W, b):
    p = EvaluateClassifier(X, W, b)
    guesses = np.argmax(p, 0)
    corrects = np.sum((y==guesses).astype(float))
    return corrects / np.size(y, 0)

'''Analytically computes gradients of the parameters W and b using backprop'''
def ComputeGradients(X, Y, W, b, lam):
    nb = np.size(X, 1)
    P_batch = softmax(np.matmul(W, X) + np.matmul(b, np.ones((1, nb))))
    G_batch = - (Y - P_batch)
    
    dL_dW = np.matmul(G_batch, np.transpose(X)) / nb
    dJ_dW = dL_dW + 2*lam*W
    
    dJ_db = np.matmul(G_batch, np.ones((nb, 1))) / nb
    
    return dJ_dW, dJ_db

''' Numerically computers gradient to verify the analytical ComputeGradients() implementation. Converted from Matlab code.'''
def ComputeGradsNum(X, Y, W, b, lamda, h):
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	c = ComputeCost(X, Y, W, b, lamda)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return grad_W, grad_b

''' Slower but more accurate numerical computation. Converted from Matlab code.'''
def ComputeGradsNumSlow(X, Y, W, b, lamda, h):
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeBonusCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeBonusCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeBonusCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeBonusCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return grad_W, grad_b

'''Initialises W and b before learning (Either random or Xavier)'''
def InitialiseParameters(use_xavier = False):
    if(use_xavier):
        W = np.random.randn(10, 3072) / np.sqrt(3072)    
    else:
        W = 0.01 * np.random.randn(10, 3072)

    b = 0.01 * np.random.randn(10, 1)
    return W, b

'''Performs a learning episode using mini batches'''
def MiniBatchGD(X, Y, W, b, lam, n_batch, eta, n_epochs, decay=False, bonus_loss=False):
    #if(n_batch * n_epochs > np.size(X, 1)):
    #    print("Not enough data!")
    train_costs = []
    train_loss = []
    val_costs = []
    val_loss = []
    res = [train_costs, train_loss, val_costs, val_loss]
    N = np.size(Y, 1)
    rand = np.random.permutation(N)
    num_per_epoch = int(N/n_batch)
    for i in range(n_epochs):
        print(f'Epoch {i}')
        if(decay):
            comp = np.floor(n_epochs/4).astype(int)
            if(i%comp == (comp - 1)):
                eta = eta / 10
                print(f"eta updated to {eta}")
                
        for j in range(num_per_epoch):
            start = j*n_batch
            end = (j+1)*n_batch - 1
            indices = rand[start:end]
            X_batch = X[:,indices]
            Y_batch = Y[:,indices]
            if(bonus_loss):
                dJ_dW, dJ_db = ComputeBonusGradients(X_batch, Y_batch, W, b, lam) 
            else:
                dJ_dW, dJ_db = ComputeGradients(X_batch, Y_batch, W, b, lam)
            W -= eta * dJ_dW
            b -= eta * dJ_db
        if(bonus_loss):
            tcost, tloss = ComputeBonusCost(train_X, train_Y, W, b, lam)
            vcost, vloss = ComputeBonusCost(val_X, val_Y, W, b, lam)
        else:
            tcost, tloss = ComputeCost(train_X, train_Y, W, b, lam)
            vcost, vloss = ComputeCost(val_X, val_Y, W, b, lam)
        train_costs.append(tcost)
        train_loss.append(tloss)
        val_costs.append(vcost)
        val_loss.append(vloss)
    return W, b, res

'''Performs a grid search for λ, η, and batch size and returns a matrix of test accuracy across the attempted values.'''
def GridSearch(X, Y, test_X, test_Y, W, b, n_epochs):
    lam_vec = np.array([0, 0.1, 0.3, 0.6, 1])
    eta_vec = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01])
    batch_size_vec = np.array([10, 30, 100, 200, 500])
    accs = np.zeros((np.size(lam_vec), np.size(eta_vec), np.size(batch_size_vec)))
    for i, n_batch in enumerate(batch_size_vec):
        for j, eta in enumerate(eta_vec):
            for k, lam in enumerate(lam_vec):
                W, b = InitialiseParameters()
                W_star, b_star, res = MiniBatchGD(X, Y, W, b, lam, n_batch, eta, n_epochs)
                test_acc = 100*ComputeAccuracy(test_X, test_y, W_star, b_star)
                print(f"Test acc: {test_acc}% with λ = {lam}, η = {eta}, batch size {n_batch}")
                accs[i][j][k] = test_acc
    return accs

'''Alternative cost function (sigmoid instead of softmax)'''
def ComputeBonusCost(X, Y, W, b, lam):
    P = sigmoid(np.matmul(W,X) + b)
    N = np.size(P,1)
    loss_term = [np.dot(Y[:,i], np.log(P[:,i])) + np.dot((1 - Y[:,i]), np.log(1 - P[:,i])) for i in range(N)]
    loss_term = - sum(loss_term) / (10*N)
    reg_term = np.sum(np.square(W)) * lam
    return loss_term + reg_term, loss_term    
    
'''Alternative gradient calculation (sigmoid instead of softmax)'''
def ComputeBonusGradients(X, Y, W, b, lam):
    nb = np.size(X, 1)
    P_batch = sigmoid(np.matmul(W, X) + np.matmul(b, np.ones((1, nb))))
    G_batch = - (Y - P_batch) / 10 # K = 10 = number of labels
    
    dL_dW = np.matmul(G_batch, np.transpose(X)) / nb
    dJ_dW = dL_dW + 2*lam*W
    
    dJ_db = np.matmul(G_batch, np.ones((nb, 1))) / nb
    
    return dJ_dW, dJ_db

'''Alternative accuracy (sigmoid instead of softmax)'''
def ComputeBonusAccuracy(X, y, W, b):
    p = sigmoid(np.matmul(W,X) + b)
    guesses = np.argmax(p, 0)
    corrects = np.sum((y==guesses).astype(float))
    return corrects / np.size(y, 0)
 
# %% Cell 1: Fetch data and preprocess
'''Fetch the necessary data and labels (y = labels, Y = one hot labels)'''
big_batch = False

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


'''Normalise all data based on training data'''

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

# %% Cell2: Initialise weights and biases, evaluate basic (should be ~10% by random chance, as dataset has 10 labels)
'''Initialise W and b as N(0, 0.01)''' 
W, b = InitialiseParameters()

'''Evaluate W and b'''
batch_X = train_X[:, 50:100]
batch_Y = train_Y[:, 50:100]
batch_y = train_y[50:100]
lam = 0.5

cost = ComputeCost(batch_X, batch_Y, W, b, lam)
print(f"Initial accuracy: {100*ComputeAccuracy(batch_X, batch_y, W, b)}%")


# %% Cell 3: Compare analytical gradient to numerical implmenetation to confirm correctness
'''Compare analytical gradient to numerical'''

grad_W, grad_b = ComputeBonusGradients(batch_X, batch_Y, W, b, lam)
num_grad_W, num_grad_b = ComputeGradsNumSlow(batch_X, batch_Y, W, b, lam, 1e-6)

W_diff = grad_W - num_grad_W 
b_diff = grad_b - num_grad_b
print(f"Largest absolute error in W: {np.max(W_diff)}")
print(f"Largest absolute error in b: {np.max(b_diff)}")
enum = np.abs(np.sum(grad_W - num_grad_W))
denom = np.max((0.0001, np.abs(np.sum(grad_W)) + np.abs(np.sum(num_grad_W))))
print(f"Relative error in W gradient: {enum/denom}") # This should always return a small value 
enum2 = np.abs(np.sum(grad_b - num_grad_b)) 
denom2 = np.max((0.0001, np.abs(np.sum(grad_b)) + np.abs(np.sum(num_grad_b))))
print(f"Relative error in b gradient: {enum2/denom2}")


# %% Cell 4: Testing new loss function

P = sigmoid(np.matmul(W,train_X) + b)
N = np.size(P,1)
term1 = (train_Y[:,1])
term2 = np.log(P[:,1])
term3 = (1 - train_Y[:,1])
term4 = np.log(1-P[:,1])
term5dot1 = np.dot(term1, term2)
term6dot2 = np.dot(term3, term4)
term7final = term5dot1 + term6dot2
first_loss = - term7final / 10
loss_term_pre = [np.dot(train_Y[:,i], np.log(P[:,i])) + np.dot((1 - train_Y[:,i]), (np.log(1 - P[:,i]))) for i in range(N)]
loss_term = - sum(loss_term_pre) / (10*N)
reg_term = np.sum(np.square(W)) * lam
print(loss_term)

#print(ComputeBonusCost(train_X, train_Y, W, b, lam))

# %% Cell 5: Perform a Mini Batch Gradient Descent
bonus_loss = True
grid_search = False
decay = False       
bonus_string = ""
if (bonus_loss):
    bonus_string = "(With bonus loss)"
decay_string = ""
if (decay):
    decay_string = "(With step decay)"

lam = 0.1
n_epochs = 40
n_batch = 100
eta = 0.01

if grid_search:
    test_accuracies = GridSearch(train_X, train_Y, test_X, test_Y, W, b, n_epochs)

else:
    W_star, b_star, res = MiniBatchGD(train_X, train_Y, W, b, lam, n_batch, eta, n_epochs, decay, bonus_loss)
    if(bonus_loss):
        test_acc = 100*ComputeBonusAccuracy(test_X, test_y, W_star, b_star)
    else:
        test_acc = 100*ComputeAccuracy(test_X, test_y, W_star, b_star)
    #Histogrammer(test_X, test_y, W_star, b_star)
    print(f"Test accuracy: {test_acc}% with λ = {lam}, η = {eta}, batch of {n_batch}, {n_epochs} epochs")
    
    tc = res[0]
    tl = res[1]
    vc = res[2]
    vl = res[3]
    x = [i+1 for i in range(n_epochs)]
    
    plt.plot(x, tl, label="Loss (Training)")
    plt.plot(x, tc, label="Cost (Training)")
    plt.plot(x, vl, label="Loss (Validation)")
    plt.plot(x, vc, label="Cost (Validation)")
    plt.title(f"Changes in cost/loss over time {decay_string} {bonus_string} \n (λ={lam}, η = {eta})")
    plt.xlabel("Epoch")
    plt.xlim(0, n_epochs+1)
    plt.legend()
    plt.show()
    
    montage(W_star)

#%% Cell 6: Making histograms of correct and incorrect guess probabilities


if(bonus_loss):
    p = sigmoid(np.matmul(W_star,test_X) + b_star)
else:
    p = softmax(np.matmul(W_star, test_X) + b_star)
guesses = np.argmax(p, 0)
guesses_p = np.max(p, 0)
rights = np.array((test_y==guesses).astype(int))
rights_p = np.multiply(guesses_p, rights)
rights_p = rights_p[rights_p != 0]


wrongs = np.array((test_y!=guesses).astype(int))
wrongs_p = np.multiply(guesses_p, wrongs)
wrongs_p = wrongs_p[wrongs_p != 0]

plt.hist(rights_p, rwidth=0.7)
plt.title("p of guessed label when correct " + bonus_string)
plt.xlim(0,1)
plt.show()
plt.hist(wrongs_p, rwidth=0.7)
plt.title("p of guessed label when incorrect " + bonus_string)
plt.xlim(0,1)
plt.show()
