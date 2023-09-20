import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def LoadText(filename):
    with open(filename, 'rb') as fo:
        byte_contents = fo.read()
        all_characters = list(byte_contents.decode('utf-8'))
        unique_characters = np.unique(all_characters)
    return all_characters, unique_characters

def softmax(x, T=1):
    """ Standard definition of the softmax function (with optional temperature)"""
    return np.exp(x/T) / np.sum(np.exp(x/T), axis=0)

def NucleusSampling(x, theta):
    x_sorted = -np.sort(-x)
    
    kt = 0
    p_prime = 0
    lowest_p = 0
    while(p_prime < theta):
        p_prime += x_sorted[kt]
        lowest_p = x_sorted[kt]
        kt += 1
    
    x_out = np.array([(0 if i<lowest_p else i/p_prime) for i in x])
    
    return x_out

def GetRandomNextX(RNN, p):
    cp = np.cumsum(p)
    a = np.random.uniform()
    indices = np.where(cp-a>0)[0]
    new_x = np.zeros(p.shape)
    new_x[indices[0]] = 1
    return new_x
    
def CreateNewText(RNN, h_0, x_0, n, mode="standard"):
    W = RNN.get("W")
    U = RNN.get("U")
    V = RNN.get("V")
    b = RNN.get("b")
    c = RNN.get("c")
    ind_to_char = RNN.get("ind_to_char")
    
    all_x = []
    
    h_i = h_0
    x_i = x_0
    
    for i in range(n):
        all_x.append(x_i)
        a_next = np.matmul(W, h_i) + np.matmul(U, x_i) + b
        h_next = np.tanh(a_next)
        o_next = np.matmul(V, h_next) + c
        
        if mode == "temp":
            T = RNN.get("temp")                 # T = (0, 1]
            p_next = softmax(o_next, T)
        elif mode == "nucleus":
            theta = RNN.get("theta")            # theta = (0, 1]
            p_next = softmax(o_next)
            p_next = NucleusSampling(p_next, theta)
        else:
            p_next = softmax(o_next)
        
        x_i = GetRandomNextX(RNN, p_next)
        h_i = h_next
    
    text_raw = np.array(all_x)
    text_flat = np.nonzero(text_raw)[1]
    text = "".join([ind_to_char[x] for x in text_flat])
    return text

def ForwardPass(RNN, X, Y, h_0, return_loss=False):
    '''Calculates the Forward pass values needed for gradients'''
    W = RNN.get("W")
    U = RNN.get("U")
    V = RNN.get("V")
    b = RNN.get("b")
    c = RNN.get("c")
    m = RNN.get("m")
    K = RNN.get("K")
    n = X.shape[1]
    
    all_a = np.empty((n, m))
    all_h = np.empty((n+1, m))
    all_o = np.empty((n, K))
    all_p = np.empty((n, K))
    
    all_h[0] = h_0
    h_i = h_0
    for i in range(n):
        x_i = np.array(X[:,i])
        a_next = np.matmul(W, h_i) + np.matmul(U, x_i) + b
        h_next = np.tanh(a_next)
        o_next = np.matmul(V, h_next) + c
        p_next = softmax(o_next)
        
        all_a[i] = a_next 
        all_h[i+1] = h_next
        all_o[i] = o_next
        all_p[i] = p_next
        
        h_i = h_next
    vectors = {"a" : np.transpose(all_a), "h" : np.transpose(all_h), "o" : np.transpose(all_o), "p" : np.transpose(all_p)}
    if return_loss:
        loss_term = [np.dot(Y[:,i], np.log(all_p[i])) for i in range(n)]
        loss_term = - sum(loss_term)
        return vectors, loss_term
    
    return vectors

def ComputeLoss(RNN, X, Y, h_0):
    '''Computes the loss of the network on the given labeled data'''
    vectors = ForwardPass(RNN, X, Y, h_0)
    n = X.shape[1]
    p = np.transpose(vectors.get("p"))
    loss_term = [np.dot(Y[:,i], np.log(p[i])) for i in range(n)]
    loss_term = - sum(loss_term) / n
    return loss_term
    
def ComputeGradients(RNN, X, Y, h_0, get_last_h=False):
    '''Calculates gradients using backprop for SGD'''
    V = RNN.get("V")
    W = RNN.get("W")
    T = X.shape[1]
    vectors, loss = ForwardPass(RNN, X, Y, h_0, True)
    a = vectors.get("a")
    h = vectors.get("h")
    p = vectors.get("p")
    
    
    '''Calculate grad_c and grad_V'''
    G = - np.transpose(Y - p)   # g_t = dL/do_t = G[t]
    
    grad_c = np.sum(G, axis=0) / T
    grad_V = np.sum([np.matmul(np.transpose(G[i:i+1]), np.transpose(h[:,i+1:i+2])) for i in range(T)], axis=0) / T
    
    '''Calculate grad_a and grad_h'''
    grad_h = np.empty(h.shape)
    grad_a = np.empty(a.shape)
    
    grad_h_tau = np.matmul(G[T-1], V) # dL/dh_T = dL/do_T * V
    grad_h[:,T-1] = grad_h_tau
    
    # METHOD 1
    '''
    tanh2_a_tau = np.tanh(np.tanh(a[:,T-1]))
    diag_tanh = np.diag(1-tanh2_a_tau)
    grad_a_tau = np.matmul(grad_h_tau, diag_tanh)
    #'''
    
    # METHOD 2 
    #'''
    t1 = np.square(h[:,T])
    t2 = np.diag(1 - t1)
    grad_a_tau = np.matmul(grad_h_tau, t2)
    #'''
    
    grad_a[:,T-1] = grad_a_tau # dL/da_T = dL_dh_T * diag(1 - tanh^2(a_T))
    
    grad_a_last = grad_a_tau
    for i in range(T-2, -1, -1):
        grad_h_i = np.matmul(G[i], V) +  np.matmul(grad_a_last, W)
        grad_h[:,i] = grad_h_i # dL/dh_t = dL/do_t * V + dL/da_t+1 * W
        
        # METHOD 1
        '''
        tanh2_a_i = np.tanh(np.tanh(a[:,i]))
        diag_tanh = np.diag(1-tanh2_a_i)
        grad_a_last = np.matmul(grad_h_i, diag_tanh)
        #'''
        
        # METHOD 2
        #'''
        t1 = np.square(h[:,i+1])
        t2 = np.diag(1 - t1)
        grad_a_last = np.matmul(grad_h_i, t2)
        #'''
        
        grad_a[:,i] = grad_a_last # dL/da_t = dL/dh_t * diag(1 - tanh^2(a_t))

    '''Calculate grad_W and grad_b'''
    grad_W = np.sum([np.matmul(grad_a[:,i:i+1], np.transpose(h[:,i:i+1])) for i in range(T)], axis=0) / T
    grad_b = np.sum(grad_a, axis=1) / T

    
    '''Calculate grad_U'''
    grad_U = np.sum([np.matmul(grad_a[:,i:i+1], np.transpose(X[:,i:i+1])) for i in range(T)], axis=0) / T
    grads = {"c" : grad_c, "V" : grad_V, "b" : grad_b, "W" : grad_W, "U" : grad_U}
    
    
    '''Clip to prevent exploding gradients'''
    for string in grads.keys():
        grad_to_clip = grads.get(string)
        grads[string] = np.clip(grad_to_clip, -5, 5)
    
    if get_last_h:
        return grads, loss, h[:,T]
    return grads, loss

def ComputeGradientsNum(RNN, X, Y, h_0):
    '''Numerical gradient calculation to ensure correct analytical solution'''
    h = 1e-4
    grad_names = ["c", "V", "b", "W", "U"]
    num_grads = {}
    for grad_name in grad_names:
        param = RNN.get(grad_name)
        dim = param.shape
        grad = np.zeros(dim)
        print(f'Working on {grad_name} of size {dim}')
        
        if(len(dim) == 1):
            for i in range(dim[0]):
                param[i] -= h    # Forward Euler
                l1 = ComputeLoss(RNN, X, Y, h_0)
                param[i] += 2*h  # Backward Euler
                l2 = ComputeLoss(RNN, X, Y, h_0)
                param[i] -= h    # Reset value to default
                grad[i] = (l2 - l1)/(2*h)
            num_grads[grad_name] = grad
            
        elif(len(dim) == 2):
            for i in range(dim[0]):
                for j in range(dim[1]):
                    param[i][j] -= h    # Forward Euler
                    l1 = ComputeLoss(RNN, X, Y, h_0)
                    param[i][j] += 2*h  # Backward Euler
                    l2 = ComputeLoss(RNN, X, Y, h_0)
                    param[i][j] -= h    # Reset value to default
                    grad[i][j] = (l2 - l1)/(2*h)
            num_grads[grad_name] = grad
        
    return num_grads 

def GetDataSegment(RNN, start):
    '''Get training data for an iteration'''
    K = RNN.get("K")
    all_data = RNN.get("all_data")
    length = RNN.get("seq_length")
    char_to_ind = RNN.get("char_to_ind")
    
    X_chars = all_data[start:start+length]
    X_chars = np.array([char_to_ind[x] for x in X_chars])
    Y_chars = all_data[start+1:start+length+1]
    Y_chars = np.array([char_to_ind[x] for x in Y_chars])
    
    Y_hot = np.zeros((Y_chars.size, K))
    Y_hot[np.arange(Y_chars.size),Y_chars] = 1
    Y_hot = np.transpose(Y_hot)
    X_hot = np.zeros((X_chars.size, K))
    X_hot[np.arange(X_chars.size),X_chars] = 1
    X_hot = np.transpose(X_hot)
    
    return X_hot, Y_hot

def AdaGrad(contents, RNN, n_epochs):
    
    book_length = len(contents)
    m = RNN.get("m")
    eta = RNN.get("eta")
    seq_length = RNN.get("seq_length")
    t = 0       # Training step 
    smooth_loss = 0
    
    texts = [] # Snippets to be saved
    losses = [] # Losses for graphing
    grad_sum_of_squares = {} # Sum of squares of gradients hitherto
    
    for i in range(n_epochs):
        h_prev = np.zeros(m)
        e = 0  # Character to start sampling at
        
        while(e <= book_length - seq_length - 1):
            '''Get gradients'''
            X, Y = GetDataSegment(RNN, e)
            gradients, loss, h_last = ComputeGradients(RNN, X, Y, h_prev, True)
            
            
            
            '''Update parameters'''
            for string in gradients.keys():
                param = RNN.get(string)
                grad = gradients.get(string)
                
                '''Initialise sum of squares'''
                if t == 0:
                    grad_sum_of_squares[string] = np.zeros(param.shape)
                
                '''Perform AdaGrad updates'''
                grad_sum = grad_sum_of_squares.get(string)
                param -= eta * grad / np.sqrt(grad_sum + 1e-2)
                
                if(t%500 == 0):
                    print(f'{string} mean update: {np.mean(np.abs(grad)/np.sqrt(grad_sum + 1e-2))}')
                
                grad_sum += np.square(grad)
                
                
            '''Update loss and print results'''
            if smooth_loss == 0:
                smooth_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                
            if(t%500 == 0):
                print(f'\n E{i}, Step {t}, Smooth Loss: {smooth_loss}')
                losses.append(smooth_loss)
                if(np.isnan(loss)):
                    print("NAN WARNING")
                    return
            if(t%1000 == 0):
                text = CreateNewText(RNN, h_last, X[:,0], 200)
                print(f'\n{text}')
                if(t%10000 == 0):
                    print("SAVED THIS SNIPPET")
                    texts.append(text)
                
            '''Update iterables'''
            h_prev = h_last
            e += seq_length
            t += 1
           
    # Fetch one final text when all is said and done
    big_text = CreateNewText(RNN, h_last, X[:,0], 1000)
    return texts, losses, big_text

def AdaGradRandomChunks(contents, RNN, n_epochs, num_of_chunks=100):
    
    book_length = len(contents)
    m = RNN.get("m")
    eta = RNN.get("eta")
    seq_length = RNN.get("seq_length")
    t = 0       # Training step 
    smooth_loss = 0
    
    texts = [] # Snippets to be saved
    losses = [] # Losses for graphing
    grad_sum_of_squares = {} # Sum of squares of gradients hitherto
    
    
    chunk_size = np.floor(book_length / num_of_chunks).astype(int)
    chunk_order = np.arange(num_of_chunks)
    
    for i in range(n_epochs):
        np.random.shuffle(chunk_order)  # Sets a random order of the chunks
        for chunk_index in chunk_order:
            
            e_start = chunk_index * chunk_size
            e_end = (chunk_index + 1) * chunk_size
            e = e_start  # Character to start sampling at
            print(f'Starting chunk {chunk_index} at character {e_start} until {e_end}')
            h_prev = np.zeros(m) # Reset h for new chunk
            
            while(e <= e_end - seq_length - 1):
                '''Get gradients'''
                X, Y = GetDataSegment(RNN, e)
                gradients, loss, h_last = ComputeGradients(RNN, X, Y, h_prev, True)
                
                
                '''Update parameters'''
                for string in gradients.keys():
                    param = RNN.get(string)
                    grad = gradients.get(string)
                    
                    '''Initialise sum of squares'''
                    if t == 0:
                        grad_sum_of_squares[string] = np.zeros(param.shape)
                    
                    '''Perform AdaGrad updates'''
                    grad_sum = grad_sum_of_squares.get(string)
                    param -= eta * grad / np.sqrt(grad_sum + 1e-2)
                    grad_sum += np.square(grad)
                    
                '''Update loss and print results'''
                if smooth_loss == 0:
                    smooth_loss = loss
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                    
                if(t%500 == 0):
                    print(f'\n E{i}, Step {t}, Smooth Loss: {smooth_loss}')
                    losses.append(smooth_loss)
                    if(np.isnan(loss)):
                        print("NAN WARNING")
                        return
                if(t%1000 == 0):
                    text = CreateNewText(RNN, h_last, X[:,0], 200)
                    print(f'\n{text}')
                    if(t%10000 == 0):
                        print("SAVED THIS SNIPPET")
                        texts.append(text)
                    
                '''Update iterables'''
                h_prev = h_last
                e += seq_length
                t += 1
           
    # Fetch one final text when all is said and done
    big_text = CreateNewText(RNN, h_last, X[:,0], 1000)
    return texts, losses, big_text

def Adam(contents, RNN, n_epochs):
    book_length = len(contents)
    m_param = RNN.get("m")
    eta = RNN.get("eta")
    seq_length = RNN.get("seq_length")
    t = 0       # Training step 
    smooth_loss = 0
    
    texts = []  # Snippets to be saved
    losses = [] # Losses for graphing
    
    m = {}          # Decaying average of gradients 
    v = {}          # Decaying average of squared gradients
    beta_1 = 0.9    # For updating m
    beta_2 = 0.999  # For updating v
    epsilon = 1e-8  # Prevent Div/0
    
    for i in range(n_epochs):
        h_prev = np.zeros(m_param)
        e = 0  # Character to start sampling at
        
        while(e <= book_length - seq_length - 1):
            '''Get gradients'''
            X, Y = GetDataSegment(RNN, e)
            gradients, loss, h_last = ComputeGradients(RNN, X, Y, h_prev, True)
            
            
            
            '''Update parameters'''
            for string in gradients.keys():
                param = RNN.get(string)
                grad = gradients.get(string)
                
                '''Initialise sum of squares'''
                if t == 0:
                    m[string] = np.zeros(param.shape)
                    v[string] = np.zeros(param.shape)
                
                    
                '''Perform Adam updates'''
                # Fetch m, v
                m_i = m.get(string)
                v_i = v.get(string)
                
                # Update m, v
                m_i = beta_1 * m_i + (1 - beta_1) * grad
                v_i = beta_2 * v_i + (1 - beta_2) * np.square(grad)
                m[string] = m_i
                v[string] = v_i
                
                # Unbiased versions
                m_hat = m_i / (1 - np.power(beta_1, t+1))
                v_hat = v_i / (1 - np.power(beta_2, t+1))
                
                # Update parameter
                param -= eta * m_hat / (np.sqrt(v_hat) + epsilon)
                
                        
            '''Update loss and print results'''
            if smooth_loss == 0:
                smooth_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                
            if(t%500 == 0):
                print(f'E{i}, Step {t}, Smooth Loss: {np.around(smooth_loss, 3)}')
                losses.append(smooth_loss)
                if(np.isnan(loss)):
                    print("NAN WARNING")
                    return
            if(t%5000 == 0):
                text = CreateNewText(RNN, h_last, X[:,0], 200)
                print(f'\n{text}\n')
                if(t%10000 == 0):
                    print("SAVED THIS SNIPPET")
                    texts.append(text)
                
            '''Update iterables'''
            h_prev = h_last
            e += seq_length
            t += 1
           
    # Fetch one final text when all is said and done
    big_text = CreateNewText(RNN, h_last, X[:,0], 1000)
    return texts, losses, big_text

def AdamRandomChunks(contents, RNN, n_epochs, num_of_chunks=100):
    
    book_length = len(contents)
    m_param = RNN.get("m")
    eta = RNN.get("eta")
    seq_length = RNN.get("seq_length")
    t = 0       # Training step 
    smooth_loss = 0
    
    texts = []  # Snippets to be saved
    losses = [] # Losses for graphing
    
    m = {}          # Decaying average of gradients 
    v = {}          # Decaying average of squared gradients
    beta_1 = 0.9    # For updating m
    beta_2 = 0.999  # For updating v
    epsilon = 1e-8  # Prevent Div/0
    
    
    chunk_size = np.floor(book_length / num_of_chunks).astype(int)
    chunk_order = np.arange(num_of_chunks)
    
    for i in range(n_epochs):
        np.random.shuffle(chunk_order)  # Sets a random order of the chunks
        for chunk_index in chunk_order:
            
            e_start = chunk_index * chunk_size
            e_end = (chunk_index + 1) * chunk_size
            e = e_start  # Character to start sampling at
            #print(f'Starting chunk {chunk_index} at character {e_start} until {e_end}')
            h_prev = np.zeros(m_param) # Reset h for new chunk
            
            while(e <= e_end - seq_length - 1):
                '''Get gradients'''
                X, Y = GetDataSegment(RNN, e)
                gradients, loss, h_last = ComputeGradients(RNN, X, Y, h_prev, True)
                
                
                '''Update parameters'''
                for string in gradients.keys():
                    param = RNN.get(string)
                    grad = gradients.get(string)
                    
                    '''Initialise sum of squares'''
                    if t == 0:
                        m[string] = np.zeros(param.shape)
                        v[string] = np.zeros(param.shape)
                    
                    
                    '''Perform Adam updates'''
                    # Fetch m, v
                    m_i = m.get(string)
                    v_i = v.get(string)
                    
                    # Update m, v
                    m_i = beta_1 * m_i + (1 - beta_1) * grad
                    v_i = beta_2 * v_i + (1 - beta_2) * np.square(grad)
                    m[string] = m_i
                    v[string] = v_i
                    
                    # Unbiased versions
                    m_hat = m_i / (1 - np.power(beta_1, t+1))
                    v_hat = v_i / (1 - np.power(beta_2, t+1))
                    
                    # Update parameter
                    param -= eta * m_hat / (np.sqrt(v_hat) + epsilon)
                    
                '''Update loss and print results'''
                if smooth_loss == 0:
                    smooth_loss = loss
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                    
                if(t%500 == 0):
                    print(f'E{i}, Step {t}, Smooth Loss: {np.around(smooth_loss, 3)}')
                    losses.append(smooth_loss)
                    if(np.isnan(loss)):
                        print("NAN WARNING")
                        return
                if(t%5000 == 0):
                    text = CreateNewText(RNN, h_last, X[:,0], 200)
                    print(f'\n{text}\n')
                    if(t%10000 == 0):
                        print("SAVED THIS SNIPPET")
                        texts.append(text)
                    
                '''Update iterables'''
                h_prev = h_last
                e += seq_length
                t += 1
           
    # Fetch one final text when all is said and done
    big_text = CreateNewText(RNN, h_last, X[:,0], 1000)
    RNN["h_last"] = h_last
    return texts, losses, big_text

def AdamFullRandom(contents, RNN, n_epochs):
    book_length = len(contents)
    m_param = RNN.get("m")
    eta = RNN.get("eta")
    seq_length = RNN.get("seq_length")
    t = 0       # Training step 
    smooth_loss = 0
    
    texts = []  # Snippets to be saved
    losses = [] # Losses for graphing
    
    m = {}          # Decaying average of gradients 
    v = {}          # Decaying average of squared gradients
    beta_1 = 0.9    # For updating m
    beta_2 = 0.999  # For updating v
    epsilon = 1e-8  # Prevent Div/0
    
    for i in range(n_epochs):
        
        # We sample randomly, but we want as many iterations as "normal"
        step_per_epoch = np.floor(book_length/seq_length).astype(int)
        
        for _ in range(step_per_epoch):
            
            # Sample from any point in the book (and reset h since new start)
            h_prev = np.zeros(m_param)
            e = np.random.randint(0, book_length-seq_length)
            
            '''Get gradients'''
            X, Y = GetDataSegment(RNN, e)
            gradients, loss, h_last = ComputeGradients(RNN, X, Y, h_prev, True)
            
            '''Update parameters'''
            for string in gradients.keys():
                param = RNN.get(string)
                grad = gradients.get(string)
                
                '''Initialise sum of squares'''
                if t == 0:
                    m[string] = np.zeros(param.shape)
                    v[string] = np.zeros(param.shape)
                
                
                '''Perform Adam updates'''
                # Fetch m, v
                m_i = m.get(string)
                v_i = v.get(string)
                
                # Update m, v
                m_i = beta_1 * m_i + (1 - beta_1) * grad
                v_i = beta_2 * v_i + (1 - beta_2) * np.square(grad)
                m[string] = m_i
                v[string] = v_i
                
                # Unbiased versions
                m_hat = m_i / (1 - np.power(beta_1, t+1))
                v_hat = v_i / (1 - np.power(beta_2, t+1))
                
                # Update parameter
                param -= eta * m_hat / (np.sqrt(v_hat) + epsilon)
                
            '''Update loss and print results'''
            if smooth_loss == 0:
                smooth_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                
            if(t%500 == 0):
                print(f'E{i}, Step {t}, Smooth Loss: {np.around(smooth_loss, 3)}')
                losses.append(smooth_loss)
                if(np.isnan(loss)):
                    print("NAN WARNING")
                    return
            if(t%5000 == 0):
                text = CreateNewText(RNN, h_last, X[:,0], 200)
                print(f'\n{text}\n')
                if(t%10000 == 0):
                    print("SAVED THIS SNIPPET")
                    texts.append(text)
                
            '''Update iterables'''
            t += 1
           
    # Fetch one final text when all is said and done
    big_text = CreateNewText(RNN, h_last, X[:,0], 1000)
    return texts, losses, big_text

def Validation(contents, RNN):
    book_length = len(contents)
    m = RNN.get("m")
    seq_length = RNN.get("seq_length")
    t = 0       # Training step 
    total_loss = 0
    
    h_prev = np.zeros(m)
    e = 0  # Character to start sampling at
    
    while(e <= book_length - seq_length - 1):
        '''Get gradients'''
        X, Y = GetDataSegment(RNN, e)
        gradients, loss, h_last = ComputeGradients(RNN, X, Y, h_prev, True)
        
        '''Update loss and print results'''
        total_loss += loss
            
        '''Update iterables'''
        h_prev = h_last
        e += seq_length
        t += 1
       
    return total_loss/t # Returns the average loss

def main():
    '''Load full training text from file'''
    contents, unique_chars = LoadText('C:/KTH/Deep Learning/goblet_book.txt')
    
    validating = False
    if(validating):
        validation_contents = contents[:10000]
        contents = contents[10000:]
        
    
    '''Set hyperparameters''' 
    K = len(unique_chars)   # Number of unique chars in text
    RNN = {}
    RNN["all_data"] = contents
    RNN["K"] = K
    
    '''Get character mappings'''
    char_to_ind = {}
    ind_to_char = {}
    for i, char in enumerate(unique_chars):
        char_to_ind[char] = i
        ind_to_char[i] = char
    RNN["char_to_ind"] = char_to_ind
    RNN["ind_to_char"] = ind_to_char
    
    '''Set hyperparameters'''
    m, eta, seq = 100, 0.001, 25
    n_epochs = 3           # Number of epochs (full revisions of text)
    RNN["m"] = m            # Dimensionality of hidden state
    RNN["eta"] = eta        # Learning rate
    RNN["seq_length"] = seq # Length of input sequences used in training
    
    '''Initialise parameters'''
    sig = 0.01                                  # Standard deviation
    RNN["b"] = np.zeros(m)                      # Bias vector
    RNN["c"] = np.zeros(K)                      # Bias vector
    RNN["U"] = np.random.randn(m, K) * sig      # Weight
    RNN["W"] = np.random.randn(m, m) * sig      # Weight
    RNN["V"] = np.random.randn(K, m) * sig      # Weight
    
    
    '''Do training and validate model'''
    #text, losses, bigtext = AdaGrad(contents, RNN, n_epochs)
    text, losses, bigtext = Adam(contents, RNN, n_epochs)
    #text, losses, bigtext = AdamFullRandom(contents, RNN, n_epochs)
    #text, losses, bigtext = AdamRandomChunks(contents, RNN, n_epochs, 1000)
    
    if(validating):
        loss = Validation(validation_contents, RNN)
        print(f"The method gave an average loss of {loss} on the validation set")
    
    return RNN, text, losses, bigtext, contents
    
    
    

RNN, snippet_list, losses, passage, contents = main()
#%%
#'''
print(f'\n#######################\n{passage}\n#######################')

for i, snip in enumerate(snippet_list):
    print(f'Snippet {i} from after {10000*i} steps:\n{snip}\n')
#'''

#%%
x = [500*i for i in range(len(losses))]

plt.plot(x, losses)
plt.title("The evolution of loss over 3 full epochs\n (Adam with 1000 random chunks)")
#plt.ylim(37, 50)
plt.xlabel("Iteration")
plt.ylabel("Smooth loss")
plt.show()

#%%
'''Sample some new text with different methods'''
X, Y = GetDataSegment(RNN, 214)
x_0 = X[:,0]
#h_0 = RNN.get("h_last")
h_0 = np.zeros(RNN.get("m"))

#%%
RNN["temp"] = 0.7
text = CreateNewText(RNN, h_0, x_0, 800, "temp")
print(text)

#%%
RNN["theta"] = 0.8
text = CreateNewText(RNN, h_0, x_0, 800, "nucleus")
print(text)







'''
params, gradients, numerical_gradients = main()
for string in gradients.keys():
    print("")
    an = gradients.get(string)
    num = numerical_gradients.get(string)
    print(f'Max absolute error for {string}: {np.max(an - num)}')
    enum = np.abs(np.sum(np.abs(an - num)))
    denom = np.max((0.0001, np.abs(np.sum(np.abs(an))) + np.abs(np.sum(np.abs(num)))))
    print(f"Relative error for {string}: {enum/denom}")

b_diff = gradients.get("b") - numerical_gradients.get("b")
b_ratio = np.divide(gradients.get("b"), numerical_gradients.get("b"))

W_diff = gradients.get("W") - numerical_gradients.get("W")
W_ratio = np.divide(gradients.get("W"), numerical_gradients.get("W"))

U_diff = gradients.get("U") - numerical_gradients.get("U")
U_ratio = np.divide(gradients.get("U"), numerical_gradients.get("U"))

print("\n")
#'''



