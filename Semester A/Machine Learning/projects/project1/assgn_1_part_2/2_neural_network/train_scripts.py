from NeuralNetwork import *

def get_error(patterns_input, patterns_output, nn):
    error = 0.0
    n_samples = patterns_input.shape[0]
    # for each sample of the dataset, do a forward pass and get the error
    for p in range(n_samples):
        predictions = nn.forward_pass(patterns_input[p])
        targets = patterns_output[p]
        error += np.sum( np.power(predictions-targets, 2) )
    error = error/(2*n_samples)
    return error

def train(patterns_input,
          patterns_output,
          n_hidden_neurons,
          iterations,
          learning_rate,
          test_set_input=None,
          test_set_output=None,
          is_iris=None):
    # get the number of samples
    n_samples = patterns_input.shape[0]
    # get the number of elements for each sample from the data
    n_inputs = patterns_input.shape[1]
    
    # get a sample from the labels
    sample_y = patterns_output[0]
    if np.isscalar(sample_y):
        n_targets = 1
    else:
        # get the number of its elements
        n_targets = len(sample_y)
    
    # initialize a neural network
    nn = NeuralNetwork(n_inputs, n_hidden_neurons, n_targets)
    
    errors = np.array([], dtype=np.float32) # empty array to store the error
    training_errors = np.array([], dtype=np.float32)
    test_errors = np.array([], dtype=np.float32)
    
    for i in range(iterations):
        error = 0.0
        for p in range(n_samples):
            # get inputs
            inputs = patterns_input[p]
            # get groundtruth targets
            targets = patterns_output[p]
            # do forward pass to get the predictions
            predictions = nn.forward_pass(inputs)
            # do backward pass. get the error between the predictions and the targets
            error += nn.backward_pass(inputs, targets, learning_rate);
        error /= n_targets
        
        if (i+1)%10==0:
            print('Iteration {:05} | Cost = {:.5f}'.format(i+1, error))
        
        if is_iris is not None:
            if is_iris==True:
                training_errors = np.append(training_errors, get_error(patterns_input, patterns_output, nn))
                test_errors = np.append(test_errors, get_error(test_set_input, test_set_output, nn))
            if i%100==0:
                test_iris(patterns_input, patterns_output, nn)
        errors = np.append(errors, error)
    return errors, nn, training_errors, test_errors 

def test_xor(patterns_input, patterns_output, nn):
    # get the number of samples
    n_samples = patterns_input.shape[0]
    # iterate over each sample
    for p in range(n_samples):
        inputs = patterns_input[p]
        target = patterns_output[p]
        prediction = nn.forward_pass(inputs)
        if not np.isscalar(prediction):
            prediction = prediction[0]
        print('Sample #{:02} | Target value: {:.2f} | Predicted value: {:.5f}'.format(p+1, target, prediction))

def test_iris(inputs, targets, nn):
    print('--------------------------------------')
    print('Testing on Iris dataset...')
    for p in range(0, 5):
        x = inputs[p]
        target = targets[p,:]
        prediction = nn.forward_pass(x)
        print('Sample #{:02} | Target value: {:.2f} | Predicted value: {:.5f}'.format(p+1, np.argmax(target), np.argmax(prediction)))
    for p in range(25, 30):
        x = inputs[p]
        target = targets[p,:]
        prediction = nn.forward_pass(x)
        print('Sample #{:02} | Target value: {:.2f} | Predicted value: {:.5f}'.format(p+1, np.argmax(target), np.argmax(prediction)))
    for p in range(50, 55):
        x = inputs[p]
        target = targets[p,:]
        prediction = nn.forward_pass(x)
        print('Sample #{:02} | Target value: {:.2f} | Predicted value: {:.5f}'.format(p+1, np.argmax(target), np.argmax(prediction)))
    print('--------------------------------------')
