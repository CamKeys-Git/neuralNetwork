# neuralNetwork
Neural network using MatLab for number clasification 

Load the data set to get stared.

```Matlab
load('ex4data1.mat');
```

Randomly initialize weights using randInitializeWeights.

```Matlab
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
```

The code below will run the function checkNNGradients.m which will create a small neural network and dataset that will be used for checking your gradients. If the backpropagation implementation is correct, you should see a relative dierence that is less than 1e-9.

```Matlab
checkNNGradients;
```


Run the code below to use fmincg to learn a good set of parameters. After the training completes, the code will report the training accuracy by computing the percentage of examples it got correct. You should see a training accuracy of around 95% (this may vary due to the random initialization).

```Matlab
options = optimset('MaxIter', 50);
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
```


