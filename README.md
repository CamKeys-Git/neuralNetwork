# numberClassifier

<h1>Neural Network Using MatLab<h1> 


This is a number classifier that uses a "vectorized implementation" of the regularized cost function and gradient to classify (recognize) handwritten numbers.

To get stared, load the dataset containing 5000 images of handwritten digits.

```Matlab
load('ex4data1.mat');
```

Randomly initialize weights for 20x20 Pixel Images of Digits using randInitializeWeights. We will use one input layer and one hidden layer.

```Matlab
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (we have mapped "0" to label 10)

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
```

Unroll 20x20 parameter matrices into one vector, 400 inputs long.
```
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
```

Using the regularized cost function shown below nnCostFunction will return the initial cost using our random weights, initial_nn_params. The cost will be high before training.



```
% Weight regularization parameter (set this to 0 here).
lambda = 0;

J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

fprintf('Cost at parameters: %f', J);
```

The code below will run the function checkNNGradients.m which will create a small neural network and dataset that will be used for checking your gradients. If the nnCostFunction (backpropagation) implementation is correct, you should see a relative dierence that is less than 1e-9.

```Matlab
checkNNGradients;
```

Use fmincg to learn a good set of parameters. After the training completes, the code will report the training accuracy by computing the percentage of examples it got correct. 

```Matlab
options = optimset('MaxIter', 50);
lambda = 1;

% Creates "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction takes in only one argument 
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
```

Now see the results after training. You should see a training accuracy of around 95% (this may vary due to random initialization).

```Matlab
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
 
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
```
