%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

a1 = [ones(m,1) X];

z2 = a1*Theta1';

a2 = [ones(m,1) sigmoid(z2)];

z3 = a2*Theta2';

a3 = sigmoid(z3);

Theta1(:,1) = 0;

Theta2(:,1) = 0;

% cost function

J = (1/m)*(sum(sum(-y_matrix.*log(a3)))-sum(sum((1-y_matrix).*log(1-a3))))+ ...
    ((lambda/(2*m))* ((sum(sum(Theta1.^2))+sum(sum(Theta2.^2)))));

d3 = a3-y_matrix;

d2 = (d3*Theta2(:,2:size(Theta2,2))).*sigmoidGradient(z2);

Delta1 = (d2'*a1);

Delta2 = (d3'*a2);

% gradiend descent

Theta1_grad = ((1/m)*Delta1)+((lambda/m).*Theta1);
Theta2_grad = ((1/m)*Delta2)+((lambda/m).*Theta2);

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
