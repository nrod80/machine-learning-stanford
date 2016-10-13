function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
data = load('ex1data1.txt');

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %iterate over each element in the theta vector

    tempTheta = theta;

    for j = 1:size(theta)



        %evaluate the derivative portion of the gradient descent equation
        summation = sum(((X * tempTheta) - y) .* X(:,j));

        %evaluate the new value of theta(j)
        theta(j) = theta(j) - (alpha * summation / m);
    end

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
