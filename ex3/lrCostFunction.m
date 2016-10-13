function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));


newTheta = [0;theta(2:size(theta))];

summation = sum(log(sigmoid(X * theta))' * -y - log(1-sigmoid(X * theta))' * (1-y));

specialTerm = [lambda/(2*m)] * sum(newTheta' * newTheta);

J = summation/m + specialTerm;


grad(1) = [sum((sigmoid(X*theta) - y)' * X(:,1))]/m;

for i=2:size(theta)
  grad(i) = [sum((sigmoid(X*theta) - y)' * X(:,i))]/m + [lambda * newTheta(i)]/m;
end

grad = grad(:);

end
