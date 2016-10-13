data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y);
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
num_iters = 1500;
alpha = 0.01;
J_history = zeros(num_iters, 1);


function [theta, J_history] = gradientDescent(X, y, theta, iterations, alpha, m)

  for iter = 1:iterations

    tempTheta = theta;

      for j = 1:size(theta)

          %evaluate the derivative portion of the gradient descent equation
          summation = sum(((X * tempTheta) - y) .* X(:,j));

          %evaluate the new value of theta(j)
          theta(j) = theta(j) - (alpha * summation / m);
      end

      J_history(iter) = computeCost(X ,y, theta);


  end

  disp(J_history)
  disp(theta)

end

function J = computeCost(X, y, theta)
  m = length(y); % number of training examples

  J = 0;

  %evaluate the summation in the cost function
  summation = sum(((X * theta) - y) .^ 2);

  %compute the cost of the particular values of theta
  J = summation / (2 * m);

end


% gradientDescent(X, y, theta, num_iters, alpha, m)

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

X = [ones(m, 1) X];



theta = pinv((X' * X)) * (X' * y);
