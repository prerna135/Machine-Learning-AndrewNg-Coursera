function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=sigmoid(theta'*X');

J=((log(h)*y) + (log(ones(1,m)-h)*(ones(m,1)-y)))/(-m);
temp=theta.^2;
J=J+((lambda/(2*m))*(sum(temp)-temp(1)));

grad=((h-y')*X)./m + theta'.*(lambda/m);
grad(1)=((h-y')*X(:, 1))/m;



% =============================================================

end
