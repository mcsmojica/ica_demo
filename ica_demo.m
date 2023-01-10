clear; close all; clc;

%% load ideal (target) images
S1 = imread('girl.png');
S2 = imread('klimt.png');

% just crop S2 to make the dimensions consistent
[m,n,p]= size(S1);
S2 = S2(1:m,1:n,:);
n = n*p;

figure(97);
subplot(3,2,1); imshow(S1); title('Ideal Image 1');
subplot(3,2,2); imshow(S2); title('Ideal Image 2');

%% create a mixing matrix and mix the images
% ith row gives the mixing coeffs necessary to get ith measured image Xi
A = [0.1 0.4;0.3 0.3];

% mixed images
% in the real world setting, these would be our measured signals
X1 = double(A(1,1)*S1 + A(1,2)*S2);
X2 = double(A(2,1)*S1 + A(2,2)*S2);

subplot(3,2,3); imshow(uint8(X1)); title('Mixed Image 1');
subplot(3,2,4); imshow(uint8(X2)); title('Mixed Image 2');

% rearrange the images as long vectors and mean-center at 0
x1 = reshape(X1,m*n,1); x1 = x1 - mean(x1);
x2 = reshape(X2,m*n,1); x2 = x2 - mean(x2);

% compute the angle of rotation
theta0 = 0.5*atan( -2*sum(x1.*x2) / sum(x1.^2-x2.^2) );

%% rotate out PC direction
% need transpose of rotation matrix
% U_star = [cos(theta0)  sin(theta0); ...
%           -sin(theta0) cos(theta0)].
U_star = [cos(theta0)  sin(theta0); -sin(theta0) cos(theta0)];

%% undo scaling of singular values
% need to calculate the inverse of the sigma matrix = 1/sqrt(variance) 
% along each of the directions of the principal component axes
sig1 = sum( (x1*cos(theta0)+x2*sin(theta0)).^2 );; % variance along 1 of the PCs
sig2 = sum( (x1*cos(theta0-pi/2)+x2*sin(theta0-pi/2)).^2 ); 
Sigma = [1/sqrt(sig1) 0; 0 1/sqrt(sig2)];

%% make probability density separable
% applying the two matrix transformations changes the coordinate system
% the transformed data are the ones we need to take statistics on
% let's call the transformed signals x1bar and x2bar
X1bar = Sigma(1,1)*(U_star(1,1)*X1 + U_star(1,2)*X2);
X2bar = Sigma(2,2)*(U_star(2,1)*X1 + U_star(2,2)*X2);

x1bar = reshape(X1bar,m*n,1);
x2bar = reshape(X2bar,m*n,1);

%% final step: undo effect of V*
% phi0 is the angle that minimizes the kurtosis in the variance
phi0 =0.25*atan( -sum(2*(x1bar.^3).*x2bar-2*x1bar.*(x2bar.^3)) ... 
    / sum(3*(x1bar.^2).*(x2bar.^2)-0.5*(x1bar.^4)-0.5*(x2bar.^4)) );

V = [cos(phi0)  sin(phi0); -sin(phi0) cos(phi0)];

% Approximate solutions -- what ICA suggests the underlying components are
S1_approx = V(1,1)*x1bar + V(1,2)*x2bar;
S2_approx = V(2,1)*x1bar + V(2,2)*x2bar;

% rescale
min1 = min(S1_approx(:)); min2 = min(S2_approx(:));
maX1 = max(S1_approx(:)); maX2 = max(S2_approx(:));
S1_approx = S1_approx+min1; S1_approx = S1_approx*(255/maX1);
S2_approx = S2_approx+min2; S2_approx = S2_approx*(255/maX2);

% some housekeeping: rescaling and mean centering
S1_approx = reshape(S1_approx,size(S1));
S2_approx = reshape(S2_approx,size(S2));
min1 = min(S1_approx(:)); min2 = min(S2_approx(:));
max1 = max(S1_approx(:)); max2 = max(S2_approx(:));
S1_approx=S1_approx - min1; S1_approx=S1_approx*(255/max1);
S2_approx=S2_approx-min2; S2_approx=S2_approx*(255/max2);

subplot(3,2,5); imshow(uint8(S1_approx)); title('Predicted Component 1');
subplot(3,2,6); imshow(uint8(S2_approx)); title('Predicted Component 2');
