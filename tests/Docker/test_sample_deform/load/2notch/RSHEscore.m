function [p] = RSHEscore(epsilon)
%Calculate a score using RSHE to describe the anisotropy under strained
%matter
%   l,m: degree and order of RSHE
%   epsilon: strain tensor

e_xx = epsilon(1);
e_yy = epsilon(2);
e_xy = epsilon(3);

p = zeros(1,5);
n_theta = 100;
d_theta = 2*pi/n_theta;
for theta = (1:n_theta)/n_theta*2*pi
    x0 = cos(theta);
    y0 = sin(theta);
    
    x = x0*(1+e_xx) + y0*e_xy;
    y = y0*(1+e_yy) + x0*e_xy;
    
    rho = sqrt(x^2 + y^2);
    
    Y_00 = sqrt(1/pi)/2;
    Y_22n = sqrt(15/pi)/2*x*y/rho^2;
    Y_22p = sqrt(15/pi)/4*(x^2 - y^2)/rho^2;
    Y_44n = sqrt(35/pi)*3/4*x*y*(x^2 - y^2)/rho^4;
    Y_44p = sqrt(35/pi)*3/16*(x^2*(x^2-3*y^2)-y^2*(3*x^2-y^2))/rho^4;
    
    
    p = p + [Y_00*rho^2/2*d_theta,...
            Y_22n*rho^2/2*d_theta,...
            Y_22p*rho^2/2*d_theta,...
            Y_44n*rho^2/2*d_theta,...
            Y_44p*rho^2/2*d_theta];
end
p = p/pi/(sqrt(1/pi)/2);

% syms theta
% 
% x0 = cos(theta);
% y0 = sin(theta);
% 
% x = x0*(1+e_xx) + y0*e_xy;
% y = y0*(1+e_yy) + x0*e_xy;
% 
% rho = sqrt(x^2 + y^2);
% 
% Y_00 = sqrt(1/pi)/2;
% Y_22n = sqrt(15/pi)/2*x*y/rho^2;
% Y_22p = sqrt(15/pi)/4*(x^2 - y^2)/rho^2;
% Y_44n = sqrt(35/pi)*3/4*x*y*(x^2 - y^2)/rho^4;
% Y_44p = sqrt(35/pi)*3/16*(x^2*(x^2-3*y^2)-y^2*(3*x^2-y^2))/rho^4;
% 
% F_00 = int(Y_00*rho^2/2,theta,0,2*pi);
% F_22n = int(Y_22n*rho^2/2,theta,0,2*pi);
% F_22p = int(Y_22p*rho^2/2,theta,0,2*pi);
% F_44n = int(Y_44n*rho^2/2,theta,0,2*pi);
% F_44p = int(Y_44p*rho^2/2,theta,0,2*pi);
% 
% p_00 = eval(F_00);
% p_22n = eval(F_22n);
% p_22p = eval(F_22p);
% p_44n = eval(F_44n);
% p_44p = eval(F_44p);
% 
% p = [p_00, p_22n, p_22p, p_44n, p_44p];
end

