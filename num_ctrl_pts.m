% Surface reconstruction with varying control points (least squares ablation experiment)

clc; clear all; close all;

T = 5; % maximum time bound
L = 1; % space bound

t = 0:0.01:T; % discretization interval for B-spline in time (fixed)
x = 0:0.01:L; % discretization interval for B-spline in space (fixed)

Nt = size(t, 2); % Fixed number of time steps
Nx = size(x, 2); % Fixed number of space steps

n_cp_t_original = 6; % number of control points in time for original surface
n_cp_x_original = 6; % number of control points in space for original surface

d = 3; % order of B-spline (3rd order polynomial)

control_points_experiments = [8, 7, 6, 5, 4, 3]; % Different control points for ablation

% Generate the original surface using a fixed number of control points
dx_original = L / (n_cp_x_original - 1); 
dt_original = T / (n_cp_t_original - 1); 

cx_original = 0:dx_original:L; 
ct_original = 0:dt_original:T; % control point meshgrid

U_original = randn(n_cp_t_original, n_cp_x_original);
U_original(1, :) = ones(1, n_cp_x_original); % Boundary condition x=0
U_original(:, 1) = ones(1, n_cp_t_original); % Initial condition t=0

% Generate the B-spline matrices for the original control points
[tk_t_original, Ln_t_original, Bit_t_original] = BsKnots(n_cp_t_original, d, Nt);
[tk_x_original, Ln_x_original, Bit_x_original] = BsKnots(n_cp_x_original, d, Nx);

% Original surface generation
Sol_original = Bit_t_original * U_original * Bit_x_original';

% Plot the original surface
figure
surf(x, t, Sol_original)
shading interp
hold on
for i = 1:size(cx_original, 2)
    for j = 1:size(ct_original, 2)
        scatter3(cx_original(i), ct_original(j), U_original(j, i), 'r', 'filled')
        str = string(i) + string(j);
        text(cx_original(i), ct_original(j), U_original(j, i) + 0.05, str)
    end
end
xlabel('x'); ylabel('t'); zlabel('U');
title('Original Surface');
hold off;

% Store the error for each experiment
errors = [];

for experiment = 1:length(control_points_experiments)
    n_cp_t = control_points_experiments(experiment); % number of control points in time for reconstruction
    n_cp_x = control_points_experiments(experiment); % number of control points in space for reconstruction

    dx = L / (n_cp_x - 1); 
    dt = T / (n_cp_t - 1); 

    cx = 0:dx:L; 
    ct = 0:dt:T; % control point meshgrid

    % Generate the B-spline matrices for the current number of control points
    [tk_t, Ln_t, Bit_t] = BsKnots(n_cp_t, d, Nt);  % Note Nt remains fixed
    [tk_x, Ln_x, Bit_x] = BsKnots(n_cp_x, d, Nx);  % Note Nx remains fixed

    % Solve for Q using the least squares approach
    intermediate_result = (Bit_t' * Bit_t) \ (Bit_t' * Sol_original);
    Q = intermediate_result * (Bit_x / (Bit_x' * Bit_x));

    % Reconstructed surface
    Sol_reconstruct = Bit_t * Q * Bit_x';  % Same dimensions as Sol_original

    % Plot the reconstructed surface
    figure
    surf(x, t, Sol_reconstruct)
    shading interp
    hold on
    for i = 1:size(cx, 2)
        for j = 1:size(ct, 2)
            scatter3(cx(i), ct(j), Q(j, i), 'r', 'filled')
            str = string(i) + string(j);
            text(cx(i), ct(j), Q(j, i) + 0.05, str)
        end
    end
    xlabel('x'); ylabel('t'); zlabel('U');
    title(['Reconstructed Surface with Control Points: ', num2str(n_cp_x), 'x', num2str(n_cp_t)]);
    hold off;

    % Compute the error between the original and reconstructed surfaces
    error = norm(Sol_original - Sol_reconstruct, 'fro');  % Error based on surfaces, not control points
    errors = [errors, error]; % Store the error for this experiment
end

% Plot the error versus number of control points
figure;
plot(control_points_experiments, errors, '-o');
xlabel('Number of Control Points');
ylabel('Frobenius Norm of Error');
title('Error vs. Number of Control Points');



%% Bspline Functions

function [tk,Ln,Bit]=BsKnots(n_cp,d,Ns)
    %  BSKNOTS B-spline parameter vector (tk) and knots vector (Ln)
    %  OUTPUTS:
    %  tk   - B-spline parameter vector;
    %  Ln   - Knots Vector;
    %  Bit  - B-spline functions, each column corresponds
    %  INPUTS:
    %  n_cp - number of control points
    %    Ns - lenght of the parameter vector tk
    %   
    % author: Raffaele Romagnoli
    % date  : 13/04/2017


    k=1;
    j=1;

    n_knots=n_cp+d+1;

    %% Knots Vector
    Ln=zeros(1,n_knots);
    for i=d+2:n_knots
        if i<n_knots-(d) 
           Ln(1,i)=k;
           k=k+j;
        else
           Ln(1,i)=k;
        end 
    end
    %% Parameter Vector 
    tk=zeros(1,Ns);
    for i=2:Ns
        tk(i)=tk(i-1)+Ln(n_knots)/Ns;
    end
    Bit=zeros(Ns,n_cp);
    i_k=1;
    for j=1:n_cp
        for i=1:Ns
            Bit(i,j)=BsFun(i_k,d,tk(i),Ln);
        end
        i_k=i_k+1;
    end
end

function  B=BsFun(i,d,t,Ln)
    %BSFUN Computes the value of the B-spline function (Cox - De Boor Algorithm)
    %OUTPUT:
    %   B - B-spline function value
    %INPUTS:
    %   i - index of the interval
    %   d - B-spline order
    %   t - parameter
    %  Ln - knots vector
    %  
    % author: Raffaele Romagnoli
    % date  : 13/04/2017

    if d==0
       if t>=Ln(i) && t<Ln(i+1)
          B=1;
       else
          B=0; 
       end
    else
       if  (Ln(d+i)-Ln(i))==0 
           a=0;
       else
           a=((t-Ln(i))/(Ln(d+i)-Ln(i)));
       end
       if (Ln(d+i+1)-Ln(i+1))==0
          b=0;
       else
          b=((Ln(d+i+1)-t)/(Ln(d+i+1)-Ln(i+1)));
       end
       B=a*BsFun(i,d-1,t,Ln)+...
            b*BsFun(i+1,d-1,t,Ln);    
    
    end
end