% =========================================================================
% Probabilistic Robotics - Localization (Single File Version)
% =========================================================================
clear; clc; close all;

%% 1. parameter and environment setup
filter_type = 'none'; % choose one among 'none', 'ekf', and 'pf'
do_plot = true;       % set true if you need plots
% seed = 42;            % you can fix for debugging
seed = 'shuffle';
num_steps = 200;      % simulation step number
data_factor = 1.0;
filter_factor = 1.0;
num_particles = 100;

if ~isempty(seed)
    rng(seed);
end

fprintf('Data factor: %f\n', data_factor);
fprintf('Filter factor: %f\n', filter_factor);

% Noise parameter
alphas = [0.05^2, 0.005^2, 0.1^2, 0.01^2];
beta = deg2rad(5)^2;

% Field structure initialization
env.NUM_MARKERS = 6;
env.MARKER_POS = [ ...
    21, 21 + 0.5 * 442, 21 + 442, 21 + 442, 21 + 0.5 * 442, 21; ... % X positions
    0,  0,              0,        292,       292,            292  ... % Y positions
];
env.alphas = data_factor * alphas;
env.beta = data_factor * beta;

% Initial condition
initial_mean = [180; 50; 0];
initial_cov = diag([10, 10, 1]);

% filter structure initialization
filt = [];
if strcmp(filter_type, 'ekf')
    filt.type = 'ekf';
    filt.alphas = filter_factor * alphas;
    filt.beta = filter_factor * beta;
    filt.mu = initial_mean;
    filt.sigma = initial_cov;
elseif strcmp(filter_type, 'pf')
    filt.type = 'pf';
    filt.alphas = filter_factor * alphas;
    filt.beta = filter_factor * beta;
    filt.num_particles = num_particles;
    filt.particles = mvnrnd(initial_mean', initial_cov, num_particles);
    filt.weights = ones(num_particles, 1) / num_particles;
    filt.mu = initial_mean;
    filt.sigma = initial_cov;
end

% create policy function handle
dt = 0.1;
policy = @(x, t) open_loop_rectangle_policy(t, dt);

% perform localization
localize(env, policy, filt, initial_mean, num_steps, do_plot);


%% ========================================================================
%  MAIN LOCALIZATION LOOP
%  ========================================================================
function [mean_position_error, anees] = localize(env, policy, filt, x0, num_steps, do_plot)
    [states_nf, states_r, action_nf, obs_nf, obs_r] = rollout(env, x0, policy, num_steps, 0.1);
    
    states_filter = zeros(num_steps + 1, 3);
    states_filter(1, :) = x0';

    errors = zeros(num_steps, 3);
    position_errors = zeros(num_steps, 1);
    mahalanobis_errors = zeros(num_steps, 1);

    if do_plot
        figure(1); hold on; axis equal;
        plot_field(env);
    end

    for i = 1:num_steps
        x_real = states_r(i+1, :)';
        u_noisefree = action_nf(i, :)';
        z_real = obs_r(i, :)';
        marker_id = get_marker_id(env, i-1); % matlab indexing

        if isempty(filt)
            mean_val = x_real;
            cov_val = eye(3);
        else
            % filter update
            if strcmp(filt.type, 'ekf')
                filt = ekf_update(filt, env, u_noisefree, z_real, marker_id);
            elseif strcmp(filt.type, 'pf')
                filt = pf_update(filt, env, u_noisefree, z_real, marker_id);
            end
            mean_val = filt.mu;
            cov_val = filt.sigma;
        end
        states_filter(i+1, :) = mean_val';

        if do_plot && ~isempty(filt) && strcmp(filt.type, 'pf') && u_noisefree(1) > 0
            plot_particles(filt.particles);
        end

        err = mean_val - x_real;
        err(3) = minimized_angle(err(3));
        errors(i, :) = err';
        position_errors(i) = norm(err(1:2));

        cond_number = cond(cov_val);
        if cond_number > 1e12
            fprintf('Badly conditioned cov (setting to identity): %e\n', cond_number);
            cov_val = eye(3);
        end
        mahalanobis_errors(i) = err' * (cov_val \ err);
    end

    mean_position_error = mean(position_errors);
    mean_mahalanobis_error = mean(mahalanobis_errors);
    anees = mean_mahalanobis_error / 3;

    if ~isempty(filt)
        fprintf(repmat('-', 1, 80)); fprintf('\n');
        fprintf('Mean position error: %f\n', mean_position_error);
        fprintf('Mean Mahalanobis error: %f\n', mean_mahalanobis_error);
        fprintf('ANEES: %f\n', anees);
    end

    if do_plot
        plot_robot(x_real, z_real);
        plot(states_nf(:, 1), states_nf(:, 2), 'g', 'LineWidth', 0.5);
        plot(states_r(:, 1), states_r(:, 2), 'b');
        if ~isempty(filt)
            plot(states_filter(:, 1), states_filter(:, 2), 'r');
        end
        drawnow;
    end
end

%% ========================================================================
%  EXTENDED KALMAN FILTER
%  ========================================================================
function filt = ekf_update(filt, env, u, z, marker_id)
    % u: action
    % z: landmark observation
    % marker_id: landmark ID
    
    % YOUR IMPLEMENTATION HERE
    
    % (save the updated mean and covarinace)
    % filt.mu = ...
    % filt.sigma = ...
end

%% ========================================================================
%  PARTICLE FILTER
%  ========================================================================
function filt = pf_update(filt, env, u, z, marker_id)
    filt.particles = pf_move_particles(filt, env, u);
    
    % YOUR IMPLEMENTATION HERE (Weight calculation and Resampling)
    % filt.particles = pf_resample(filt.particles, filt.weights);
    
    [filt.mu, filt.sigma] = pf_mean_and_variance(filt.particles);
end

function new_particles = pf_move_particles(filt, env, u)
    new_particles = filt.particles;
    % YOUR IMPLEMENTATION HERE
end

function resampled_particles = pf_resample(particles, weights)
    % YOUR IMPLEMENTATION HERE (Low-variance sampler)
    resampled_particles = particles; 
end

function [mean_val, cov_val] = pf_mean_and_variance(particles)
    mean_pos = mean(particles, 1);
    mean_theta = atan2(sum(sin(particles(:, 3))), sum(cos(particles(:, 3))));
    mean_val = [mean_pos(1:2), mean_theta]';
    
    zero_mean = particles - mean_val';
    for i = 1:size(zero_mean, 1)
        zero_mean(i, 3) = minimized_angle(zero_mean(i, 3));
    end
    
    num_p = size(particles, 1);
    cov_val = (zero_mean' * zero_mean) / num_p;
    cov_val = cov_val + eye(3) * 1e-6; % Avoid bad conditioning
end

%% ========================================================================
%  FIELD & DYNAMICS
%  ========================================================================
function grad_G = G(env, x, u)
    prev_theta = x(3);
    rot1 = u(1);
    trans = u(2);
    % YOUR IMPLEMENTATION HERE
    grad_G = eye(3); 
end

function grad_V = V(env, x, u)
    prev_theta = x(3);
    rot1 = u(1);
    trans = u(2);
    % YOUR IMPLEMENTATION HERE
    grad_V = zeros(3, 3);
end

function grad_H = H(env, x, marker_id)
    % YOUR IMPLEMENTATION HERE
    grad_H = zeros(1, 3);
end

function x_next = forward_model(x, u)
    x_next = zeros(3, 1);
    theta = x(3) + u(1);
    x_next(1) = x(1) + u(2) * cos(theta);
    x_next(2) = x(2) + u(2) * sin(theta);
    x_next(3) = minimized_angle(theta + u(3));
end

function z = observe(env, x, marker_id)
    dx = env.MARKER_POS(1, marker_id) - x(1);
    dy = env.MARKER_POS(2, marker_id) - x(2);
    z = minimized_angle(atan2(dy, dx) - x(3));
end

function cov_val = noise_from_motion(u, alphas)
    v = zeros(3, 1);
    v(1) = alphas(1) * u(1)^2 + alphas(2) * u(2)^2;
    v(2) = alphas(3) * u(2)^2 + alphas(4) * (u(1)^2 + u(3)^2);
    v(3) = alphas(1) * u(3)^2 + alphas(2) * u(2)^2;
    cov_val = diag(v);
end

function u_noisy = sample_noisy_action(u, alphas)
    cov_val = noise_from_motion(u, alphas);
    u_noisy = mvnrnd(u', cov_val)';
end

function z_noisy = sample_noisy_observation(env, x, marker_id, beta_val)
    z = observe(env, x, marker_id);
    z_noisy = mvnrnd(z, beta_val);
end

function marker_id = get_marker_id(env, step)
    marker_id = mod(floor(step / 2), env.NUM_MARKERS) + 1;
end

function [states_nf, states_r, action_nf, obs_nf, obs_r] = rollout(env, x0, policy, num_steps, dt)
    states_nf = zeros(num_steps, 3);
    states_r  = zeros(num_steps, 3);
    action_nf = zeros(num_steps, 3);
    obs_nf    = zeros(num_steps, 1);
    obs_r     = zeros(num_steps, 1);
    
    x_nf = x0; x_real = x0;
    
    for i = 1:num_steps
        t = (i-1) * dt;
        
        u_nf = policy(x_real, t);
        x_nf = forward_model(x_nf, u_nf);
        
        u_real = sample_noisy_action(u_nf, env.alphas);
        x_real = forward_model(x_real, u_real);
        
        marker_id = get_marker_id(env, i-1);
        z_nf = observe(env, x_real, marker_id);
        z_real = sample_noisy_observation(env, x_real, marker_id, env.beta);
        
        states_nf(i, :) = x_nf';
        states_r(i, :)  = x_real';
        action_nf(i, :) = u_nf';
        obs_nf(i)       = z_nf;
        obs_r(i)        = z_real;
    end
    
    states_nf = [x0'; states_nf];
    states_r = [x0'; states_r];
end

%% ========================================================================
%  POLICIES
%  ========================================================================
function u = open_loop_rectangle_policy(t, dt)
    n = round(t / dt);
    index = mod(n, round(5 / dt));

    if index == 2 * round(1 / dt)
        u = [deg2rad(45); 100 * dt; deg2rad(45)];
    elseif index == 4 * round(1 / dt)
        u = [deg2rad(45); 0; deg2rad(45)];
    else
        u = [0; 100 * dt; 0];
    end
end

%% ========================================================================
%  UTILITIES
%  ========================================================================
function a = minimized_angle(a)
    a = mod(a + pi, 2*pi) - pi;
end

function plot_field(env)
    for m = 1:env.NUM_MARKERS
        x = env.MARKER_POS(1, m);
        y = env.MARKER_POS(2, m);
        rectangle('Position', [x-20, y-20, 40, 40], 'Curvature', [1 1], ...
            'EdgeColor', 'k', 'FaceColor', 'w');
        text(x, y, num2str(m), 'HorizontalAlignment', 'center');
    end
end

function plot_robot(x, z)
    radius = 5;
    rectangle('Position', [x(1)-radius, x(2)-radius, radius*2, radius*2], ...
        'Curvature', [1 1], 'EdgeColor', 'k', 'FaceColor', 'c');
    
    % Orientation
    plot([x(1), x(1) + cos(x(3)) * (radius + 5)], ...
         [x(2), x(2) + sin(x(3)) * (radius + 5)], 'k');
     
    % Observation
    plot([x(1), x(1) + cos(x(3) + z(1)) * 100], ...
         [x(2), x(2) + sin(x(3) + z(1)) * 100], 'b', 'LineWidth', 0.5);
end

function plot_particles(particles)
    scatter(particles(:, 1), particles(:, 2), 5, 'k', '.');
end