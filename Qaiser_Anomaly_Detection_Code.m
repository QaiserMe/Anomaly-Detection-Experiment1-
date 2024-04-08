% Clear workspace and close all figures
clear all
close all

% Load training dataset
load P_Data.mat

% Plot training dataset
figure
plot(x_train1(1,1:100), x_train1(2,1:100), 'b+'); % Plot class 1 data points
hold on
plot(x_train1(1,101:200), x_train1(2,101:200), 'r*'); % Plot class 2 data points
hold on
plot(x_train1(1,201:300), x_train1(2,201:300), 'go'); % Plot class 3 data points
xlabel('parameters (in millions)'); % X-axis label
ylabel('parameters (in millions)'); % Y-axis label
xlim([0 2.5]); % Set X-axis limits
ylim([0 2.5]); % Set Y-axis limits

% Plot test dataset
figure
plot(x_test1(1,1:100), x_test1(2,1:100), 'b+'); % Plot class 1 data points
hold on
plot(x_test1(1,101:200), x_test1(2,101:200), 'r*'); % Plot class 2 data points
hold on
plot(x_test1(1,201:300), x_test1(2,201:300), 'go'); % Plot class 3 data points
xlabel('parameters (in millions)'); % X-axis label
ylabel('parameters (in millions)'); % Y-axis label
xlim([0 2.5]); % Set X-axis limits
ylim([0 2.5]); % Set Y-axis limits

% Parameters
num_of_instances = size(x_train1, 2); % Number of training instances
num_of_instances_test = size(x_test1, 2); % Number of test instances
x = x_train1; % Input data
r = r_train1; % Output labels
x = [ones(1,num_of_instances); x]; % Add bias term to input data
x_test = [ones(1,num_of_instances_test); x_test1]; % Add bias term to test data
D = size(x, 1); % Dimensionality of input
K = size(r, 1); % Number of output neurons
H = 5; % Number of hidden neurons
w = rand(D,H) * 0.01 - 0.005; % Initialize weights for input layer
delta_w = zeros(D,H); % Initialize weight change for input layer
v = rand(H,K) * 0.01 - 0.005; % Initialize weights for hidden layer
delta_v = zeros(H,K); % Initialize weight change for hidden layer
lr = 0.05; % Learning rate
num_of_epoch = 20000; % Number of training epochs
weights_v = zeros(H,K,num_of_epoch); % Array to store weights of hidden layer over epochs
weights_w = zeros(D,H,num_of_epoch); % Array to store weights of input layer over epochs
y_epoch = zeros(K, num_of_instances, num_of_epoch); % Array to store output over epochs for training instances
accuracy_training = zeros(num_of_epoch,1); % Array to store training accuracy over epochs
accuracy_test = zeros(num_of_epoch,1); % Array to store test accuracy over epochs

% Training
for e = 1:num_of_epoch % Loop over epochs
    for iter = 1:num_of_instances % Loop over training instances
        instance = floor(rand * num_of_instances + 1); % Randomly select an instance
        o = w' * x(:,instance); % Compute input to hidden layer
        z = sigmoid_hidden(o); % Apply activation function to hidden layer
        o = v' * z; % Compute input to output layer
        y = softmax_output(o); % Apply activation function to output layer
        delta_v = lr * (r(:,instance) - y) * z'; % Compute weight change for hidden-output layer
        delta_w = lr * (v * (r(:,instance) - y) .* z .* (1 - z)) * x(:,instance)'; % Compute weight change for input-hidden layer
        v = v + delta_v; % Update weights for hidden-output layer
        w = w + delta_w; % Update weights for input-hidden layer
    end
    weights_v(:,:,e) = v; % Save weights of hidden layer for current epoch
    weights_w(:,:,e) = w; % Save weights of input layer for current epoch
    mse_training = 0; % Initialize mean squared error for training
    for instance = 1:num_of_instances % Loop over training instances
        o = w' * x(:,instance); % Compute input to hidden layer
        z = sigmoid_hidden(o); % Apply activation function to hidden layer
        o = v' * z; % Compute input to output layer
        y(:,instance) = softmax_output(o); % Apply activation function to output layer
        mse_instance = sum((r(:,instance) - y(:,instance)).^2); % Compute squared error for current instance
        mse_training = mse_training + mse_instance / K; % Accumulate squared error
        [~, index] = max(y(:,instance)); % Determine predicted class
        if r(index,instance) == 1 % Check if prediction matches actual class
            accuracy_training(e) = accuracy_training(e) + 1; % Increment accuracy counter
        end
    end
    error_training(e) = mse_training / num_of_instances; % Compute mean squared error for training
    accuracy_training(e) = accuracy_training(e) / num_of_instances * 100; % Compute accuracy for training
    mse_test = 0; % Initialize mean squared error for test
    for instance = 1:num_of_instances_test % Loop over test instances
        o = w' * x_test(:,instance); % Compute input to hidden layer for test instance
        z = sigmoid_hidden(o); % Apply activation function to hidden layer for test instance
        o = v' * z; % Compute input to output layer for test instance
        y_test(:,instance) = softmax_output(o); % Apply activation function to output layer for test instance
        mse_instance = sum((r_test1(:,instance) - y_test(:,instance)).^2); % Compute squared error for current instance
        mse_test = mse_test + mse_instance / K; % Accumulate squared error
        [~, index] = max(y_test(:,instance)); % Determine predicted class for test instance
        if r_test1(index,instance) == 1 % Check if prediction matches actual class for test instance
            accuracy_test(e) = accuracy_test(e) + 1; % Increment accuracy counter
        end
    end
    error_test(e) = mse_test / num_of_instances_test; % Compute mean squared error for test
    accuracy_test(e) = accuracy_test(e) / num_of_instances_test * 100; % Compute accuracy for test
    if error_training(e) < 0.000025 % Check if training error is below threshold
        break; % Exit training loop if condition met
    end
    if mod(e,1000) == 0 % Check if current epoch is a multiple of 1000
        disp(e); % Display current epoch
    end
end

% Plot accuracy over epochs for training and test datasets
figure
plot(accuracy_training(1:e));
xlabel('epochs');
ylabel('accuracy training (%)');
ylim([0 100]);

figure
plot(accuracy_test(1:e));
xlabel('epochs');
ylabel('accuracy test(%)');
ylim([0 100]);

% Patternnet comparison
net = patternnet(10);
net = train(net,x_train1,r_train1);
YPredicted = net(x_train1);
YPredicted(:,1:10)
plotconfusion(r_train1,YPredicted);
hold on

% Plot MSE through epochs
figure
plot(1:e, error_training(1:e), 'b');
hold on
plot(1:e, error_test(1:e), 'r');
xlabel('epochs');
ylabel('MSE');

% Plot decision boundaries and datasets for training and test datasets
figure
plot_decision_boundaries(x_train1, r_train1, weights_w(:,:,e), weights_v(:,:,e), 'Training Dataset');
figure
plot_decision_boundaries(x_test1, r_test1, weights_w(:,:,e), weights_v(:,:,e), 'Test Dataset');

% Plot synapse weights connected to hidden and output neurons for each class
plot_synapse_weights(weights_w, weights_v, H, K);

% Helper functions
function z = sigmoid_hidden(o)
    z = 1 ./ (1 + exp(-o)); % Apply sigmoid activation function to input
    z(1) = 1; % Set bias neuron to 1
end

function y = softmax_output(o)
    y = exp(o) ./ sum(exp(o)); % Apply softmax activation function to input
end

function plot_decision_boundaries(x_data, r_data, weights_w, weights_v, title_text)
    plot(x_data(1,1:100), x_data(2,1:100), 'k+'); % Plot class 1 data points
    hold on
    plot(x_data(1,101:200), x_data(2,101:200), 'k*'); % Plot class 2 data points
    hold on
    plot(x_data(1,201:300), x_data(2,201:300), 'ko'); % Plot class 3 data points
    xlim([0 2.5]); % Set X-axis limits
    ylim([0 2.5]); % Set Y-axis limits
    for i = 0:0.05:2.5 % Iterate over X-axis
        for j = 0:0.05:2.5 % Iterate over Y-axis
            x_grid = [1; i ; j]; % Create grid point
            o = weights_w' * x_grid; % Compute input to hidden layer
            z = sigmoid_hidden(o); % Apply activation function to hidden layer
            o = weights_v' * z; % Compute input to output layer
            y = softmax_output(o); % Apply activation function to output layer
            [~, index] = max(y); % Determine predicted class
            if index == 1 % Check predicted class
                plot(i , j, 'b.'); % Plot grid point as class 1
                hold on
            elseif index == 2 % Check predicted class
                plot(i , j, 'r.'); % Plot grid point as class 2
                hold on
            elseif index == 3 % Check predicted class
                plot(i , j, 'g.'); % Plot grid point as class 3
                hold on
            end
        end
    end
    title(title_text); % Set title
end

function plot_synapse_weights(weights_w, weights_v, H, K)
    % Plot synapse weights for each hidden neuron
    for h = 2:H % Exclude bias neuron
        plot_data_w(h).weights = zeros(D,e);
        for j = 1:D
            for epo = 1:e
                plot_data_w(h).weights(j,epo) = weights_w(j,h,epo); % Store weights over epochs
            end
        end
    end
    for h = 2:H % Exclude bias neuron
        figure
        for j = 1:D
            plot(plot_data_w(h).weights(j,:)); % Plot weight evolution over epochs
            hold on;
        end
        title(['Synapses connected to Hidden Layer - Neuron ', num2str(h)-1]); % Set title
        ylabel('weights'); % Set Y-axis label
        xlabel('epochs'); % Set X-axis label
    end

    % Plot synapse weights for each output neuron
    for i = 1:K
        plot_data_v(i).weights = zeros(H,e);
        for h = 1:H
            for epo = 1:e
                plot_data_v(i).weights(h,epo) = weights_v(h,i,epo); % Store weights over epochs
            end
        end
    end
    for i = 1:K
        figure
        for h = 1:H
            plot(plot_data_v(i).weights(h,:)); % Plot weight evolution over epochs
            hold on;
        end
        title(['Synapses connected to Output Neuron ', num2str(i)]); % Set title
        ylabel('weights'); % Set Y-axis label
        xlabel('epochs'); % Set X-axis label
    end
end
