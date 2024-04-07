% {

Data Loading and Visualization:

Loads training and test datasets from 'P_Data.mat'.
Plots the training and test datasets with different markers and colors.

}%
clear all
close all


load P_Data.mat  % load training dataset

figure, 
plot (x_train1(1,1:100), x_train1(2,1:100), 'b+');
hold on
plot (x_train1(1,101:200), x_train1(2,101:200), 'r*');
hold on
plot (x_train1(1,201:300), x_train1(2,201:300), 'go');

xlabel('parameters (in millions)');
ylabel('parameters (in millions) ');

xlim([0 2.5]);
ylim([0 2.5]);

figure, 
plot (x_test1(1,1:100), x_test1(2,1:100), 'b+');
hold on
plot (x_test1(1,101:200), x_test1(2,101:200), 'r*');
hold on
plot (x_test1(1,201:300), x_test1(2,201:300), 'go');

xlabel('parameters (in millions)');
ylabel('parameters (in millions) ');
xlim([0 2.5]);
ylim([0 2.5]);

num_of_instances = size (x_train1,2);
num_of_instances_test = size (x_test1,2);

x = x_train1;
r = r_train1;




x = [ones(1,num_of_instances);x];                   %% add bias as input (x_0). 
x_test = [ones(1,num_of_instances_test);x_test1];    %% add bias as input (x_0). 


D = size (x,1);
K = size (r,1);  % number of output neurons = 3
H = 5; % number of hidden neurons (including bias)
w = rand(D,H);
w = (w-0.5)*2*0.01;  % (-0.01,0.01)
delta_w =zeros(D,H);

v = rand(H,K);
v = (v-0.5)*2*0.01;  % (-0.01,0.01)
delta_v =zeros(H,K);

lr = 0.05;  % learning rate
num_of_instances = size (x,2);
num_of_epoch = 20000;
weights_v = zeros(H,K,num_of_epoch);
weights_w = zeros(D,H,num_of_epoch);
y_epoch = zeros(K, num_of_instances, num_of_epoch);
accuracy_training = zeros(num_of_epoch,1);
accuracy_test= zeros(num_of_epoch,1);
for e=1:1:num_of_epoch
    for iter=1:1:num_of_instances
        instance = floor(rand * num_of_instances + 1); %% select the instance randomly 
        o = (w')*x(:,instance);
        z = sigmoid_hidden (o); 
        o = (v')*z;            
        y = softmax_output(o);  % use softmax for multiple output units (number of classes > 2)
        for i=1:1:K
            for h=1:1:H
              delta_v(h,i) = 0;
            end
        end
        for i=1:1:K
            for h=1:1:H
              delta_v(h,i) = lr*(r(i,instance)-y(i))*z(h);
            end
        end        
        for h = 1:1:H
            for j=1:1:D
                delta_w(j,h) = 0;
                for i = 1:1:K
                    delta_w(j,h) = delta_w(j,h)+ (r(i,instance)-y(i))*v(h,i);
                end
                delta_w(j,h) = lr*delta_w(j,h)*z(h)*(1-z(h))*x(j,instance);
            end
        end
        for i = 1:1:K
            for h=1:1:H
                v(h,i) = v(h,i) + delta_v(h,i);
            end
        end
        for h = 1:1:H
            for j=1:1:D
                w(j,h) = w(j,h) + delta_w(j,h);
            end
        end        
    end
    weights_v(:,:,e) = v;
    weights_w(:,:,e) = w;
    mse_training = 0;
    for instance=1:1:num_of_instances
        o = (w')*x(:,instance);
        z = sigmoid_hidden (o); 
        o = (v')*z;
        y(:,instance) = softmax_output (o);
        mse_instance = 0;
        for i = 1:1:K
            mse_instance = mse_instance + (r(i,instance) - y(i,instance))^2;
        end
        mse_training = mse_training + mse_instance/K;
        
        [M, index] = max (y(:,instance));   %% select the class for which the network outputs highest probability
        if r(index,instance) == 1   %% check whether classificaiton is correct
            accuracy_training(e) = accuracy_training(e) + 1;
        end
    end
    error_training (1, e) = mse_training/num_of_instances;
    accuracy_training(e) = accuracy_training(e)/num_of_instances*100;
    mse_test = 0;
    for instance=1:1:num_of_instances_test
        o = (w')*x_test(:,instance);
        z = sigmoid_hidden (o);
        o = (v')*z;
        y_test(:,instance) = softmax_output (o);
        mse_instance = 0;
        for i = 1:1:K
            mse_instance = mse_instance + (r_test1(i,instance) - y_test(i,instance))^2;
        end
        mse_test = mse_test + mse_instance/K;
        [M, index] = max (y_test(:,instance));   %% select the class for which the network outputs highest probability
        if r_test1(index,instance) == 1   %% check whether classificaiton is correct
            accuracy_test(e) = accuracy_test(e) + 1;
        end
    end
    error_test (1, e) = mse_test/num_of_instances_test;    
    accuracy_test(e) = accuracy_test(e)/num_of_instances_test*100;
    if (error_training (1, e) < 0.000025)
        %         if (error_training (1, e) < 0.03) && (error_test (1, e) < 0.01)
        break;   % little error for training and test data achieved. terminate training
    end
    if (mod(e,1000) == 0)
        disp(e);
    end
end

figure, 
plot (accuracy_training(1:e));
xlabel("epochs");
ylabel("accuracy training (%)");
ylim([0 100]);

figure, 
plot (accuracy_test(1:e));
xlabel("epochs");
ylabel("accuracy test(%)");
ylim([0 100]);

net = patternnet(10);
net = train(net,x_train1,r_train1);
YPredicted = net(x_train1);
YPredicted(:,1:10)

plotconfusion(r_train1,YPredicted)

hold on

figure,
title("MSE through epochs");
plot (1:e, error_training(1,1:e), 'b');
hold on
plot (1:e, error_test(1,1:e)), 'r';
xlabel("epochs");
ylabel("MSE");
figure, 
plot (x_train1(1,1:100), x_train1(2,1:100), 'k+');
hold on
plot (x_train1(1,101:200), x_train1(2,101:200), 'k*');
hold on
plot (x_train1(1,201:300), x_train1(2,201:300), 'ko');
xlim([0 2.5]);
ylim([0 2.5]);
for i = 0:0.05:2.5
    for j = 0:0.05:2.5
        x_grid = [1; i ; j]; %% [x0=1 ; x1 ; x2]
        o = (w')* x_grid;
        z = sigmoid_hidden (o); 
        o = (v')*z;
        y = softmax_output (o); 
        [M, index] = max (y);   %% select the class for which the network outputs highest probability
        if index == 1
            plot (i , j, 'b.');
            hold on
        elseif index == 2
            plot (i , j, 'r.');
            hold on
        elseif index == 3
            plot (i , j, 'g.');
            hold on
        end
    end
end
title ("Decision boundaries and training dataset");


figure, 
plot (x_test1(1,1:100), x_test1(2,1:100), 'k+');
hold on
plot (x_test1(1,101:200), x_test1(2,101:200), 'k*');
hold on
plot (x_test1(1,201:300), x_test1(2,201:300), 'ko');
xlim([0 2.5]);
ylim([0 2.5]);
for i = 0:0.05:2.5
    for j = 0:0.05:2.5
        x_grid = [1; i ; j]; %% [x0=1 ; x1 ; x2]
        o = (w')* x_grid;
        z = sigmoid_hidden (o); 
        o = (v')*z;
        y = softmax_output (o); 
        [M, index] = max (y);   %% select the class for which the network outputs highest probability
        if index == 1
            plot (i , j, 'b.');
            hold on
        elseif index == 2
            plot (i , j, 'r.');
            hold on
        elseif index == 3
            plot (i , j, 'g.');
            hold on
        end
    end
end
title ("Decision boundaries and test (validation) dataset");


for h = 2:1:H   % exclude the sypases to bias z0
    plot_data_w(h).weights = zeros (D,e);
    for j = 1:1:D
        for epo= 1:1:e
            plot_data_w(h).weights(j,epo) = weights_w(j,h,epo);
        end
    end
end

for h = 2:1:H   % exclude the sypases to bias z0
    figure,
    for j = 1:1:D   
        plot (plot_data_w(h).weights(j,:));
        hold on;
    end
    title(['Synapses connected to Hidden Layer - Neuron ', num2str(h)-1]);
    ylabel ('weights');
    xlabel('epochs');
end


for i = 1:1:K  
    plot_data_v(i).weights = zeros (H,e);
    for h = 1:1:H
        for epo= 1:1:e
            plot_data_v(i).weights(h,epo) = weights_v(h,i,epo);
        end
    end
end


for i = 1:1:K   
    figure,
    for h = 1:1:H   
        plot (plot_data_v(i).weights(h,:));
        hold on;
    end
    title(['Synapses connected to Output Neuron', num2str(i)]);
    ylabel ('weights');
    xlabel('epochs');
end






function z = sigmoid_hidden(o)
    for i=1:1:size (o,1)
        z(i,1) = 1/(1+exp(-o(i,1)));  
    end
    z(1,1) = 1;  % z0 bias = 1 always
end

function y = sigmoid_output(o)
    for i=1:1:size (o,1)
        y(i,1) = 1/(1+exp(-o(i,1)));  
    end
end

function y = threshold(o)
    y = o > 0.5;  
end

function y = softmax_output(o)
    denominator = 0;
    for i=1:1:size (o,1)
        denominator = denominator + exp(o(i,1));  
    end    
    for i=1:1:size (o,1)
        y(i,1) = exp(o(i,1))/denominator;  
    end
end


