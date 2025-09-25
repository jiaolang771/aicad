#Add the CVX path on the computational cluster
addpath('/data/EPS-Data-Processing/EPS_data/dhcp_tutorial/wan5fg/cvx-a64/cvx/');

#load('IS_binary.mat');
load('example.mat');

cvx_setup

num_of_subject = 32;

L1 = cell(num_of_subject,350);
S1 = cell(num_of_subject,350);
% P = cell(num_of_subject,350);
window_size = 50;

tic
parpool(16);
parfor i = 1 : num_of_subject%size(temp,2)
    i+0*num_of_subject
    original_signal = normalize_column(data(i+0*num_of_subject).time_series');
%     for j = 1 : (size(original_signal, 2)- window_size)
    for j = 1 : 350
        temp = abs(data(i+1*num_of_subject).fa_sc);
        if any(isnan(temp(:)))
            temp(isnan(temp)) = 0;
        end
        temp = Laplacian(temp);
        temp = temp/(trace(temp)/82);
      
% [L{i,j}, S{i,j}] = dynamic_graph_learning(original_signal(:,(j:window_size+j-1)), 0, 0.5, IS_score(i,j), 20);
% [L{i,j}, S{i,j}] = dynamic_graph_learning(original_signal(:,(j:window_size+j-1)), 0, 0.5, 0, 2);    
        [L1{i,j}, S1{i,j}] = dynamic_graph_learning(original_signal(:,(j:window_size+j-1)), temp, 0.5, 0.2, 100);
    end
   
end
toc



save graph_L1 L1
save graph_S1 S1
