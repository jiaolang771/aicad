function [laplacian, graph_signal] = dynamic_graph_learning(original_signal, SC_Lap, alpha1, alpha2, alpha3)
%% Inputs:
% SC_Lap is the laplacian matrix of the structural connectivity.
% alpha1 is the coefficient regulating the smoothness of the learned
% Laplacian.
% alpha2 is the coefficient regulating the guidance of structural
% connectivity. A higher alpha2 is expected during IS states. This
% parameter should come from prior knowledge.
% alpha3 is the coefficient regulating the strength of the learned graph
% Laplacian.
%%
X = original_signal;

for i = 1 : 2

B = original_signal*original_signal';

if any(isnan(B(:)))
    disp('NaN value found in B')
    i
    break
else
    L = convex_solution(B, alpha3);
end

% L = convex_solution(B, alpha3);
if any(isnan(L(:)))
    disp('NaN value found in L')
    i
    break
end
%% choice of Objective function
%use this form if considering SC_Lap as F norm distance between learned
%Laplacian
% original_signal = ((eye(87)+alpha1*L)^-1)*X;     

%use this form if considering smoothness on SC.
%original_signal = ((eye(82)+alpha1*L+alpha2*SC_Lap)^-1)*X;
 original_signal = ((eye(82)+alpha1*L)^-1)*X;

end
laplacian = L;
graph_signal = original_signal;
