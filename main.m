%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%   this code is designed for double Laplcian regularization  %%%%
%%%%   fully connected neural networks  %%%%
%%%%   date: 2023-02-01  %%%%

%%%% input parameters %%%%
act_fun = "ReLU";    %%% the activation function,  "tanh" or "ReLU"
repeat_num = 50;      %%% the number of experments  
lamb1 = -5:-1;       %%% the tunning parameters for attraction Laplacian  
lamb1 = 10.^lamb1;
lamb1 = [0 lamb1];
lamb2 = -18:-12;     %%% the tunning parameters for repulsion Laplacian  
lamb2 = 10.^lamb2;
lamb2 = [0 lamb2];
epoch = 100;          %%% the number of iterations

allmatrix = zeros(length(lamb1)*length(lamb2),epoch);
for i = 1:length(lamb1)
    for j = 1:length(lamb2)
        allmatrix((i-1)*length(lamb2)+j,:) = lambda(lamb1(i),lamb2(j),act_fun,repeat_num,epoch);
    end  
end

%%%% save the final results %%%%
save allmatrix
xlswrite('results.csv',allmatrix);

