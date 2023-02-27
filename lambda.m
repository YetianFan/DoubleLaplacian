function [mean_result] = lambda(lamb1,lamb2,act_fun,repeat_num,epoch)

%%%% loading datasets  %%%%
train_set = loadMNISTImages("./train-images-idx3-ubyte")';
train_label = loadMNISTLabels("./train-labels-idx1-ubyte");
test_set = loadMNISTImages("./t10k-images-idx3-ubyte")';
test_label = loadMNISTLabels("./t10k-labels-idx1-ubyte");
numclass = 10;

randIndex = randperm(length(train_label));
train_set = train_set(randIndex,:);
train_label = train_label(randIndex);

train_set = double(train_set) / 255;
test_set  = double(test_set) / 255;  

set = [train_set; test_set];
threshold_pca = 0.95;
[set, mapping] = pca(set, threshold_pca);

SampleNum = size(set,1);
range = max(set)-min(set);
label  = range==0;
range(label) = 1;
set = (set - repmat(min(set),SampleNum,1))./repmat(range,SampleNum,1) - 0.5 ;
set = set .* repmat(~label,SampleNum,1);

train_set = set(1:60000,:);
test_set = set(60001:70000,:);

train_num = length(train_label);
test_num = length(test_label);
tmp1 = zeros(train_num,numclass);
for i = 1:train_num
    tmp1(i,train_label(i)+1) = 1;
end
train_label = tmp1;

tmp2 = zeros(test_num,numclass);
for i = 1:test_num
    tmp2(i,test_label(i)+1) = 1;
end
test_label = tmp2;

if lamb1 + lamb2 ==0
    i16 = 1;
    use_penalty = "no";              %%%% if no means without Laplacian regularization
else
    i16 = 2;
    use_penalty = "yes";             %%%% if yes means with Laplacian regularization
end

all_cor = []; 
for i17 = 1:repeat_num  
    [i16, i17]
    seed = i17;
    rng(seed);
	
    in_num = size(train_set,2);
    neu_nums = [in_num 100 80 60 numclass];   %%%% the structure of neural networks
    lr = 0.003;                               %%%% learning rate
    batch_size = 100;                         %%%% the batch size 
    rho = 0.9; 
    last_layer_penalty = "yes";       
    layers = numel(neu_nums);         
    train_num = size(train_set, 1);   
    batch_num = ceil(train_num / batch_size);  
    his_loss = zeros(epoch*batch_num,1);  
    
    if i16 == 1
        w = cell(1,layers-1);   
        for i1=2:layers
            w{i1-1} = (rand(neu_nums(i1),neu_nums(i1-1)+1)-0.5); 
        end
        save(['w',num2str(i17),'.mat'],'w')
        ID_matrix = zeros(repeat_num,train_num);
        for i = 1:repeat_num
            ID_matrix(i,:) = randperm(train_num);
        end
        save('ID_matrix.mat','ID_matrix')
    else
        load(['w',num2str(i17),'.mat']);
        load('ID_matrix.mat')
    end
    
    rms = cell(1,layers-1);   
    momentum = cell(1,layers-1);    
    for i2=2:layers
        rms{i2-1} = (rand(neu_nums(i2),neu_nums(i2-1)+1)-0.5) .* 0;  
        momentum{i2-1} = (rand(neu_nums(i2),neu_nums(i2-1)+1)-0.5).* 0; 
    end
    
    %%%% training the neural networks
    iters=1; 
    L = cell(1,batch_num);
    batch_idx = ID_matrix(i17,:);  
    cor_rate = [];
    for i7 = 1 : epoch
        for i8 = 1 : batch_num
            if i8 ~= batch_num
                batch_trian = train_set(batch_idx((i8 - 1) * batch_size + 1 : i8 * batch_size), :); 
                batch_label = train_label(batch_idx((i8 - 1) * batch_size + 1 : i8 * batch_size), :); 
            else
                batch_trian = train_set(batch_idx((i8 - 1) * batch_size + 1 : end), :); 
                batch_label = train_label(batch_idx((i8 - 1) * batch_size + 1 : end), :); 
            end
            
            subset_size = size(batch_trian,1);
            batch_trian = [ones(subset_size,1) batch_trian]; 
            out{1} = batch_trian;  
            for i6 = 2 : layers-1
                
                if act_fun == "tanh"
                    out{i6} = tansig(out{i6-1} * w{i6-1}');
                elseif act_fun == "ReLU"
                    out{i6} = (out{i6-1} * w{i6-1}') .* (out{i6-1} * w{i6-1}'>0);
                end
                out{i6} = [ones(subset_size,1) out{i6}]; 
            end
            
            out{layers} = logsig(out{layers - 1} * w{layers - 1}');
            sigma = zeros(subset_size,1);
			
			%%%% calculate the Laplacian matrix  %%%%%
            if i7 == 1
                for i12 = 1:subset_size
                    tmp_sigma =  norm(batch_trian(i12,2:end),2);
                    sigma(i12) = tmp_sigma^2;
                end
                sum_sigma = 1 / subset_size * sum(sigma);
                dis = exp(-squareform(pdist(batch_trian(:,2:end))) .^2  ./ sum_sigma);
                CrossMatrix = zeros(subset_size);
                Target = zeros(1,subset_size);
                for i = 1:subset_size
                    Target(i) = find(batch_label(i,:)==1);
                end
                label = unique(Target);
                for i = 1:length(label)
                    index = (Target==label(i));
                    CrossMatrix(index,index) = 1;
                end
                D1 = dis.*CrossMatrix.*(ones(subset_size)-eye(subset_size));
                D2 = dis.*(~CrossMatrix);
                sum_row = sum(D1, 2);
                for i11 = 1:subset_size
                    if sum_row(i11) >0
                        D1(i11,:) = D1(i11,:) ./ sqrt(sum_row(i11));
                        D1(:, i11) = D1(:, i11) ./ sqrt(sum_row(i11));
                    end
                end
                sum_row = sum(D2, 2);
                for i11 = 1:subset_size
                    if sum_row(i11) >0
                        D2(i11,:) = D2(i11,:) ./ sqrt(sum_row(i11));
                        D2(:, i11) = D2(:, i11) ./ sqrt(sum_row(i11));
                    end
                end
                D1 =  -D1 + diag(ones(1,subset_size));
                D2 =  -D2 + diag(ones(1,subset_size));
                L{i8} = lamb1.*D1 - lamb2.*D2;
            end
            
            loss = batch_label - out{layers};  
            his_loss(iters) = 1/2 * sum(sum(loss.^2)) / subset_size;
            iters=iters+1;
            delta{layers} = -loss .* (out{layers} .* (1 - out{layers})) ;
            for i14 = (layers - 1) : -1 : 2
                if act_fun == "tanh"
                    diff_act = 1 - tansig(out{i14}).^2; 
                elseif act_fun == "ReLU"
                    diff_act = (out{i14})>0;     
                end
                
                if i14+1==layers
                    delta{i14} = (delta{i14 + 1} * w{i14}) .* diff_act;
                else
                    delta{i14} = (delta{i14 + 1}(:,2:end) * w{i14}).* diff_act;
                end
            end
            
            for i3 = 1 : layers-1
                if i3 + 1 == layers
                    if (use_penalty == "yes") && (last_layer_penalty == "yes")
                        diff_w{i3} = (delta{i3 + 1}' * out{i3}) ./ size(delta{i3 + 1}, 1) +  w{i3} * out{i3}' * L{i8} * out{i3} ./ size(delta{i3 + 1}, 1);
                    else
                        diff_w{i3} = (delta{i3 + 1}' * out{i3}) ./ size(delta{i3 + 1}, 1);
                    end
                else
                    if use_penalty == "yes"
                        diff_w{i3} = (delta{i3 + 1}(:,2:end)' * out{i3}) ./ size(delta{i3 + 1}, 1) +  w{i3} * out{i3}' * L{i8} * out{i3} ./ size(delta{i3 + 1}, 1);
                    else
                        diff_w{i3} = (delta{i3 + 1}(:,2:end)' * out{i3}) ./ size(delta{i3 + 1}, 1);
                    end
                end
            end
            
            for i9=1:layers-1
                momentum{i9} = rho .* momentum{i9} + (1-rho) .* diff_w{i9}.* diff_w{i9};
                rms{i9} = lr ./ sqrt(momentum{i9} + eps);
            end
            
            for i4 = 1 : layers - 1
                w{i4} = w{i4} - rms{i4} .* diff_w{i4};
            end
            
        end
        
        
        %%%% test the neural networks  %%%%%
        test_size = size(test_set,1);
        test_out = out;  
        tmp_test_set = [ones(test_size,1) test_set]; 
        test_out{1} = tmp_test_set;
        for i5 = 2 : layers-1
            if act_fun == "tanh"
                test_out{i5} = tansig(test_out{i5 - 1} * w{i5 - 1}');
            elseif act_fun == "ReLU"
                test_out{i5} = (test_out{i5 - 1} * w{i5 - 1}').*(test_out{i5 - 1} * w{i5 - 1}'>0);
            end
            test_out{i5} = [ones(test_size,1) test_out{i5}];
        end
        test_out{layers} = logsig(test_out{layers - 1} * w{layers - 1}');
        
        [~, i15] = max(test_out{end},[],2);
        labels = i15;
        [~, correct] = max(test_label,[],2);
        wrong = find(labels ~= correct);
        wrong_rat = numel(wrong) / size(test_set, 1);
        cor_rate = [cor_rate; (1 - wrong_rat)];
        
    end
    
    all_cor = [all_cor cor_rate];
    
end

mean_result = mean(all_cor, 2);
mean_result = mean_result';