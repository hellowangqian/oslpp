function [acc, acc_per_class] = OSDA_OSLPP(domainS_features,domainS_labels,domainT_features,domainT_labels,options)
num_iter = options.T;
options.ReducedDim = options.lppDim;
options.alpha = 1;
source_classes = unique(domainS_labels);
num_class = length(source_classes);
W = zeros(size(domainS_features,1)+size(domainT_features,1));
W_s = constructW(domainS_labels);
W(1:size(W_s,1),1:size(W_s,2)) =  W_s;
% looping
rejectable = zeros(1, length(domainT_labels));
fprintf('d=%d\n',options.ReducedDim);
for iter = 1:num_iter
    P = OSLPP([domainS_features;domainT_features],W,options);
    %P = eye(size(domainS_features,2));
    domainS_proj = domainS_features*P;
    domainT_proj = domainT_features*P;
    proj_mean = mean([domainS_proj;domainT_proj]);
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);
    %% distance to class means
    classMeans = zeros(num_class,size(domainS_proj,2));
    for i = 1:num_class
        classMeans(i,:) = mean(domainS_proj(domainS_labels==source_classes(i),:));
    end
    classMeans = L2Norm(classMeans);    
    distClassMeans = EuDist2(domainT_proj,classMeans);
    expMatrix = exp(-distClassMeans);
    probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
    [prob,predLabels] = max(probMatrix'); 
    p=1-iter/(num_iter-1);
    p = max(p,0);
    [sortedProb,index] = sort(prob);
    sortedPredLabels = predLabels(index);
    % class-wise selection
    trustable = zeros(1,length(prob));
    for i = 1:num_class
        thisClassProb = sortedProb(sortedPredLabels==i);
        if length(thisClassProb)>0
            trustable = trustable+ (prob>thisClassProb(floor(length(thisClassProb)*p)+1)).*(predLabels==i);
        end
    end
    %% detect rejectable based on one-nearest-neighbour
    trustable = trustable.*(1-rejectable);
    % choose to reject samples
    if iter >= 2
        if iter == 2 % initially reject samples
            sorted_prob = sort(prob);            
            rejectable(prob<=sorted_prob(options.n_r)) = 1; % probability smaller than a threshold
        end
        % reject those close to the rejected points
        dist2rejPoints = EuDist2(domainT_proj,domainT_proj(logical(rejectable),:));
        dist2trustPoints = EuDist2(domainT_proj, domainT_proj(logical(trustable),:));
        rejectable(min(dist2rejPoints') < min(dist2trustPoints')) = 1;
    end
    %%  calculate the results: OS, OS*, UNK, etc.
    trustable = trustable.*(1-rejectable);
    pseudoLabels = source_classes(predLabels); % selected  samples are indicated by their pseudo-labels
    pseudoLabels(~trustable) = -1; % uncertain samples are indicated by -1
    hybridLabels = pseudoLabels;
    hybridLabels(logical(rejectable)) = -2; % rejected samples are indicaed by -2
    W = constructW([domainS_labels,hybridLabels]);
    %% calculate ACC
    domainT_labels_merged = -1*ones(1,length(domainT_labels)); % merge the unknown classes as on unified class -1
    for i = 1:num_class
        acc_per_class(iter,i) = nansum((pseudoLabels == domainT_labels).*(domainT_labels==source_classes(i)))/nansum(domainT_labels==source_classes(i));
        domainT_labels_merged(domainT_labels==source_classes(i)) = source_classes(i);
    end
    acc_per_class(iter,i+1) = nansum((pseudoLabels == domainT_labels_merged).*(domainT_labels_merged==-1))/nansum(domainT_labels_merged==-1);
    acc(iter) = sum(pseudoLabels==domainT_labels_merged)/length(domainT_labels);
    fprintf('Iteration=%d, All:%0.3f, OS: %0.3f, OS*: %0.3f, UNK: %0.3f\n', iter, acc(iter), nanmean(acc_per_class(iter,:)), nanmean(acc_per_class(iter,1:end-1)), acc_per_class(iter,end));
end
