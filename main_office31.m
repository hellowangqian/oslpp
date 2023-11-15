%% Loading Data:
% Features are extracted using resnet50 pretrained on ImageNet without
% fine-tuning
clear all %#ok<CLALL>
addpath('./utils/');
data_dir = './data/';
domains = {'A','D','W'};
count = 0; % count the adaptation task
options.pcaDim = 16; % dimensionality of the PCA space
options.lppDim = 16; % dimensionality of the OSLPP space
options.T = 10; % number of iterations
options.n_r = 140;
tic
for source_domain_index = 1:length(domains)
    load([data_dir 'office-' domains{source_domain_index} '-resnet50-noft']);
    domainS_features_ori = L2Norm(resnet50_features);
    domainS_labels = labels+1;
    sourceInstanceSelector = domainS_labels <= 10;
    domainS_features_ori = domainS_features_ori(sourceInstanceSelector,:);
    domainS_labels = domainS_labels(sourceInstanceSelector);
    
    for target_domain_index = 1:length(domains)
        if target_domain_index == source_domain_index
            continue;
        end
        fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
        load([data_dir 'office-' domains{target_domain_index} '-resnet50-noft']);
        domainT_features = L2Norm(resnet50_features);
        domainT_labels = labels+1;
        targetInstanceSelector = (domainT_labels <= 10) + (domainT_labels >= 21);
        targetInstanceSelector = logical(targetInstanceSelector);
        domainT_features = domainT_features(targetInstanceSelector,:);
        domainT_labels = domainT_labels(targetInstanceSelector);

        if options.pcaDim > 0
            opts.ReducedDim = options.pcaDim;
            X = double([domainS_features_ori;domainT_features]);
            P_pca = PCA(X,opts);
            domainS_features = domainS_features_ori*P_pca;
            domainT_features = domainT_features*P_pca;
            domainS_features = L2Norm(domainS_features);
        else
            domainS_features = L2Norm(domainS_features_ori);
        end
        domainT_features = L2Norm(domainT_features);
        num_class = length(unique(domainT_labels));
        %% Proposed method:
        [acc,acc_per_class]= OSDA_OSLPP(domainS_features,domainS_labels,domainT_features,domainT_labels,options);          
        count = count + 1;
        all_acc_per_class(count,:) = nanmean(acc_per_class,2);
        all_acc_per_image(count,:) = acc;
        all_acc_per_class2(count,:) = nanmean(acc_per_class(:,1:end-1),2);
        all_acc_unk(count,:) = acc_per_class(:,end);
    end
end
mean_acc_per_class = nanmean(all_acc_per_class,1); %OS
mean_acc_per_image = nanmean(all_acc_per_image,1); % ALL
mean_acc_per_class2 = nanmean(all_acc_per_class2,1); % OS*
numS = length(unique(domainS_labels));
mean_acc_unknown = mean_acc_per_class*(numS+1) - mean_acc_per_class2*numS; %UNK
HOS = 2*all_acc_per_class2.*all_acc_unk./(all_acc_per_class2+all_acc_unk);
fprintf('Average performance over all tasks: OS=%2.1f, OS*=%2.1f, UNK=%2.1f, HOS=%2.1f\n',100*mean_acc_per_class(end),100*mean_acc_per_class2(end),100*mean_acc_unknown(end),100*mean(HOS(:,end)));
save(['office31-OSDA-InitRejNum-' num2str(options.n_r) '-PcaDim-' num2str(options.pcaDim) '-LppDim-' num2str(options.lppDim) '-T-' num2str(options.T) '.mat'],'all_*','mean_*', 'HOS');
toc
