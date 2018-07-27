function []=dealData()
    test=single(csvread('TFIDF/test/TF.csv'));
    len=[1,10001,20001,30001,40001,50001,60001,70001,80001,90049];
    i=1;
    P=cell(1,9);
    %%---------ÒÔÏÂÎªPCA
    for i=1:9
        tmp=single(csvread(['TFIDF/train/TF/train_' num2str(i-1) '.csv']));
        tmp = [tmp;test(:,len(i):len(i+1)-1)];
        [COEFF,SCORE,latent] = pca(tmp);
        l=max(find(latent>=7e-7))
        save(['pca' num2str(i) '.mat'],'tmp','COEFF','latent','-v7.3');
        P{i}=tmp*COEFF(:,1:l);
    end
    train_test=[P{1} P{2} P{3} P{4} P{5} P{6} P{7} P{8} P{9}];
    ans_sign= single(csvread('TFIDF/train/TF/result.csv'));
    save('train_test.mat','train_test','ans_sign','-v7.3');
end