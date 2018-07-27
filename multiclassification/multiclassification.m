function []=multiclassification()
%%数据载入预处理
    load('train_test.mat');
    all=train_test;
    all  = mapminmax(all',0,1)';
    Train=all(1:62522,:);
    Train=[Train,ans_sign];
    [row,col] = size(Train);
    test=all(62523:71193,:);
%%训练集验证经济划分
    Train = Train(randperm(row), :);
    train=Train(1:62522,1:col-3);
    train_ans=Train(1:62522,col-2:col);
    val=Train(52523:62522,1:col-3);
    val_ans=Train(52523:62522,col-2:col);
    G_train=train;
    G_train_ans=train_ans;
    G_val=val;
    G_val_ans=val_ans;
%%参数设置
    Node = [col-3 ,60,3];
    layers = size(Node,2);
    step = [0.001, 0.001,0.001,0.001];
    lambda = [0.001000 0.00100 0.001000 0.00100];
    iter =200000;
    G_Node=single(Node);
    G_step =single(step);
    G_lambda=single(lambda);
    G_iter=single(iter);

    [w,b,best_drop,ow,ob,odrop]=BPNN(G_train,G_train_ans,G_val,G_val_ans,G_iter, G_Node, layers, G_step, G_lambda);
%%结果获取
    all_ans=get_ans(test,w,b,best_drop,layers);
    [m,l] = max(all_ans, [], 2);
    all_ans(all_ans>=m)=1;
    all_ans(all_ans<m)=0;
    str=sum(all_ans .* [1,2,3], 2);
    csvwrite('66_v1.csv',str);
end
function [w,theta,best_drop,ow,ob,odrop] = BPNN(train,train_ans,val,val_ans,iter, Node, layers, step, lambda)
    [train_row,train_col] = size(train);
    [val_row,val_col] = size(val);
    %%数据初始化
    w = cell(1, layers-1);
    out = cell(1, layers - 1);
    out_val = cell(1, layers - 1);
    delta = cell(1, layers - 1);
    theta = cell(1, layers - 1);
    drop =  cell(1, layers - 1);
    m = cell(1, layers - 1);
    v = cell(1, layers - 1);
    max_F1=0.5;
    best_w = w;
    best_b = theta;
    best_drop = drop;
    layer=layers;
    layers=single(layers);
    low_acc =zeros(1,iter,'single');
    mid_acc =zeros(1,iter,'single');
    hig_acc =zeros(1,iter,'single');
    beta1 = 0.9;
    beta2 = 0.999;
    p=0.5;
%     beta1 = gpuArray(beta1);
%     beta2 = gpuArray(beta2);
%     p = gpuArray(p);
%     lengths = gather(iter);
    cnt=1;
    h=waitbar(0,'iter');
    %%--------------初始化权值矩阵----------------
    for i = 1 :layers-1
        w{i} = rand(Node(i), Node(i+1),'single')-0.5;
        m{i} =zeros(Node(i), Node(i+1),'single');
        v{i} = zeros(Node(i), Node(i+1),'single');
        theta{i} = rand(1,Node(i+1),'single')*0.2-0.1;
        drop{i} = rand(1,Node(i+1), 'single')<p;
    end
    ow=w;
    ob=theta;
    odrop=drop;
    %%--------------迭代----------------------------
    for it = 1 : iter
                waitbar(it/iter,h,num2str(it));
        %cnt=cnt+1;
        %%------------计算各层结果--------------------
        %输入层->隐藏层
        out{1} = 1 ./ (1 + exp(-train * w{1} + theta{1}));%训练集
        out_val{1} = 1 ./ (1 + exp(-val * w{1} + theta{1}));%验证集
        %隐藏层
        for i=2 : layers-2
             out{i} =1 ./ (1 + exp(-out{i - 1}   * w{i}  + theta{i}));%训练集
             out_val{i} = 1 ./ (1 + exp(-out_val{i - 1}   * w{i} + theta{i}));%验证集
        end
        %输出层
        out{layers-1} =  1 ./ (1 +exp(out{layers-2} * w{layers-1} + theta{layers-1})) ;%训练集
        out_val{layers-1} = 1 ./ (1 +exp(out_val{layers-2} * w{layers-1}+ theta{layers-1})) ;%验证集e
        %%------验证集结果
        all_ans=out_val{layers-1};
        [n,l] = max(all_ans, [], 2);
        all_ans(all_ans>=n)=1;
        all_ans(all_ans<n)=0;
        S=val_ans + all_ans*10;
        low_acc(it) = sum(S(:,1)==11) / (sum(S(:,1)==1)+sum(S(:,1)==11));
        mid_acc(it) = sum(S(:,2)==11) / (sum(S(:,2)==1)+sum(S(:,2)==11));
        hig_acc(it) = sum(S(:,3)==11 )/ (sum(S(:,3)==1)+sum(S(:,3)==11));
        
        %%-----------反向传播算法,计算各层delta-----
        %输出层delta
        Err = out{layers-1}-train_ans;
        %Err_val =  val_ans - out_val{layers-1};
        delta{layers-1} = Err;
        %隐藏层delta
        for i=layers-2 : -1 : 1
            delta{i} = out{i} .* (1-out{i}) .* (delta{i+1} * w{i+1}');
        end
        %%-----------------------更新w----------------
        %隐藏层  adam
        for i = layers-1 : -1 : 2
            g=out{i-1}'*delta{i} ; %计算梯度
            m{i} = beta1 * m{i} + (1 - beta1) * g; %更新m
            v{i} = beta2 * v{i} + (1 - beta2) * (g .* g);%更新v
            m_=m{i}/(1-beta1); %偏差m更正
            v_=v{i}/(1-beta2);%偏差v更正
            w{i} = w{i} + step(i) *  (m_ ./ (sqrt(v_)+1e-8) + lambda(i)  * w{i});%更新w
            theta{i} = theta{i} + step(i)  / train_row * sum(delta{i},1) ;%更新b
        end
        %输入层->隐藏层 adam
        g=train'*delta{1} ; 
        m{1} = beta1 * m{1} + (1 - beta1) * g;
        v{1} = beta2 * v{1} + (1 - beta2) * (g .* g);
        m_=m{1}/(1-beta1);
        v_=v{1}/(1-beta2);
        w{1} = w{1} + step(1) *  (m_ ./ (sqrt(v_)+eps) + lambda(1)  * w{1});
        theta{1} = theta{1} + step(1)  / train_row * sum(delta{1},1) ;
        
        
        
    end
    close(h);
    avg_acc=(low_acc + mid_acc + hig_acc)/3;
    subplot(2,2,1)
    plot(1:iter,avg_acc);
    subplot(2,2,2)
    plot(1:iter,low_acc);
    subplot(2,2,3)
    plot(1:iter,mid_acc);
    subplot(2,2,4)
    plot(1:iter,hig_acc);
end


function [test_ans]=get_ans(test,w,theta,drop,layers)
    out_test = cell(1, layers - 1);
    out_test{1} = 1 ./ (1 + exp(-test *w{1} + theta{1}));
    %隐藏层
    for i=2 : layers-2
        out_test{i} = 1 ./ (1 + exp(-out_test{i - 1}   * w{i}  + theta{i}));
    end
    %输出层
    out_test{layers-1} = 1 ./ (1 +exp(out_test{layers-2} * w{layers-1} + theta{layers-1})) ;
    test_ans=out_test{layers-1};
%     test_ans(test_ans >= 0.5) = 1;
%     test_ans(test_ans < 0.5) = 0;
end