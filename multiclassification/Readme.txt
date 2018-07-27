先运行 dealData_TFIDF.py  再运行 dealData.m  最后multiclassification.m  结果输出为66_v1.csv 再运行 num2str.py 将输出结果转化为 字符串 保存在final66_v1.csv

会有略微误差

dealData_TFIDF.py 为数据集处理，请使用py3运行。
会在当前目录下生产 数据集的TF以及TFIDF矩阵  
（需要空间30G，时间6-7小时）

Hash.csv 是已经准备好的单词表，用于加速处理

dealData.m 是将数据进行PCA处理的代码 内部只做了PCA降维  
运行完成后生产train_test.mat 
为处理好的数据集
（运行要求 额外的30G空间 + 50G内存 时间2小时）


GPUmulticlassification.m 为训练部分
（运行要求 电脑带有NVIDIA 的GPU（GTX 1050及以上同等性能） 显存要求超过 2G 最好有CUDA驱动 时间4小时左右（GTX 970下）CPU根本跑不动）


train_test.mat 文件提供下载地址 https://pan.baidu.com/s/1pMdKj6F  密码g64r
			 校园网 http://d.chonor.cn/pro/train_test.mat

代码只在MATLAB2016a的以上版本运行，不保证在在低版本可用
如果代码无法运行可以联系我们进行远程桌面的操作


v2版本
增加一个 CPU运行的multiclassification.m文件
（运行时间为 24小时左右）