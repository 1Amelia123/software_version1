一、需求规格与环境配置
需要python3环境，同时能够满足源代码中所需要的第三方库包括torch，sklearn，numpy，pandas，matplotlib，pickle，tqdm，seaborn，joblib，pylab。
二、源代码文件中的介绍与操作说明：
源代码中共有三个文件夹，分别命名为“任务123”，“任务4”，“任务5”。
任务123文件夹下中的data为原始数据，在复现此部分任务时，需要先执行data_process文件，然后再执行text_1&2文件，即可得到任务1，2的结果。最后执行text3文件，即可得到任务3的结果。
任务4文件夹下包含三个文件夹，分别命名为“code and model”，“origin_dataset“和“processed_dataset”，其中“origin_dataset”中包含的是原始数据集，“processed_dataset”是经过”code and model“中的”combining“和”pre_train“得到的数据集，最后用于模型训练的数据集是data_1中的数据。运行”code and model“中的“LSTM预测模型”文件即可得到“企业电力营销模型.mkl”。或者可以直接加载此模型进行预测。
任务5文件夹下中的train为原始数据，在复现此部分任务时，只需要运行“按月聚类的代码”文件即可得到“电力用户集群模型.mkl”，进行聚类分析，或者可以直接加载此模型得到对数据的类别进行判定。如果想要判断其中每个季度的类型，则可以运行按季度聚类的代码，或者直接加载对应季度的模型，进行聚类分析即可。