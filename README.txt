*****关于notebook源代码文件夹与python源代码文件夹*****

	两个文件夹中均包含两个文件夹，分别为ml_1m_dataset与ml_100k_dataset，分别是较大数据集与较小数据集。
	python文件夹与notebook文件夹源代码一致，一个为.py文件，另一个为完成工程时所用环境对应的.ipynb文件。

*****ml_100k_dataset文件*****

	codeRev_adv_trainxxpercent：基于余弦相似度公式的模型训练代码，其中数字代表该模型训练中训练集所占比例。
	euli_prove_sim：归一化后的基于欧几里得公式的推荐系统。
	euli_prove_sim：没有进行归一化（注释中含有错误归一化的部分算法）基于欧几里得公式的推荐系统。
	本数据集中的余弦相似度代码即实验指导书复现代码，故放在复现代码文件夹下。

*****ml_100k_dataset文件*****

	ml_1m_cosine:基于余弦相似度公式的推荐系统。
	ml_1m_cosine_xxpercent_train:基于余弦相似度公式的模型训练，数字代表训练集所占比例。
	ml_1m_eu:基于欧几里得公式的推荐系统。
	
*****数据集*****
	movieLens的ml_100k 及 ml_1m,暂未上传
