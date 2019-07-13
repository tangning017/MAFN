#机器学习模型
 ### +监督学习
    -线性模型及二次模型
        -线性模型: 逻辑回归，Lasso, [Hi-LASSO](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8684195), 岭回归, Elastic-Net, [Minimax Concave Penalty(MCP)](https://arxiv.org/pdf/1002.4734.pdf), [SCAD](http://www.myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf), [thePrecisionLasso](https://github.com/HaohanWang/thePrecisionLasso), 线性判别分析(LDA), Robust Sparse Linear Discriminant Analysis(RSLDA)
        -二次模型: 二次判别分析(QDA)
    -支持向量机
        -SVC, NuSVC, LinearSVC, SVR, NuSVR, OneClassSVM
    -近邻算法
        -KNN(Brute Force, K-D Tree, Ball Tree), RadiusNeighbors, NearestCentroid, [K-RCC](https://content.iospress.com/articles/journal-of-intelligent-and-fuzzy-systems/ifs181336)
    -贝叶斯
        -Gaussian Naive Bayes, Multinomial Naive Bayes, Complement Naive Bayes, Bernoulli Naive Bayes
    -决策树
        -DecisionTree
        -Boosting: AdaBoost, GBDT, XGBoost, [CatBoost](https://github.com/catboost/catboost), LightGBM(https://github.com/Microsoft/LightGBM)
        -Bagging: RandomForest

 ### +无监督学习
    -流形学习（降维）
        -MDS, no-metric MDS, ISOMAP, [M-Isomap](https://ieeexplore.ieee.org/document/6413899), [S-Isomap++](https://arxiv.org/pdf/1710.06462.pdf), LLE, [LargeVis](https://github.com/lferry007/LargeVis), TSNE, PCA, kernel PCA, NCA(NeighborhoodComponentsAnalysis)
    -聚类
        -Kmeans, Kmeans++, [DSKmeans](https://www.sciencedirect.com/science/article/abs/pii/S0950705114002664), MeanShift, [BorderShift](https://link.springer.com/article/10.1007%2Fs10044-018-0709-0), Spectral clustering, Hierarchical clustering, DBSCAN, [DBSCAN++](http://proceedings.mlr.press/v97/jang19a/jang19a.pdf), OPTICS, Birch
