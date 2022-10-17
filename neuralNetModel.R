data(banknote, package="mclust")
banknoteTib <- as_tibble(banknote)
banknoteTib
summary(banknoteTib)
ggplot(banknoteTib, aes(Length, Diagonal, col = Status)) + geom_point() + theme_bw() 
ggplot(banknoteTib, aes(Right, Left, col = Status)) + geom_point() + theme_bw()
ggplot(banknoteTib, aes(Top, Bottom, col = Status)) + geom_point() + theme_bw()
banknoteTask <- makeClassifTask(data=banknote, target="Status")
banknoteTask <- normalizeFeatures(banknoteTask, method="standardize")
banknoteTask
neuralnet <- makeLearner("classif.neuralnet", par.vals = list("hidden"=2, "learningrate"=0.001, "rep"=6))
neuralnetModel <- train(neuralnet, banknoteTask)
neuralnetPredict <- predict(neuralnetModel, newdata=banknoteTib)
performance(neuralnetPredict, measures=list(mmce, acc))

holdout <- makeResampleDesc(method="Holdout", split=2/3, stratify=TRUE)
holdoutCV <- resample(learner=neuralnet, task=banknoteTask, resampling=holdout, measures=list(mmce,acc))
holdoutCV$aggr

kfold <- makeResampleDesc(method="RepCV", folds=10, reps=50, stratify = TRUE)
kfoldCV <- resample(learner=neuralnet, task=banknoteTask, resampling=kfold, measures=list(mmce, acc))
kfoldCV$aggr

LOO <- makeResampleDesc(method="LOO")
LOOCV <- resample(learner=neuralnet, task=banknoteTask, resampling=LOO, measures=list(mmce, acc))
LOOCV$aggr
