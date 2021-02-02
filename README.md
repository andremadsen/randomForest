randomForest [R code]

**SUPERVISED MACHINE LEARNING: Classify female puberty status by machine learning**

In this project I apply the 'random forest' machine learning algorithm to the Bergen Growth Study 2 [vekststudien.no] female dataset, 
in order to predict Tanner breast stage from the endocrine profile





**To plot the 'prevailing' model decision tree:** 
options(repos='http://cran.rstudio.org')
have.packages <- installed.packages()
cran.packages <- c('devtools','plotrix','randomForest','tree')
to.install <- setdiff(cran.packages, have.packages[,1])
if(length(to.install)>0) install.packages(to.install)

library(devtools)
if(!('reprtree' %in% installed.packages())){
  install_github('araastat/reprtree')
}
for(p in c(cran.packages, 'reprtree')) eval(substitute(library(pkg), list(pkg=p)))
Then go ahead and make the model and tree:

library(randomForest)
library(reprtree)

model <- randomForest(Species ~ ., data=iris, importance=TRUE, ntree=500, mtry = 2, do.trace=100)

reprtree:::plot.getTree(model)
