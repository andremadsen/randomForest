#######################
#R code: endocrine profiling with outcome variable = female puberty status

#In this project I apply the 'random forest' machine learning algorithm 
#to the Bergen Growth Study 2 [vekststudien.no] female dataset, in order to predict Tanner breast stage from the endocrine profile
#SUPERVISED MACHINE LEARNING
#=======================================================================================

install.packages(c("randomForest", "partykit", "ggplot2", "pROC", "caret", "arulesViz", "Rgraphviz", "sp","rpart"))



#=================================================================
# Set up dataframe from the bigger dataframe
#=================================================================

#Define Outcome variable in main dataframe
main$Outcome <- main$Tanner_B <- ifelse(main$Tanner_B == 1, "Prepubertal", 
                                             ifelse(main$Tanner_B >= 2, "Pubertal", NA))

#Dataframe excision from main dataframe and define Outcome variable
Data <- data.frame(main$Outcome, main$V1, main$V2, main$V3, main$V4, main$V5)
colnames(Data) <- c("Outcome","Hormone1","Hormone2","Hormone3","Hormone4", "Hormone5")

Data$Outcome          #Outcome variable e.g. ordinal Tanner puberty stage
Data$Hormone1         #feature/dependent variable#1 e.g. hormone, nmol/L
Data$Hormone2         #feature/dependent variable#2 e.g. hormone, nmol/L   
Data$Hormone3         #feature/dependent variable#3 e.g. hormone, nmol/L
Data$Hormone4         #feature/dependent variable#4 e.g. hormone, nmol/L
Data$Hormone5         #feature/dependent variable#5 e.g. hormone, IU/L

#Remove observations with outcome variable NAs
Data2 <- Data[complete.cases(Data$Outcome),]

#Annotate variables correctly
str(Data2)
Data2$Outcome <- as.factor(Data2$Outcome)
Data2$Hormone1 <- as.numeric(Data2$Hormone1)

#Impute dependent variable NAs (if applicable)
Data.imputed <- rfImpute(Outcome ~ ., data=Data2, iter=6)

#Partition train/test dataframes
library(caret)
indexes <- createDataPartition(Data.imputed$Outcome,
                               times = 1,
                               p = 0.75,
                               list = FALSE)

train.MLdata <- Data.imputed[indexes,]
test.MLdata <- Data.imputed[-indexes,]


#Optimize randomForest 'mtry' hyperparameter, set 'ntree' to an odd number 501-2001
oob.values <- vector(length=10)
for(i in 1:10) {
    temp.model <- randomForest(Outcome ~ .,data=train.MLdata, mtry=i, ntree=1001)
    oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}

#USE THE LOWEST-VALUE 'mtry' #1-10 printed out, e.g. mtry=3
oob.values

#Generate the randomForest model
library(randomForest)
RFmodel <- randomForest(Outcome ~ ., data=train.MLdata, ntree=1001, mtry=3, proximity=TRUE)

#Inspect model and view stats
RFmodel
result <- data.frame(test.MLdata$Outcome, predict(RFmodel, test.MLdata[,2:6], type="response"))
head(result)
plot(result)


#Error plots
plot(RFmodel, log="y")
varImpPlot(RFmodel)
MDSplot(RFmodel, train.MLdata$Outcome)
getTree(randomForest(Outcome ~ ., data=train.MLdata, ntree=1001), 3, labelVar=TRUE)

oob.error.data <- data.frame(
    Trees=rep(1:nrow(RFmodel$err.rate), times=3),
    Type=rep(c("OOB","0","1"), each=nrow(RFmodel$err.rate)),
    Error=c(RFmodel$err.rate[,"OOB"],
            RFmodel$err.rate[,"0"],
            RFmodel$err.rate[,"1"]))

head(oob.error.data)

library(ggplot2)
ggplot(data=oob.error.data, aes(x=Trees, y=Error)) + geom_line(aes(color=Type))

#Plot decision tree
library(partykit)
library(arulesViz)
library(Rgraphviz)
library(sp)
library(rpart)

c <- ctree(Outcome ~ ., data=train.MLdata) 

plot(c, type="simple") #extended decision tree

plot(c, type="simple", inner_panel=node_inner(c,
                                              abbreviate = FALSE,      # short variable names
                                              pval = TRUE,             # no p-values
                                              id = FALSE),             # no id of node
     terminal_panel=node_terminal(c,
                                  abbreviate = TRUE,
                                  digits = 1,                          # few digits on numbers
                                  fill = c("grey"),                    # make box white not grey
                                  id = FALSE))

#Save image
png("randomForestDecisionTree.png", res=80, height=900, width=1200)
dev.off()


#Evaluate randomForest overall model classification performance (training dataset) by ROC curve
RFmodel <- randomForest(Outcome ~ ., data=train.MLdata, ntree=1001, mtry=3, proximity=TRUE)

library(pROC)
rf.roc <- roc(train.MLdata$Outcome, RFmodel$votes[,2])
length(rf.roc$controls)
plot(rf.roc)
auc(rf.roc)
coords(rf.roc, "best", transpose=TRUE, ret=c("threshold", "ppv", "npv", "sens", "spec", "accuracy"))


#Evaluate randomForest model classification performance by ROC curve (test dataset) by ROC CURVE
RFmodel <- randomForest(Outcome ~ ., data=train.MLdata, ntree=1001, mtry=3, proximity=TRUE)
result <- data.frame(test.MLdata$Outcome, predict(RFmodel, test.MLdata[,2:6], type="response"),
                     predict(RFmodel, test.MLdata[,2:6], type="prob"))

library(pROC)
rf.roc <- roc(result$test.MLdata.Outcome, result$X1)
plot(rf.roc)
auc(rf.roc)
coords(rf.roc, "best", transpose=TRUE, ret=c("threshold", "ppv", "npv", "sens", "spec", "accuracy"))


#Confusion matrix for ML test dataset
head(result)
str(result)
library(caret)
confusionMatrix(result[,2], result[,1])
