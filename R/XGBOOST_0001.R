library(data.table)
library(readr)
library(ranger)
library(xgboost)
library(caTools)
library(funModeling)
library(caret)

train <- read_csv("../Data/ItemInfo_train.csv")
test <- read_csv("../Data/ItemInfo_test.csv")
trainItem <- read_csv("../Data/ItemPairs_train.csv")
testItem <- read_csv("../Data/ItemPairs_test.csv")
trainItem <- data.table(trainItem)
testItem <- data.table(testItem)

train <- train[,c("itemID","categoryID","locationID","lat","lon","title","description","price")]
names(train) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1")
train <- data.table(train)
setkey(train,"itemID_1")
setkey(trainItem,"itemID_1")

trainItem <- merge(trainItem,train,all.x=TRUE)
names(train) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2")
setkey(train,"itemID_2")
setkey(trainItem,"itemID_2")

trainItem <- merge(trainItem,train,all.x=TRUE)

test <- test[,c("itemID","categoryID","locationID","lat","lon","title","description","price")]
names(test) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1")
test <- data.table(test)
setkey(test,"itemID_1")
setkey(testItem,"itemID_1")

testItem <- merge(testItem,test,all.x=TRUE)

names(test) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2")

setkey(test,"itemID_2")
setkey(testItem,"itemID_2")

testItem <- merge(testItem,test,all.x=TRUE)

trainItem$sameLat <- as.numeric(trainItem$lat1 == trainItem$lat2)
trainItem$sameLon <- as.numeric(trainItem$lon1 == trainItem$lon2)
trainItem$sameLoc <- as.numeric(trainItem$locationID1 == trainItem$locationID2)
trainItem$priceDiff <- (trainItem$price1 - trainItem$price2)
trainItem$sameT <- as.numeric(trainItem$title1 == trainItem$title2)
trainItem$sameD <- as.numeric(trainItem$description1 == trainItem$description2)


testItem$sameLat <- as.numeric(testItem$lat1 == testItem$lat2)
testItem$sameLon <- as.numeric(testItem$lon1 == testItem$lon2)
testItem$sameLoc <- as.numeric(testItem$locationID1 == testItem$locationID2)
testItem$priceDiff <- (testItem$price1 - testItem$price2)
testItem$sameT <- as.numeric(testItem$title1 == testItem$title2)
testItem$sameD <- as.numeric(testItem$description1 == testItem$description2)

trainItem[is.na(trainItem)] <- -9999
testItem[is.na(testItem)] <- -9999


features <- c("sameLat","sameLon","sameLoc","priceDiff","sameT","sameD")

trainx <- trainItem[,features,with=FALSE]
testx <- testItem[,features,with=FALSE]
response <- trainItem$isDuplicate

split <- createDataPartition(as.factor(response),p=0.9)$Resample1
model <- glm(response[split] ~. ,data=trainx[split,],family="binomial")
predglm <- predict(model,trainx[-split,],type="response")
colAUC(data.frame(predglm),response[-split])


dval <- xgb.DMatrix(data=data.matrix(trainx)[split,],label=response[split])
dtrain <- xgb.DMatrix(data=data.matrix(trainx)[-split,],label=response[-split])
watchlist <- list(val=dval,train=dtrain)

param <- list(
  objective = "binary:logistic",
  booster = "gbtree",
  eta                 = 0.05,
  max_depth           = 4,
  subsample           = 0.8,
  colsample_bytree    = 0.8,
  min_child_weight=10,
  nthread = 40
)

clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    nrounds             = 5000, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 10,
                    watchlist           = watchlist,
                    maximize            = TRUE,
                    print.every.n = 10,
                    eval_metric ="auc"
)

pred <- predict(clf,data.matrix(testx))
sub <- data.frame(id=testItem$id,probability=pred)
write.csv(sub,file="../LB/XGBOOST_0001.csv",row.names=FALSE)

