#Model 38
library(data.table)
library(readr)
library(ranger)
library(xgboost)
library(caTools)
library(funModeling)
library(caret)
library(stringr)
library(sqldf)
library(stringdist)
library(geosphere)
library(fossil)
library(rARPACK)
library(GPfit)
library(rBayesianOptimization)
library(feather)
library(Matrix)

is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))


#system.time(testItem <- read_feather("testItem.feather"))
#system.time(trainItem <- read_feather("trainItem.feather"))
#system.time(trainJSON <- read_feather("trainJSON.feather"))
#system.time(testJSON <- read_feather("testJSON.feather"))

system.time(testItem <- readRDS("testItem.RDS"))
system.time(trainItem <- readRDS("trainItem.RDS"))
system.time(trainJSON <- readRDS("trainJSON.RDS"))
system.time(testJSON <- readRDS("testJSON.RDS"))


# "countItem1","countItem2","countItemSum","sameLoc",

featuresBasic  <-  c("categoryID1","countItemTotal1","countItemTotal2","countItemTotalSum","sameLat","sameLon","ncharTitle1","ncharTitle2","ncharDesc1","ncharDesc2", 
                     "samePrice","priceDiff","priceMax","priceMin","price1Missing","price2Missing","onePriceMiss","twoPriceMiss","sameT","sameD",
                     "sameM","distT1","distT2","distD1","title1StartsWithTitle2","title2StartsWithTitle1","titleCharDiff","titleCharMin", "titleCharMax",
                     "descriptionCharDiff","descriptionCharMin","descriptionCharMax","distanceAd","isLocation","location","metroID1Missing","metroID2Missing", 
                     "oneMetroIDMiss","twoMetroIDMiss","isMetroID","metroID","countLat1","countLat2","countLon1","countLon2","countLatSum","countLonSum",
                     "oneAttrJsonMiss","twoAttrJsonMiss","oneImagesArrayMiss","twoImagesArrayMiss","network_size")

featuresImage  <-  c("dhash_count", "dhash_flip_count", "image1_count", "image2_count", "img_count_diff", "white_count1", "white_count2", "white_count_diff",
                     "top_200_count1", "top_200_count2", "top_200_count_diff", "cluster_x_15", "cluster_y_15", "same_cluster_15", "lt_40_sig_count", 
                     "lt_20_sig_count","sig_count_diff")


featuresJSON <- intersect(names(trainJSON)[4:ncol(trainJSON)],names(testJSON)[4:ncol(testJSON)])

features <- c(featuresBasic,featuresImage,featuresJSON)


trainx <- data.frame(trainItem[,features,with=FALSE])
testx <- data.frame(testItem[,features,with=FALSE])
response <- trainItem$isDuplicate.x

trainx[is.nan(trainx)] <- -9999
testx[is.nan(testx)] <- -9999

trainx[is.na(trainx)] <- -9999
testx[is.na(testx)] <- -9999

trainx <- Filter(function(x)(length(unique(x))>1), trainx)
testx <- Filter(function(x)(length(unique(x))>1), testx)

id <- testItem$id.x

rm(testItem)
rm(trainItem)
rm(trainJSON)
rm(testJSON)
gc()

system.time(trainx <- sparse.model.matrix(~.,data=trainx))

system.time(testx <- sparse.model.matrix(~.,data=testx))

common <- intersect(colnames(trainx),colnames(testx))

split <- createDataPartition(as.factor(response),p=0.9)$Resample1

trainx1 <- trainx[split,common]
valx1 <- trainx[-split,common]

gc()

dval <- xgb.DMatrix(data=valx1,label=response[-split])
dtrain <- xgb.DMatrix(data=trainx1,label=response[split])
watchlist <- list(val=dval,train=dtrain)

gc()

param <- list(
  objective = "binary:logistic",
  booster = "gbtree",
  eta                 = 0.05,
  max_depth           = 9,
  subsample           = 0.9,
  colsample_bytree    = 0.3,
  min_child_weight=5,
  nthread = 40,
  gamma=6,
  lambda=0.001
)

clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    nrounds             = 40000, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 20,
                    watchlist           = watchlist,
                    maximize            = TRUE,
                    print.every.n = 10,
                    eval_metric ="auc"
)

pred <- predict(clf,testx[,common])
sub <- data.frame(id=id,probability=pred)
write.csv(sub,file="../LB/T_0040.csv",row.names=FALSE)

