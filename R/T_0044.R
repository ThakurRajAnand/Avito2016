#Model 44
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

system.time(testItem <- readRDS("testItem.RDS"))
system.time(trainItem <- readRDS("trainItem.RDS"))
system.time(trainJSON <- readRDS("trainJSON.RDS"))
system.time(testJSON <- readRDS("testJSON.RDS"))

locationData <- fread("../Data/Location.csv")
categoryData <- fread("../Data/Category.csv")

names(locationData)[1] <- "location"
names(categoryData)[1] <- "categoryID1"

setkey(locationData,location)
setkey(categoryData,categoryID1)

setkey(trainItem,location)
setkey(testItem,location)
trainItem <- merge(trainItem,locationData,all.x=TRUE)
testItem <- merge(testItem,locationData,all.x=TRUE)

setkey(trainItem,categoryID1)
setkey(testItem,categoryID1)
trainItem <- merge(trainItem,categoryData,all.x=TRUE)
testItem <- merge(testItem,categoryData,all.x=TRUE)


oleksii_attrsJSON_features_train <- fread("../Data/oleksii_attrsJSON_features_train.csv")
oleksii_attrsJSON_features_test <- fread("../Data/oleksii_attrsJSON_features_test.csv")

oleksii_description_features_train <- fread("../Data/oleksii_description_features_train.csv")
oleksii_description_features_test <- fread("../Data/oleksii_description_features_test.csv")

oleksii_title_features_train <- fread("../Data/oleksii_title_features_train.csv")
oleksii_title_features_test <- fread("../Data/oleksii_title_features_test.csv")

setkey(trainItem,itemID_1,itemID_2)
setkey(testItem,itemID_1,itemID_2)

setkey(oleksii_attrsJSON_features_train,itemID_1,itemID_2)
setkey(oleksii_attrsJSON_features_test,itemID_1,itemID_2)

setkey(oleksii_description_features_train,itemID_1,itemID_2)
setkey(oleksii_description_features_test,itemID_1,itemID_2)

setkey(oleksii_title_features_train,itemID_1,itemID_2)
setkey(oleksii_title_features_test,itemID_1,itemID_2)

trainItem <- merge(trainItem,oleksii_title_features_train,all.x=TRUE)
testItem <- merge(testItem,oleksii_title_features_test,all.x=TRUE)

setkey(trainItem,itemID_1,itemID_2)
setkey(testItem,itemID_1,itemID_2)

trainItem <- merge(trainItem,oleksii_description_features_train,all.x=TRUE)
testItem <- merge(testItem,oleksii_description_features_test,all.x=TRUE)

setkey(trainItem,itemID_1,itemID_2)
setkey(testItem,itemID_1,itemID_2)

trainItem <- merge(trainItem,oleksii_attrsJSON_features_train,all.x=TRUE)
testItem <- merge(testItem,oleksii_attrsJSON_features_test,all.x=TRUE)



# "countItem1","countItem2","countItemSum","sameLoc",

featuresBasic  <-  c("categoryID1","countItemTotal1","countItemTotal2","countItemTotalSum","sameLat","sameLon","ncharTitle1","ncharTitle2","ncharDesc1","ncharDesc2", 
                     "samePrice","priceDiff","priceMax","priceMin","price1Missing","price2Missing","onePriceMiss","twoPriceMiss","sameT","sameD",
                     "sameM","distT1","distT2","distD1","title1StartsWithTitle2","title2StartsWithTitle1","titleCharDiff","titleCharMin", "titleCharMax",
                     "descriptionCharDiff","descriptionCharMin","descriptionCharMax","distanceAd","isLocation","location","metroID1Missing","metroID2Missing", 
                     "oneMetroIDMiss","twoMetroIDMiss","isMetroID","metroID","countLat1","countLat2","countLon1","countLon2","countLatSum","countLonSum",
                     "oneAttrJsonMiss","twoAttrJsonMiss","oneImagesArrayMiss","twoImagesArrayMiss","network_size","parentCategoryID","regionID")

featuresImage  <-  c("dhash_count", "dhash_flip_count", "image1_count", "image2_count", "img_count_diff", "white_count1", "white_count2", "white_count_diff",
                     "top_200_count1", "top_200_count2", "top_200_count_diff", "cluster_x_15", "cluster_y_15", "same_cluster_15", "lt_40_sig_count", 
                     "lt_20_sig_count","sig_count_diff")


featuresOleksii <- c("t_jaccard_dist","t_sum_equal","t_cosine_sim","t_dice_dist","t_manhattan_distance","t_hamming_distance","t_tfidf_cosine",
                     "d_jaccard_dist","d_sum_equal","d_cosine_sim","d_dice_dist","d_manhattan_distance","d_hamming_distance","d_tfidf_cosine",
                     "json_jaccard_dist","json_sum_equal","json_cosine_sim","json_dice_dist","json_manhattan_distance","json_hamming_distance","json_tfidf_cosine")



featuresJSON <- intersect(names(trainJSON)[4:ncol(trainJSON)],names(testJSON)[4:ncol(testJSON)])

features <- c(featuresBasic,featuresImage,featuresJSON,featuresOleksii)

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
                    early.stop.round    = 40,
                    watchlist           = watchlist,
                    maximize            = TRUE,
                    print.every.n = 10,
                    eval_metric ="auc"
)
gc()

pred <- predict(clf,testx[,common])
sub <- data.frame(id=id,probability=pred)
write.csv(sub,file="../LB/T_0044.csv",row.names=FALSE)


