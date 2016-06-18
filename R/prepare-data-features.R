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

source("utils.R")

train <- read_csv("../Data/ItemInfo_train.csv")
test <- read_csv("../Data/ItemInfo_test.csv")
trainItem <- read_csv("../Data/ItemPairs_train.csv")
testItem <- read_csv("../Data/ItemPairs_test.csv")
imageHashes <- read_csv("../Data/image_features_dhash.csv")
imageSimTrain <- read_csv("../features//vcc_euc_train.csv")
imageSimTest <- read_csv("../features//vcc_euc_test.csv")

names(imageSimTrain) <- c("itemID_1", "euc_dist_25")
names(imageSimTest) <- c("itemID_1", "euc_dist_25")

imageSimTrain[is.na(imageSimTrain)] <- -9999
imageSimTest[is.na(imageSimTest)] <- -9999

imageSimTrain <- data.table(imageSimTrain)
imageSimTest <- data.table(imageSimTest)

imageHashes <- data.table(imageHashes)
setkey(imageHashes,image)
trainItem <- data.table(trainItem)
testItem <- data.table(testItem)

train <- train[,c("itemID","categoryID","locationID","lat","lon","title","description","price","images_array","metroID")]
names(train) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1","images_array1","metroID1")
train <- data.table(train)

train <- merge_with_key(train, imageSimTrain, "itemID_1")
trainItem <- merge_with_key(trainItem, train, "itemID_1")

names(train) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2","images_array2","metroID2","euc_dist_25")

trainItem <- merge_with_key(trainItem, trian, "itemID_2")

test <- test[,c("itemID","categoryID","locationID","lat","lon","title","description","price","images_array","metroID")]
names(test) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1","images_array1","metroID1")

test <- data.table(test)

test <- merge_with_key(test, imageSimTest, "itemID_1")
testItem <- merge_with_key(testItem, test, "itemID_1")

names(test) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2","images_array2","metroID2","euc_dist_25")

testItem <- merge_with_key(testItem, test, "itemID_2")

trainItem <- as.data.table(trainItem)

# =========

add_features <- function(dt) {
    dt[, ':' (sameLat = as.numeric(lat1 == lat2),
                     sameLon = as.numeric(lon1 == lon2),
                     sameLoc = as.numeric(locationID1 == locationID2),
                     ncharTitle1 = nchar(title1),
                     ncharTitle2 = nchar(title2),
                     ncharDesc1 = nchar(description1),
                     ncharDesc2 = nchar(description2),
                     samePrice = as.numeric(price1 == price2),
                     priceDiff = abs(price1 - price2),
                     priceRatio = pmax(price1/price2, price2/price1),
                     sameT = as.numeric(title1 == title2),
                     sameD = as.numeric(description1 == description2),
                     sameM = as.numeric(metroID1 == metroID2),
                     distT1 = stringdist(title1, title2, method = "jw"),
                     distD1 = stringdist(description1, description2, method = "jw"),
                     titleCharDiff = pmax(ncharTitle1/ncharTitle2, ncharTitle2/ncharTitle1),
                     descriptionCharDiff = pmax(ncharDesc1/ncharDesc2, ncharDesc2/ncharDesc1),
                     distance = sqrt((lat1 - lat2)^2+(lon1 - lon2)^2),
                     index1 = 1:.N)]
    
    dt[, ':' (image_count1 = length(str_split(images_array1, ",")[[1]]),
              image_count2 = length(str_split(images_array2, ",")[[1]])), by = index1]
    
    lldf1 <- get_dt_with_hashes(dt[['images_array1']], images_array = imageHashes)
    lldf2 <- get_dt_with_hashes(dt[['images_array2']], images_array = imageHashes)
    
    lldf <- rbind(lldf1,lldf2)
    lldf <- sqldf("select index1, dhash, count(*) as Freq From lldf Group By index1 , dhash")
    
    lldf <- subset(lldf,Freq>=2)
    
    lldf <- sqldf("select index1, count(*) as commonHash From lldf Group By index1")
    
    lldf <- data.table(lldf)
    setkey(lldf,index1)
    setkey(dt,index1)
    dt <- merge(dt,lldf,all.x=TRUE)
    dt$commonHash[is.na(dt$commonHash)] <- 0
    
    dt[, image_count1Ratio := commonHash / image_count1, by = index1]
    dt[, image_count2Ratio := commonHash / image_count2, by = index1]
    return(dt)
}

# =========

trainItem <- add_features(trainItem)
testItem <- add_features(testItem)

trainItem[is.nan(trainItem)] <- NULL
testItem[is.nan(testItem)] <- NULL

trainItem[is.na(trainItem)] <- -9999
testItem[is.na(testItem)] <- -9999

features <- c("sameLat","sameLon","sameLoc","ncharTitle1","ncharTitle2","ncharDesc1","ncharDesc2","samePrice","priceDiff","priceMax","priceMin","sameT","sameD",
              "sameM","distT1","distT2","distD1", "title1StartsWithTitle2","title2StartsWithTitle1","titleCharDiff","titleCharMin", "titleCharMax","descriptionCharDiff","descriptionCharMin",
              "descriptionCharMax","distance","image_count1","image_count2","commonHash","image_count1Ratio","image_count2Ratio","countItem1","countItem2","countItemSum","countLat1",
              "countLat2","countLon1","countLon2","countLatSum","countLonSum","euc_dist_25.x","euc_dist_25.y")

trainx <- trainItem[,features,with=FALSE]
testx <- testItem[,features,with=FALSE]
response <- trainItem$isDuplicate

set.seed(4444)
split <- createFolds(as.factor(response),10)
score <- c()
for(i in 1:10){
  dval <- xgb.DMatrix(data=data.matrix(trainx)[-split[[i]],],label=response[-split[[i]]])
  dtrain <- xgb.DMatrix(data=data.matrix(trainx)[split[[i]],],label=response[split[[i]]])
  watchlist <- list(val=dval,train=dtrain)
  
  param <- list(
    objective = "binary:logistic",
    booster = "gbtree",
    eta                 = 0.05,
    max_depth           = 15,
    subsample           = 0.8,
    colsample_bytree    = 0.4,
    min_child_weight=10,
    nthread = 40,
    gamma=6,
    lambda=0.001
  )
  
  clf <- xgb.train(   params              = param,
                      data                = dtrain,
                      nrounds             = 5000, #300, #280, #125, #250, # changed from 300
                      verbose             = 0,
                      early.stop.round    = 50,
                      watchlist           = watchlist,
                      maximize            = TRUE,
                      print.every.n = 20,
                      eval_metric ="auc"
  )
  
  if(i==1) pred <- predict(clf, data.matrix(testx))^2 else pred <- pred + predict(clf,data.matrix(testx))^2
  score <- c(score,clf$bestScore)
  print(mean(score))
  print(i)
}

sub <- data.frame(id=testItem$id,probability=sqrt(pred/i))
write.csv(sub,file="../LB/T_0013.csv",row.names=FALSE)
