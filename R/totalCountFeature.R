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

train <- read_csv("../Data/ItemInfo_train.csv")
test <- read_csv("../Data/ItemInfo_test.csv")
trainItem <- read_csv("../Data/ItemPairs_train.csv")
testItem <- read_csv("../Data/ItemPairs_test.csv")
trainItemImages2 <- read_csv("../Data/trainItem_w_features_2.csv")
testItemImages2 <- read_csv("../Data/testItem_w_features_2.csv")

trainItem$trainFlag <- 1
testItem$trainFlag <- 0

trainItem$id <- 9999
testItem$isDuplicate <- 9999
testItem$generationMethod <- 9999

testItem <- testItem[,names(trainItem)]

trainItem <- data.table(rbind(trainItem,testItem))

train <- data.frame(rbind(train,test))

train <- train[,c("itemID","categoryID","locationID","lat","lon","title","description","price","images_array","metroID")]

names(train) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1","images_array1","metroID1")
train <- data.table(train)
setkey(train,itemID_1)
setkey(trainItem,itemID_1)
trainItem <- merge(trainItem,train,all.x=TRUE)


names(train) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2","images_array2","metroID2")
setkey(train,"itemID_2")
setkey(trainItem,"itemID_2")
trainItem <- merge(trainItem,train,all.x=TRUE)


trainItem <- trainItem[,countItem1:=.N,by=itemID_1]
trainItem <- trainItem[,countItem2:=.N,by=itemID_2]

trainItem$countItemSum <- trainItem$countItem1 + trainItem$countItem2

trainItem$sameLat <- as.numeric(trainItem$lat1 == trainItem$lat2)
trainItem$sameLon <- as.numeric(trainItem$lon1 == trainItem$lon2)
trainItem$sameLoc <- as.numeric(trainItem$locationID1 == trainItem$locationID2)

trainItem$ncharTitle1 <- nchar(trainItem$title1)
trainItem$ncharTitle2 <- nchar(trainItem$title2)

trainItem$ncharDesc1 <- nchar(trainItem$description1)
trainItem$ncharDesc2 <- nchar(trainItem$description2)

trainItem$samePrice <- as.numeric(trainItem$price1 == trainItem$price2)
trainItem$priceDiff <- pmax(trainItem$price1/trainItem$price2, trainItem$price2/trainItem$price1)
trainItem$priceMax <- pmax(trainItem$price1, trainItem$price2, na.rm=TRUE)
trainItem$priceMin <- pmin(trainItem$price1, trainItem$price2, na.rm=TRUE)
trainItem$priceDiff <- ifelse(is.na(trainItem$priceDiff), 0, trainItem$priceDiff)
trainItem$priceMin <- ifelse(is.na(trainItem$priceMin), 0, trainItem$priceMin)
trainItem$priceMax <- ifelse(is.na(trainItem$priceMax), 0, trainItem$priceMax)
trainItem <- trainItem[, price1Missing := 1*(is.na(price1))]
trainItem <- trainItem[, price2Missing := 1*(is.na(price2))]
trainItem <- trainItem[, onePriceMiss := 1*(price1Missing == 1 | price2Missing)]
trainItem <- trainItem[, twoPriceMiss := 1*(price1Missing == 1 & price2Missing)]



trainItem$sameT <- as.numeric(trainItem$title1 == trainItem$title2)
trainItem$sameD <- as.numeric(trainItem$description1 == trainItem$description2)
trainItem$sameM <- as.numeric(trainItem$metroID1 == trainItem$metroID2)

trainItem$distT1 <- stringdist(trainItem$title1, trainItem$title2, method = "jw")
trainItem$distT2 <- (stringdist(trainItem$title1, trainItem$title2, method = "lcs") / pmax(trainItem$ncharTitle1,trainItem$ncharTitle2,na.rm=TRUE))

trainItem$distD1 <- stringdist(trainItem$description1, trainItem$description2, method = "jw")
#trainItem$distD2 <- (stringdist(trainItem$description1, trainItem$description2, method = "lcs") / pmax(trainItem$ncharDesc1,trainItem$ncharDesc2,na.rm=TRUE))

trainItem$distT1 <- ifelse(is.na(trainItem$distT1), 0, trainItem$distT1)
trainItem$distT2 = ifelse(is.na(trainItem$distT2) | trainItem$distT2 == Inf, 0, trainItem$distT2) 

trainItem$distD1 <- ifelse(is.na(trainItem$distD1), 0, trainItem$distD1)

trainItem$title1StartsWithTitle2 <- as.numeric(substr(trainItem$title1, 1, nchar(trainItem$title2)) == trainItem$title2)
trainItem$title2StartsWithTitle1 <- as.numeric(substr(trainItem$title2, 1, nchar(trainItem$title1)) == trainItem$title1)

trainItem$titleCharDiff <- pmax(trainItem$ncharTitle1/trainItem$ncharTitle2, trainItem$ncharTitle2/trainItem$ncharTitle1)
trainItem$titleCharMin <- pmin(trainItem$ncharTitle1, trainItem$ncharTitle2, na.rm=TRUE)
trainItem$titleCharMax <- pmax(trainItem$ncharTitle1, trainItem$ncharTitle2, na.rm=TRUE)

trainItem$descriptionCharDiff = pmax(trainItem$ncharDesc1/trainItem$ncharDesc2, trainItem$ncharDesc2/trainItem$ncharDesc1)
trainItem$descriptionCharMin = pmin(trainItem$ncharDesc1, trainItem$ncharDesc2, na.rm=TRUE)
trainItem$descriptionCharMax = pmax(trainItem$ncharDesc1, trainItem$ncharDesc2, na.rm=TRUE)

trainItem <- trainItem[,countLat1:=.N,by=lat1]
trainItem <- trainItem[,countLat2:=.N,by=lat2]
trainItem <- trainItem[,countLon1:=.N,by=lon1]
trainItem <- trainItem[,countLon2:=.N,by=lon2]

trainItem$countLatSum <- trainItem$countLat1 + trainItem$countLat2
trainItem$countLonSum <- trainItem$countLon1 + trainItem$countLon2


#Features suggested by Oleksii
trainItem <- trainItem[, distanceAd := distHaversine(as.matrix(trainItem[, c("lon1", "lat1"), with = FALSE]),
                                                     as.matrix(trainItem[, c("lon2", "lat2"), with = FALSE]))]


trainItem <- trainItem[, isLocation := 1*(locationID1 == locationID2)]
trainItem <- trainItem[isLocation == 1, location := locationID1]
trainItem <- trainItem[isLocation == 0, location := -999]
trainItem <- trainItem[, metroID1Missing := 1*(is.na(metroID1))]
trainItem <- trainItem[, metroID2Missing := 1*(is.na(metroID2))]
trainItem <- trainItem[, oneMetroIDMiss := 1*(metroID1Missing == 1 | metroID2Missing)]
trainItem <- trainItem[, twoMetroIDMiss := 1*(metroID1Missing == 1 & metroID2Missing)]
trainItem <- trainItem[oneMetroIDMiss == 0, isMetroID := 1* (metroID1 == metroID2)]
trainItem <- trainItem[oneMetroIDMiss != 0, isMetroID := 0]
trainItem <- trainItem[isMetroID == 1, metroID := metroID1]
trainItem <- trainItem[isMetroID == 0, metroID := -999]


trainItemImages2$id <- 0
testItemImages2$isDuplicate <- 0
testItemImages2$generationMethod <- 0

trainItemImages2 <- data.table(rbind(trainItemImages2,testItemImages2))


setkey(trainItem,itemID_1,itemID_2)
setkey(trainItemImages2,itemID_1,itemID_2)

trainItem <- merge(trainItem,trainItemImages2,all.x=TRUE)

#trainItem[is.nan(trainItem)] <- NULL
#testItem[is.nan(testItem)] <- NULL

trainItem[is.na(trainItem)] <- -9999


featuresBasic  <-  c("categoryID1","countItem1","countItem2","countItemSum","sameLat","sameLon","sameLoc","ncharTitle1","ncharTitle2","ncharDesc1","ncharDesc2", 
                     "samePrice","priceDiff","priceMax","priceMin","price1Missing","price2Missing","onePriceMiss","twoPriceMiss","sameT","sameD",
                     "sameM","distT1","distT2","distD1","title1StartsWithTitle2","title2StartsWithTitle1","titleCharDiff","titleCharMin", "titleCharMax",
                     "descriptionCharDiff","descriptionCharMin","descriptionCharMax","distanceAd","isLocation","location","metroID1Missing","metroID2Missing", 
                     "oneMetroIDMiss","twoMetroIDMiss","isMetroID","metroID","countLat1","countLat2","countLon1","countLon2","countLatSum","countLonSum")

featuresImage  <-  c("dhash_count", "dhash_flip_count", "image1_count", "image2_count", "img_count_diff", "white_count1", "white_count2", "white_count_diff",
                     "top_200_count1", "top_200_count2", "top_200_count_diff", "cluster_x_15", "cluster_y_15", "same_cluster_15", "lt_40_sig_count", 
                     "lt_20_sig_count","sig_count_diff")


features <- c(featuresBasic,featuresImage)


trainx <- trainItem[trainFlag==1,features,with=FALSE]
testx <- trainItem[trainFlag==0,features,with=FALSE]
response <- trainItem$isDuplicate.x[trainItem$trainFlag==1]

split <- createDataPartition(as.factor(response),p=0.9)$Resample1

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


pred <- predict(clf,data.matrix(testx))
sub <- data.frame(id=trainItem$id.x[trainItem$trainFlag==0],probability=pred)
write.csv(sub,file="../LB/T_0027.csv",row.names=FALSE)



