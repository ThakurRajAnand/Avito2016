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
imageHashes <- read_csv("../Data/image_features_dhash.csv")
imageHashes <- data.table(imageHashes)
setkey(imageHashes,image)
trainItem <- data.table(trainItem)
testItem <- data.table(testItem)
#hashID <- imageHashes$dhash
#imageID <- imageHashes$image
#lookup <- hashID
#names(lookup) <- imageID

train <- train[,c("itemID","categoryID","locationID","lat","lon","title","description","price","images_array","metroID")]
names(train) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1","images_array1","metroID1")
train <- data.table(train)
setkey(train,"itemID_1")
setkey(trainItem,"itemID_1")

trainItem <- merge(trainItem,train,all.x=TRUE)
names(train) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2","images_array2","metroID2")
setkey(train,"itemID_2")
setkey(trainItem,"itemID_2")

trainItem <- merge(trainItem,train,all.x=TRUE)

test <- test[,c("itemID","categoryID","locationID","lat","lon","title","description","price","images_array","metroID")]
names(test) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1","images_array1","metroID1")
test <- data.table(test)
setkey(test,"itemID_1")
setkey(testItem,"itemID_1")

testItem <- merge(testItem,test,all.x=TRUE)

names(test) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2","images_array2","metroID2")
setkey(test,"itemID_2")
setkey(testItem,"itemID_2")

testItem <- merge(testItem,test,all.x=TRUE)

trainItem$sameLat <- as.numeric(trainItem$lat1 == trainItem$lat2)
trainItem$sameLon <- as.numeric(trainItem$lon1 == trainItem$lon2)
trainItem$sameLoc <- as.numeric(trainItem$locationID1 == trainItem$locationID2)
trainItem$priceDiff <- abs(trainItem$price1 - trainItem$price2)
trainItem$sameT <- as.numeric(trainItem$title1 == trainItem$title2)
trainItem$sameD <- as.numeric(trainItem$description1 == trainItem$description2)
trainItem$sameM <- as.numeric(trainItem$metroID1 == trainItem$metroID2)
trainItem$distT <- stringdist(trainItem$title1, trainItem$title2, method = "jw")
trainItem$distD <- stringdist(trainItem$description1, trainItem$description2, method = "jw")
trainItem$index1 <- 1:nrow(trainItem)
trainItem <- trainItem[,image_count1:=length(str_split(images_array1,",")[[1]]),by=index1]
trainItem <- trainItem[,image_count2:=length(str_split(images_array2,",")[[1]]),by=index1]
trainItem <- trainItem[,llDist:=earth.dist(rbind(c(lon1, lat1), c(lon2, lat2))),by=index1]
trainItem <- trainItem[,priceRatio:=abs(price1-price2)/min(c(price1,price2)),by=index1]


ll1 <- str_split(gsub(" ","",trainItem$images_array1),",")

lldf1 <- data.frame(image = unlist(ll1),
                    index1 = rep(seq_along(ll1), lapply(ll1, length)))

lldf1$image <- as.numeric(as.character(lldf1$image))
lldf1 <- data.table(lldf1)
setkey(lldf1,image)
dim(lldf1)
lldf1 <- data.frame(merge(lldf1,imageHashes,all.x=TRUE))
lldf1 <- lldf1[order(lldf1$index1),]

ll2 <- str_split(gsub(" ","",trainItem$images_array2),",")

lldf2 <- data.frame(image = unlist(ll2),
                    index1 = rep(seq_along(ll2), lapply(ll2, length)))


lldf2$image <- as.numeric(as.character(lldf2$image))
lldf2 <- data.table(lldf2)
setkey(lldf2,image)
dim(lldf2)
lldf2 <- data.frame(merge(lldf2,imageHashes,all.x=TRUE))
lldf2 <- lldf2[order(lldf2$index1),]

lldf <- rbind(lldf1,lldf2)
lldf <- sqldf("select index1, dhash, count(*) as Freq From lldf Group By index1 , dhash")

lldf <- subset(lldf,Freq>=2)

lldf <- sqldf("select index1, count(*) as commonHash From lldf Group By index1")

lldf <- data.table(lldf)
setkey(lldf,index1)
setkey(trainItem,index1)
trainItem <- merge(trainItem,lldf,all.x=TRUE)
trainItem$commonHash[is.na(trainItem$commonHash)] <- 0

trainItem <- trainItem[,image_count1Ratio:=commonHash/image_count1,by=index1]
trainItem <- trainItem[,image_count2Ratio:=commonHash/image_count2,by=index1]


testItem$sameLat <- as.numeric(testItem$lat1 == testItem$lat2)
testItem$sameLon <- as.numeric(testItem$lon1 == testItem$lon2)
testItem$sameLoc <- as.numeric(testItem$locationID1 == testItem$locationID2)
testItem$priceDiff <- abs(testItem$price1 - testItem$price2)
testItem$sameT <- as.numeric(testItem$title1 == testItem$title2)
testItem$sameD <- as.numeric(testItem$description1 == testItem$description2)
testItem$sameM <- as.numeric(testItem$metroID1 == testItem$metroID2)
testItem$distT <- stringdist(testItem$title1, testItem$title2, method = "jw")
testItem$distD <- stringdist(testItem$description1, testItem$description2, method = "jw")
testItem$index1 <- 1:nrow(testItem)
testItem <- testItem[,image_count1:=length(str_split(images_array1,",")[[1]]),by=index1]
testItem <- testItem[,image_count2:=length(str_split(images_array2,",")[[1]]),by=index1]
testItem <- testItem[,llDist:=earth.dist(rbind(c(lon1, lat1), c(lon2, lat2))),by=index1]

testItem <- testItem[,priceRatio:=abs(price1-price2)/min(c(price1,price2)),by=index1]


ll1 <- str_split(gsub(" ","",testItem$images_array1),",")

lldf1 <- data.frame(image = unlist(ll1),
                    index1 = rep(seq_along(ll1), lapply(ll1, length)))

lldf1$image <- as.numeric(as.character(lldf1$image))
lldf1 <- data.table(lldf1)
setkey(lldf1,image)
dim(lldf1)
lldf1 <- data.frame(merge(lldf1,imageHashes,all.x=TRUE))
lldf1 <- lldf1[order(lldf1$index1),]

ll2 <- str_split(gsub(" ","",testItem$images_array2),",")

lldf2 <- data.frame(image = unlist(ll2),
                    index1 = rep(seq_along(ll2), lapply(ll2, length)))


lldf2$image <- as.numeric(as.character(lldf2$image))
lldf2 <- data.table(lldf2)
setkey(lldf2,image)
dim(lldf2)
lldf2 <- data.frame(merge(lldf2,imageHashes,all.x=TRUE))
lldf2 <- lldf2[order(lldf2$index1),]

lldf <- rbind(lldf1,lldf2)
lldf <- sqldf("select index1, dhash, count(*) as Freq From lldf Group By index1 , dhash")

lldf <- subset(lldf,Freq>=2)

lldf <- sqldf("select index1, count(*) as commonHash From lldf Group By index1")

lldf <- data.table(lldf)
setkey(lldf,index1)
setkey(testItem,index1)
testItem <- merge(testItem,lldf,all.x=TRUE) 
testItem$commonHash[is.na(testItem$commonHash)] <- 0

testItem <- testItem[,image_count1Ratio:=commonHash/image_count1,by=index1]
testItem <- testItem[,image_count2Ratio:=commonHash/image_count2,by=index1]

trainItem[is.nan(trainItem)] <- NULL
testItem[is.nan(testItem)] <- NULL

trainItem[is.na(trainItem)] <- -9999
testItem[is.na(testItem)] <- -9999

features <- c("sameLat","sameLon","sameLoc","priceDiff","sameT","sameD","sameM","distT","distD","commonHash","llDist","priceRatio","image_count1Ratio","image_count2Ratio")

trainx <- trainItem[,features,with=FALSE]
testx <- testItem[,features,with=FALSE]
response <- trainItem$isDuplicate

#split <- createDataPartition(as.factor(response),p=0.9)$Resample1
#model <- glm(response[split] ~. ,data=trainx[split,],family="binomial")
#predglm <- predict(model,trainx[-split,],type="response")
#colAUC(data.frame(predglm),response[-split])


split <- createFolds(as.factor(response),10)
score <- c()
for(i in 1:10){
  dval <- xgb.DMatrix(data=data.matrix(trainx)[-split[[i]],],label=response[-split[[i]]])
  dtrain <- xgb.DMatrix(data=data.matrix(trainx)[split[[i]],],label=response[split[[i]]])
  watchlist <- list(val=dval,train=dtrain)
  
  param <- list(
    objective = "binary:logistic",
    booster = "gbtree",
    eta                 = 0.025,
    max_depth           = 10,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    min_child_weight=5,
    nthread = 40
  )
  
  clf <- xgb.train(   params              = param,
                      data                = dtrain,
                      nrounds             = 5000, #300, #280, #125, #250, # changed from 300
                      verbose             = 0,
                      early.stop.round    = 20,
                      watchlist           = watchlist,
                      maximize            = TRUE,
                      print.every.n = 20,
                      eval_metric ="auc"
  )
  
  if(i==1) pred <- predict(clf,data.matrix(testx))^2 else pred <- pred + predict(clf,data.matrix(testx))^2
  score <- c(score,clf$bestScore)
  print(mean(score))
  print(i)
}

sub <- data.frame(id=testItem$id,probability=sqrt(pred/i))
write.csv(sub,file="../LB/T_0006.csv",row.names=FALSE)
