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
imageSimTrain <- read_csv("../Data/vcc_euc_train.csv")
imageSimTest <- read_csv("../Data/vcc_euc_test.csv")


names(imageSimTrain) <- c("itemID_1","euc_dist_25")
names(imageSimTest) <- c("itemID_1","euc_dist_25")

imageSimTrain[is.na(imageSimTrain)] <- -9999
imageSimTest[is.na(imageSimTest)] <- -9999

imageSimTrain <- data.table(imageSimTrain)
imageSimTest <- data.table(imageSimTest)

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
setkey(imageSimTrain,"itemID_1")
train <- merge(train,imageSimTrain,all.x=TRUE)

setkey(train,"itemID_1")
setkey(trainItem,"itemID_1")

trainItem <- merge(trainItem,train,all.x=TRUE)
names(train) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2","images_array2","metroID2","euc_dist_25")
setkey(train,"itemID_2")
setkey(trainItem,"itemID_2")

trainItem <- merge(trainItem,train,all.x=TRUE)

test <- test[,c("itemID","categoryID","locationID","lat","lon","title","description","price","images_array","metroID")]
names(test) <- c("itemID_1","categoryID1","locationID1","lat1","lon1","title1","description1","price1","images_array1","metroID1")

test <- data.table(test)
setkey(test,"itemID_1")
setkey(imageSimTest,"itemID_1")
test <- merge(test,imageSimTest,all.x=TRUE)


setkey(test,"itemID_1")
setkey(testItem,"itemID_1")
testItem <- merge(testItem,test,all.x=TRUE)


names(test) <- c("itemID_2","categoryID2","locationID2","lat2","lon2","title2","description2","price2","images_array2","metroID2","euc_dist_25")
setkey(test,"itemID_2")
setkey(testItem,"itemID_2")

testItem <- merge(testItem,test,all.x=TRUE)

trainItem <- trainItem[,countItem1:=.N,by=itemID_1]
trainItem <- trainItem[,countItem2:=.N,by=itemID_2]

testItem <- testItem[,countItem1:=.N,by=itemID_1]
testItem <- testItem[,countItem2:=.N,by=itemID_2]

trainItem$countItemSum <- trainItem$countItem1 + trainItem$countItem2
testItem$countItemSum <- testItem$countItem1 + testItem$countItem2

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
trainItem$distance <-  sqrt((trainItem$lat1-trainItem$lat2)^2+(trainItem$lon1-trainItem$lon2)^2)

trainItem$index1 <- 1:nrow(trainItem)
trainItem <- trainItem[,image_count1:=length(str_split(images_array1,",")[[1]]),by=index1]
trainItem <- trainItem[,image_count2:=length(str_split(images_array2,",")[[1]]),by=index1]
#trainItem <- trainItem[,llDist:=earth.dist(rbind(c(lon1, lat1), c(lon2, lat2))),by=index1]


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

testItem$ncharTitle1 <- nchar(testItem$title1)
testItem$ncharTitle2 <- nchar(testItem$title2)

testItem$ncharDesc1 <- nchar(testItem$description1)
testItem$ncharDesc2 <- nchar(testItem$description2)

testItem$samePrice <- as.numeric(testItem$price1 == testItem$price2)
testItem$priceDiff <- pmax(testItem$price1/testItem$price2, testItem$price2/testItem$price1)
testItem$priceMax <- pmax(testItem$price1, testItem$price2, na.rm=TRUE)
testItem$priceMin <- pmin(testItem$price1, testItem$price2, na.rm=TRUE)
testItem$priceDiff <- ifelse(is.na(testItem$priceDiff), 0, testItem$priceDiff)
testItem$priceMin <- ifelse(is.na(testItem$priceMin), 0, testItem$priceMin)
testItem$priceMax <- ifelse(is.na(testItem$priceMax), 0, testItem$priceMax)


testItem$sameT <- as.numeric(testItem$title1 == testItem$title2)
testItem$sameD <- as.numeric(testItem$description1 == testItem$description2)
testItem$sameM <- as.numeric(testItem$metroID1 == testItem$metroID2)


testItem$distT1 <- stringdist(testItem$title1, testItem$title2, method = "jw")
testItem$distT2 <- (stringdist(testItem$title1, testItem$title2, method = "lcs") / pmax(testItem$ncharTitle1,testItem$ncharTitle2,na.rm=TRUE))

testItem$distD1 <- stringdist(testItem$description1, testItem$description2, method = "jw")
#testItem$distD2 <- (stringdist(testItem$description1, testItem$description2, method = "lcs") / pmax(testItem$ncharDesc1,testItem$ncharDesc2,na.rm=TRUE))

testItem$distT1 <- ifelse(is.na(testItem$distT1), 0, testItem$distT1)
testItem$distT2 = ifelse(is.na(testItem$distT2) | testItem$distT2 == Inf, 0, testItem$distT2) 

testItem$distD1 <- ifelse(is.na(testItem$distD1), 0, testItem$distD1)

testItem$title1StartsWithTitle2 <- as.numeric(substr(testItem$title1, 1, nchar(testItem$title2)) == testItem$title2)
testItem$title2StartsWithTitle1 <- as.numeric(substr(testItem$title2, 1, nchar(testItem$title1)) == testItem$title1)

testItem$titleCharDiff <- pmax(testItem$ncharTitle1/testItem$ncharTitle2, testItem$ncharTitle2/testItem$ncharTitle1)
testItem$titleCharMin <- pmin(testItem$ncharTitle1, testItem$ncharTitle2, na.rm=TRUE)
testItem$titleCharMax <- pmax(testItem$ncharTitle1, testItem$ncharTitle2, na.rm=TRUE)

testItem$descriptionCharDiff = pmax(testItem$ncharDesc1/testItem$ncharDesc2, testItem$ncharDesc2/testItem$ncharDesc1)
testItem$descriptionCharMin = pmin(testItem$ncharDesc1, testItem$ncharDesc2, na.rm=TRUE)
testItem$descriptionCharMax = pmax(testItem$ncharDesc1, testItem$ncharDesc2, na.rm=TRUE)
testItem$distance <-  sqrt((testItem$lat1-testItem$lat2)^2+(testItem$lon1-testItem$lon2)^2)

testItem$index1 <- 1:nrow(testItem)
testItem <- testItem[,image_count1:=length(str_split(images_array1,",")[[1]]),by=index1]
testItem <- testItem[,image_count2:=length(str_split(images_array2,",")[[1]]),by=index1]
#testItem <- testItem[,llDist:=earth.dist(rbind(c(lon1, lat1), c(lon2, lat2))),by=index1]

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

trainItem <- trainItem[,countLat1:=.N,by=lat1]
trainItem <- trainItem[,countLat2:=.N,by=lat2]
trainItem <- trainItem[,countLon1:=.N,by=lon1]
trainItem <- trainItem[,countLon2:=.N,by=lon2]

trainItem$countLatSum <- trainItem$countLat1 + trainItem$countLat2
trainItem$countLonSum <- trainItem$countLon1 + trainItem$countLon2

testItem <- testItem[,countLat1:=.N,by=lat1]
testItem <- testItem[,countLat2:=.N,by=lat2]
testItem <- testItem[,countLon1:=.N,by=lon1]
testItem <- testItem[,countLon2:=.N,by=lon2]
testItem$countLatSum <- testItem$countLat1 + testItem$countLat2
testItem$countLonSum <- testItem$countLon1 + testItem$countLon2

features <- c("sameLat","sameLon","sameLoc","ncharTitle1","ncharTitle2","ncharDesc1","ncharDesc2","samePrice","priceDiff","priceMax","priceMin","sameT","sameD",
"sameM","distT1","distT2","distD1", "title1StartsWithTitle2","title2StartsWithTitle1","titleCharDiff","titleCharMin", "titleCharMax","descriptionCharDiff","descriptionCharMin",
"descriptionCharMax","distance","image_count1","image_count2","commonHash","image_count1Ratio","image_count2Ratio","countItem1","countItem2","countItemSum","countLat1",
"countLat2","countLon1","countLon2","countLatSum","countLonSum","euc_dist_25.x","euc_dist_25.y")

trainx <- trainItem[,features,with=FALSE]
testx <- testItem[,features,with=FALSE]
response <- trainItem$isDuplicate

#split <- createDataPartition(as.factor(response),p=0.9)$Resample1
#model <- glm(response[split] ~. ,data=trainx[split,],family="binomial")
#predglm <- predict(model,trainx[-split,],type="response")
#colAUC(data.frame(predglm),response[-split])

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

if(i==1) pred <- predict(clf,data.matrix(testx))^2 else pred <- pred + predict(clf,data.matrix(testx))^2
score <- c(score,clf$bestScore)
print(mean(score))
print(i)
}

sub <- data.frame(id=testItem$id,probability=sqrt(pred/i))
write.csv(sub,file="../LB/T_0013.csv",row.names=FALSE)




