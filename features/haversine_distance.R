library(data.table)
library(readr)
library(stringdist)
library(geosphere)
library(xgboost)
library(caTools)
setwd("~/Workspace/kaggle/Avito2016/R/")

train <- read_csv("../Data/ItemInfo_train.csv")
test <- read_csv("../Data/ItemInfo_test.csv")
trainItem <- read_csv("../Data/ItemPairs_train.csv")
testItem <- read_csv("../Data/ItemPairs_test.csv")
train <- as.data.table(train)
test <- as.data.table(test)
trainItem <- as.data.table(trainItem)
testItem <- as.data.table(testItem)

train[, c("images_array", "attrsJSON") := NULL]
test[, c("images_array", "attrsJSON") := NULL]

# === prepare features ==== FOR TRIAN ONLY ====

setnames(train, names(train), paste0(names(train), '_1'))
setkey(train, itemID_1)
setkey(trainItem, itemID_1)
tr <- merge(train, trainItem, by = "itemID_1")
setnames(train, names(train), gsub("_1", "_2", names(train)))
setkey(train, itemID_2)
setkey(tr, itemID_2)
tr <- merge(tr, train, by = "itemID_2")

# Category is factor 
tr <- tr[, Category := categoryID_1][, c("categoryID_1", "categoryID_2") := NULL]
tr[, isLocation := 1*(locationID_1 == locationID_2)]
tr[, c("locationID_1", "locationID_2") := NULL]
tr[, c("generationMethod") := NULL]
tr[, metroID1Missing := 1*(is.na(metroID_1))]
tr[, metroID2Missing := 1*(is.na(metroID_2))]
tr[is.na(metroID_1), metroID_1 := -1]
tr[is.na(metroID_2), metroID_2 := -2]
tr[, isMetroID := 1*(metroID_1 == metroID_2)]
tr[, c("metroID_1", "metroID_2") := NULL]
tr[, price1Missing := 1*(is.na(price_1))]
tr[, price2Missing := 1*(is.na(price_2))]
tr[, onePriceMiss := 1*(price1Missing == 1 | price2Missing)]
tr[, twoPriceMiss := 1*(price1Missing == 1 & price2Missing)]
tr[ onePriceMiss == 0, priceRatio := pmax(price_1/price_2, price_2/price_1)]
tr[ onePriceMiss != 0, priceRatio := -10]
tr[, c("price1Missing", "price2Missing") := NULL]
tr[, oneMetroMiss := 1*(metroID1Missing == 1 | metroID2Missing)]
tr[, twoMetroMiss := 1*(metroID1Missing == 1 & metroID2Missing)]
tr[, c("metroID1Missing", "metroID2Missing") := NULL]
tr[is.na(price_1), price_1 := -10]
tr[is.na(price_2), price_2 := -30]
tr[, equalPrice := 1*(price_1 == price_2)]
tr[, c("price_1", "price_2") := NULL]
tr[, distanceAd := distHaversine(as.matrix(tr[, c("lon_1", "lat_1"), with = FALSE]),
                                 as.matrix(tr[, c("lon_2", "lat_2"), with = FALSE]))]

tr[, c("lon_1", "lat_1", "lon_2", "lat_2") := NULL]
# Text
tr[, equalTitle := 1*(title_1 == title_2)]
tr[, equalDescription := 1*(description_1 == description_2)]
tr[, titleStringDist := stringdist(title_1, title_2, method = "jw")]
tr[, titleCharRatio := pmax(nchar(title_1)/nchar(title_2),
                            nchar(title_2)/nchar(title_1))]
tr[, descriptionCharRatio := pmax(nchar(description_1)/nchar(description_2),
                                  nchar(description_2)/nchar(description_1))]
tr[, c("title_1", "title_2", "description_1", "description_2") := NULL]
tr[, c("itemID_2", "itemID_1") := NULL]
tr[is.na(titleStringDist), titleStringDist := -1]
tr[is.na(equalTitle), equalTitle := 0]
tr[is.na(equalDescription), equalDescription := 0]

response <- tr[['isDuplicate']]
tr[, c("isDuplicate") := NULL]

# === Training ===
split <- sample.split(response, 0.8)
dtrain <- xgb.DMatrix(data = data.matrix(subset(tr, split)), label = response[split])
dval <- xgb.DMatrix(data = data.matrix(subset(tr, !split)), label = response[!split])
watchlist <- list(val = dval, train = dtrain)bvc,m

param <- list(
  objective = "binary:logistic",
  booster = "gbtree",
  eta                 = 0.05,
  max_depth           = 4,
  subsample           = 0.8,
  colsample_bytree    = 0.8)

clf <- xgb.train(params              = param,
                 data                = dtrain,
                 nrounds             = 350, #300, #280, #125, #250, # changed from 300
                 verbose             = 0,
                 early.stop.round    = 10,
                 watchlist           = watchlist,
                 maximize            = TRUE,
                 print.every.n = 1,
                 eval_metric ="auc")

imp <- xgb.importance(feature_names = names(tr), model = clf)
imp

xgb.dump(clf, "test.dump")
