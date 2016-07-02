setwd("~/Workspace/kaggle/Avito2016/R/")
library(data.table)
library(readr)
library(stringdist)
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

compute_features <- function(dt) {
    dt[, title_reduced_1 := trimws(gsub("[^A-Za-z0-9]+", " ", title1))]
    dt[, title_reduced_2 := trimws(gsub("[^A-Za-z0-9]+", " ", title2))]
    
    dt[, description_reduced_1 := trimws(gsub("[^A-Za-z0-9]+", " ", description1))]
    dt[, description_reduced_2 := trimws(gsub("[^A-Za-z0-9]+", " ", description2))]
    
    dt[, equalTitle_reduced := 1*(title_reduced_1 == title_reduced_2)]
    dt[, equalDescription_reduced := 1*(description_reduced_1 == description_reduced_2)]
    dt[, equalLenTitle_reduced := 1*(nchar(title_reduced_1) == nchar(title_reduced_2))]
    dt[, equalLenDescription_reduced := 1*(nchar(description_reduced_1) == nchar(description_reduced_2))]
    dt[, title_reduced_lev_dist := stringdist(title_reduced_1, title_reduced_2, method = "dl")/(pmax(nchar(title_reduced_1), nchar(title_reduced_2))+1)]
    dt[, descr_reduced_lev_dist := stringdist(description_reduced_1, description_reduced_2, method = "dl")/(pmax(nchar(description_reduced_1), nchar(description_reduced_2))+1)]
    
    # Utils 1
    dt[, n_title_1 := nchar(title1)]
    dt[, n_title_2 := nchar(title2)]
    dt[, title_len_lte_10 := 1*(n_title_1 < 10 & n_title_2 < 10)]
    dt[, title_len_lte_20 := 1*(n_title_1 < 20 & n_title_2 < 20)]
    dt[, title_len_lte_30 := 1*(n_title_1 < 30 & n_title_2 < 30)]
    dt[, title_len_gte_40 := 1*(n_title_1 > 40 & n_title_2 > 40)]
    
    # Utils 2
    dt[, n_desc_1 := nchar(description1)]
    dt[, n_desc_2 := nchar(description2)]
    dt[, desc_len_lte_10 := 1*(n_desc_1 < 10 & n_desc_2 < 10)]
    dt[, desc_len_lte_30 := 1*(n_desc_1 < 30 & n_desc_2 < 30)]
    dt[, desc_len_lte_50 := 1*(n_desc_1 < 50 & n_desc_2 < 50)]
    dt[, desc_len_gte_70 := 1*(n_desc_1 > 70 & n_desc_2 > 70)]
    
    output <- dt[, c("itemID_1", "itemID_2", "desc_len_lte_10", "desc_len_lte_30", "desc_len_lte_50", "desc_len_gte_70", "title_len_lte_10", "title_len_lte_20", "title_len_lte_30", "title_len_gte_40",
                        "descr_reduced_lev_dist", "title_reduced_lev_dist", "equalTitle_reduced",
                        "equalDescription_reduced", "equalLenTitle_reduced", "equalLenDescription_reduced"), with = FALSE]
    return(output)
}


train_nn <- compute_features(trainItem)
test_nn <- compute_features(testItem)

write.csv(train_nn, "../features/additional_string_base_features_train.csv", quote = FALSE, row.names = FALSE)
write.csv(test_nn, "../features/additional_string_base_features_test.csv", quote = FALSE, row.names = FALSE)
