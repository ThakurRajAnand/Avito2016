library(data.table)
library(readr)
library(stringdist)
library(geosphere)
library(xgboost)
library(caTools)
library(jsonlite)

train <- read_csv("../Data/ItemInfo_train.csv")
test <- read_csv("../Data/ItemInfo_test.csv")
trainItem <- read_csv("../Data/ItemPairs_train.csv")
testItem <- read_csv("../Data/ItemPairs_test.csv")
train <- as.data.table(train)
test <- as.data.table(test)
trainItem <- as.data.table(trainItem)
testItem <- as.data.table(testItem)

trainItem <- testItem
train <- test
#for test
N <- 1044196
abbs <- c(trainItem$itemID_1[1:N], trainItem$itemID_2[1:N])
abbs <- unique(abbs)

# 111 minutes on full data
system.time(ff <- lapply(train[itemID %in% abbs][['attrsJSON']], function(x) {
  if (is.na(x)) return(data.frame(EmptyJson = 1))
  data.frame(fromJSON(x), stringsAsFactors = FALSE)
})
)

gg <- rbindlist(ff, use.names = TRUE, fill = TRUE)

num <- unlist(lapply(gg, function(x) sum(!is.na(x))))

selected <- names(gg)

gg2 <- subset(gg, select = selected)
gg2[, itemID := train[itemID %in% abbs][['itemID']]]

reduced_train <- trainItem[1:N]
reduced_train[, generationMethod := NULL]

setnames(gg2, names(gg2), paste0(names(gg2), '_1'))
setkey(gg2, itemID_1)
setkey(reduced_train, itemID_1)

tr <- merge(reduced_train, gg2, by = "itemID_1")

setnames(gg2, names(gg2), gsub("_1", "_2", names(gg2)))
setkey(gg2, itemID_2)
setkey(tr, itemID_2)
tr <- merge(tr, gg2, by = "itemID_2")

#precompute ones

for (feature in selected) {
  fe1 <- paste0(feature, '_1')
  fe2 <- paste0(feature, '_2')
  eqf <- paste0('isEqual', feature)
  val <- paste0('Value', feature)
  tr[, temp1Missing := 1*(is.na(tr[[fe1]]))]
  tr[, temp2Missing := 1*(is.na(tr[[fe2]]))]
  tr[temp1Missing == 1, (fe1) := '-1']
  tr[temp2Missing == 1, (fe2) := '-2']
  tr[, (eqf) := 1*(tr[[fe1]] == tr[[fe2]])]
  tr[[val]] <- tr[[fe1]]
  tr[tr[[eqf]] == 0, (val) := '-999' ]
  tr[, c('temp1Missing', 'temp2Missing', fe1, fe2) := NULL]
}

testJSON <- tr 
saveRDS(testJSON,file="testJSON.RDS")
