library(data.table)
library(xgboost)
library(text2vec)
library(SnowballC)
library(readr)

train <- read_csv("../Data/ItemInfo_train.csv")
test  <- read_csv("../Data/ItemInfo_test.csv")

train <- as.data.table(train)
test <- as.data.table(test)

full <- rbind(train[, c("itemID", "description"), with=FALSE],
              test[, c("itemID", "description"), with=FALSE])

train_items <- train[['itemID']]
test_items <- test[['itemID']]

rm(train)
rm(test)

stem_tokenizer <- function(x, tokenizer = word_tokenizer) {
  x %>% 
    tokenizer %>% 
    lapply(wordStem, 'ru')
}

token_title <- full[['description']] %>% 
  tolower %>% 
  stem_tokenizer

it_title <- itoken(token_title)
vocab_title <- create_vocabulary(it_title, ngram = c(1L, 3L))

pruned_vocab_title<- prune_vocabulary(vocab_title, term_count_min = 10, doc_proportion_max = 0.01)
it_title <- itoken(token_title)
v_vectorizer_title <- vocab_vectorizer(pruned_vocab_title)
dtm_title <- create_dtm(it_title, v_vectorizer_title)

rm(pruned_vocab_title)
rm(token_title)
rm(vocab_title)

train_dtm <- dtm_title[1:length(train_items), ]
test_dtm <- dtm_title[(length(train_items)+1):dim(dtm_title)[1], ]

rm(dtm_title)

train_pairs <- fread("../Data/ItemPairs_train.csv")
test_pairs  <- fread("../Data/ItemPairs_test.csv")

gen_title <- function(dtm, pairs, items) {
  pairs[, ord := 1:nrow(pairs)]
  ID <- data.table(itemID = items, row.id = 1:length(items))
  
  # Train
  Item_1 <- dtm[merge(pairs, ID, by.x = 'itemID_1', by.y = 'itemID', all.x = TRUE)[order(ord)][['row.id']] ,]
  Item_2 <- dtm[merge(pairs, ID, by.x = 'itemID_2', by.y = 'itemID', all.x = TRUE)[order(ord)][['row.id']] ,]
  
  mult <- Item_1 * Item_2
  
  mult_bin <- transform_binary(mult)
  sum_bin  <- transform_binary(Item_1 + Item_2)
  
  # Jaccard Distance
  jaccard_dist <- (Matrix::rowSums(mult_bin)/Matrix::rowSums(sum_bin))
  jaccard_dist <- as.numeric(jaccard_dist)
  jaccard_dist[is.na(jaccard_dist)] <- 0.0
  
  # Equal sum
  sum_equal <- 1*(Matrix::rowSums(Item_1) == Matrix::rowSums(Item_2))
  
  # Cosine Similarity
  cosine_sim <- Matrix::rowSums(mult) / (sqrt(Matrix::rowSums(Item_1 * Item_1)) * sqrt(Matrix::rowSums(Item_2 * Item_2)))
  cosine_sim[is.na(cosine_sim)] <- 0
  
  # Dice distance
  dice_dist <- 1 - 2 * Matrix::rowSums(mult_bin) / (Matrix::rowSums(Item_1) + Matrix::rowSums(Item_2))
  dice_dist[is.na(dice_dist)] <- 1
  
  #Manhattan distance
  manhattan_distance <- Matrix::rowSums(abs(Item_2 - Item_1))
  
  #Hamming distance
  hamming_distance <- Matrix::rowSums(abs(transform_binary(Item_2) - transform_binary(Item_1)))
  
  #Cosine using tf-idf
  Item_1 <- transform_tfidf(Item_1)
  Item_2 <- transform_tfidf(Item_2)
  mult <- Item_1 * Item_2
  
  sum_bin  <- Item_1 + Item_2
  
  cosine_tfidf_sim <- Matrix::rowSums(mult) / (sqrt(Matrix::rowSums(Item_1 * Item_1)) * sqrt(Matrix::rowSums(Item_2 * Item_2)))
  cosine_tfidf_sim[is.na(cosine_tfidf_sim)] <- 0
  
  dt <- data.table(d_jaccard_dist = jaccard_dist,
                   d_sum_equal = sum_equal,
                   d_cosine_sim = cosine_sim,
                   d_dice_dist = dice_dist,
                   d_manhattan_distance = manhattan_distance,
                   d_hamming_distance = hamming_distance,
                   d_tfidf_cosine = cosine_tfidf_sim,
                   itemID_1 = pairs[['itemID_1']],
                   itemID_2 = pairs[['itemID_2']])
  return(dt)
}

trainNew <- gen_title(train_dtm, train_pairs, train_items)
testNew <- gen_title(test_dtm, test_pairs, test_items)

write_csv(trainNew, "../Data/oleksii_description_features_train.csv")
write_csv(testNew, "../Data/oleksii_description_features_test.csv")

