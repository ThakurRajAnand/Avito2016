merge_with_key <- function(dt1, dt2, feature) {
  setkeyv(dt1, feature)
  setkeyv(dt2, feature)
  merge(dt1, dt2, by = c(feature), all.x = TRUE)
}

get_dt_with_hashes <- function(images_array, image_hashes) {
  ll <- str_split(gsub(" ", "", images_array1), ",")
  lldf <- data.frame(image = unlist(ll),
                     index1 = rep(seq_along(ll), lapply(ll, length)))
  
  lldf$image <- as.numeric(as.character(lldf$image))
  lldf <- as.data.table(lldf)
  setkey(lldf,image)
  lldf <- data.frame(merge(lldf, image_hashes, all.x=TRUE))
  lldf <- lldf[order(lldf$index1)]
  return(lldf)
}
