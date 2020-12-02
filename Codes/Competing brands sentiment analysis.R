library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)
library(SentimentAnalysis)
library(doMC)
library("SnowballC")
library("tm")
library("twitteR")
library("wordcloud")

setwd('C:/Users/PKN/Documents/Sentiment Project')

df<- read.csv("vivonex_-_2018_08_01_08_38_838153_2.csv", stringsAsFactors = FALSE)
glimpse(df)

set.seed(1)
df <- df[sample(nrow(df)), ]
df <- df[sample(nrow(df)), ]
glimpse(df)                                         

#remove hastags, urls and @s
tweets.df2 <- gsub("&amp", "", df$CONTENT)
tweets.df2 <- gsub("(s?)(f|ht)tp(s?)://\\S+\\b","",tweets.df2)
tweets.df2<- gsub("[A-Za-z]{1,5}[.][A-Za-z]{2,3}/[A-Za-z0-9]+\\b", "", tweets.df2)
tweets.df2<- gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", tweets.df2)
tweets.df2 <- gsub("@\\w+", "", tweets.df2)


#remove punctuation, stop words and do stemming
myCorpus  <- Corpus(VectorSource(tweets.df2))
myCorpus <- tm_map(myCorpus, removeWords, stopwords('english'))
myCorpus <- tm_map(myCorpus, content_transformer(tolower))
myCorpus <- tm_map(myCorpus, removePunctuation)
corpus.clean <- myCorpus
myCorpus <- tm_map(myCorpus, stemDocument)


# tokenize the corpus
myCorpusTokenized <- lapply(myCorpus, scan_tokenizer)
# stem complete each token vector
myTokensStemCompleted <- lapply(myCorpusTokenized, stemCompletion, corpus.clean)
# concatenate tokens by document, create data frame
myDf <- data.frame(text = sapply(myTokensStemCompleted, paste, collapse = " "), stringsAsFactors = FALSE)

Df_Final <- myDf$text
corpus.clean= Corpus(VectorSource(Df_Final))
sentiment <- analyzeSentiment(corpus.clean)
sentiment$SentimentGI
sentiment <- convertToDirection(sentiment$SentimentGI)
class(sentiment)
dtm <- DocumentTermMatrix(corpus.clean)
inspect(dtm[40:50, 10:15])
df = cbind(df,sentiment)
glimpse(df)
df.train <- df[1:3000,]
df.test <- df[3001:4375,]
dtm.train <- dtm[1:3000,]
dtm.test <- dtm[3001:4375,]
corpus.clean.train <- corpus.clean[1:3000]
corpus.clean.test <- corpus.clean[3001:4375]
dim(dtm.train)
frequency <- findFreqTerms(dtm.train, 0)
length((frequency))
# Use no frequent words (fivefreq) to build the DTM 
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train,control=list(dictionary = frequency))
dim(dtm.train.nb) 
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test,control=list(dictionary = frequency))
dim(dtm.test.nb)
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

# Apply the convert_count function to get final training and testing DTMs
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)
system.time( classifier <- naiveBayes(trainNB, df.train$sentiment, laplace = 1) ) 

#Training Prediction Statistics
system.time( pred_train <- predict(classifier, newdata=trainNB) )
table("Predictions"= pred_train,  "Actual" = df.train$sentiment )
conf.mat <- confusionMatrix(pred_train, df.train$sentiment)
conf.mat


#Test Prediction Statistics
system.time( pred_test <- predict(classifier, newdata=testNB) )
table("Predictions"= pred_test,  "Actual" = df.test$sentiment )
conf.mat <- confusionMatrix(pred_test, df.test$sentiment)
conf.mat

# Overall sentiments
dtm.nb <- DocumentTermMatrix(corpus.clean, control=list(dictionary = fivefreq)) 
fullNB <- apply(dtm.nb, 2, convert_count)
new_sentiment <- predict(classifier, newdata=fullNB)
conf.mat <- confusionMatrix(new_sentiment, df$sentiment)
conf.mat

category_senti <- new_sentiment
category_senti2 <- cbind(myDf,category_senti)
table(category_senti)

#getting flag if the tweet has Nex in it
nex_y<-grepl("nex+",category_senti2$text, perl=TRUE)
category_senti3<-cbind(category_senti2,nex_y)
View(category_senti3)

#getting flag if the tweet has findx in it
find_y<-grepl("find+",category_senti2$text, perl=TRUE)
category_senti4<-cbind(category_senti3,find_y)
View(category_senti4)    

findx_tweets<-category_senti4[ which(category_senti4$find_y=='TRUE'), ]
nex_tweets<-category_senti4[ which(category_senti4$nex_y=='TRUE'), ]
both_tweets<-category_senti4[ which(category_senti4$nex_y=='TRUE' & category_senti4$find_y=='TRUE'), ]
nex_sentiments<-table(nex_tweets$category_senti)
findx_sentiments<-table(findx_tweets$category_senti)

nex_sentiments  
findx_sentiments

docs <- Corpus(VectorSource(nex_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

docs <- Corpus(VectorSource(findx_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))


cam_y<-grepl("cam+",category_senti4$text, perl=TRUE)
category_senti5<-cbind(category_senti4,cam_y)
View(category_senti5)
dis_y<-grepl("dis+",category_senti5$text, perl=TRUE)
category_senti6<-cbind(category_senti5,dis_y)
View(category_senti6) 
bezel_y<-grepl("bez+",category_senti6$text, perl=TRUE)
category_senti7<-cbind(category_senti6,bezel_y)
View(category_senti7)
pop_y<-grepl("pop+",category_senti7$text, perl=TRUE)
category_senti8<-cbind(category_senti7,pop_y)
View(category_senti8)

#camera comparison
findx_tweets<-category_senti8[ which(category_senti8$find_y=='TRUE' & category_senti8$cam_y=='TRUE'), ]
nex_tweets<-category_senti8[ which(category_senti8$nex_y=='TRUE' & category_senti8$cam_y=='TRUE'), ]
Camera_nex<-table(nex_tweets$category_senti)
Camera_findx<-table(findx_tweets$category_senti)
Camera_nex
Camera_findx

docs <- Corpus(VectorSource(nex_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))
docs <- Corpus(VectorSource(findx_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

#display comparison
findx_tweets<-category_senti8[ which(category_senti8$find_y=='TRUE' & category_senti8$dis_y=='TRUE'), ]
nex_tweets<-category_senti8[ which(category_senti8$nex_y=='TRUE' & category_senti8$dis_y=='TRUE'), ]
Display_nex<-table(nex_tweets$category_senti)
Display_findx<-table(findx_tweets$category_senti)
Display_nex
Display_findx

docs <- Corpus(VectorSource(nex_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))
docs <- Corpus(VectorSource(findx_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))



#bezelless comparison
findx_tweets<-category_senti8[ which(category_senti8$find_y=='TRUE' & category_senti8$bezel_y=='TRUE'), ]
nex_tweets<-category_senti8[ which(category_senti8$nex_y=='TRUE' & category_senti8$bezel_y=='TRUE'), ]
bezel_nex<-table(nex_tweets$category_senti)
bezel_findx<-table(findx_tweets$category_senti)
bezel_nex
bezel_findx

docs <- Corpus(VectorSource(nex_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))
docs <- Corpus(VectorSource(findx_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))


#popup comparison
findx_tweets<-category_senti8[ which(category_senti8$find_y=='TRUE' & category_senti8$pop_y=='TRUE'), ]
nex_tweets<-category_senti8[ which(category_senti8$nex_y=='TRUE' & category_senti8$pop_y=='TRUE'), ]
popup_nex<-table(nex_tweets$category_senti)
popup_findx<-table(findx_tweets$category_senti)
popup_nex
popup_findx

docs <- Corpus(VectorSource(nex_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))
docs <- Corpus(VectorSource(findx_tweets$text))
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

