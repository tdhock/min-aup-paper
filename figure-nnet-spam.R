library(ggplot2)
library(data.table)
library(directlabels)

if(!file.exists("spam.csv")){
  download.file(
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data",
    "spam.csv")
}
spam.dt <- data.table::fread("spam.csv")
y.col <- 58
X.mat <- as.matrix(spam.dt[, -y.col, with=FALSE])
y01.vec <- spam.dt[[y.col]]
y.vec <- ifelse(y01.vec==1, 1, -1)

n.hidden.units <- 100 # u
n.folds <- 2
logistic.loss <- function(pred, y){
  log(1+exp(-y * pred))
}
sigmoid <- function(a){
  1/(1+exp(-a))
}
set.seed(2)
(unique.folds <- 1:n.folds)
fold.vec <- sample(rep(unique.folds, l=nrow(X.mat)))
validation.fold <- 1
is.train <- fold.vec != validation.fold
set.list <- list(
  train=is.train,
  validation=!is.train)
table(is.train)
X.train <- X.mat[is.train, ]
dim(X.mat)
dim(X.train)
y.train <- y.vec[is.train]
length(y.vec)
length(y.train)
##Scaling.
head(X.sc <- scale(X.train))
str(X.sc)
##attr(X.sc, "scaled:center")
attr(X.sc, "scaled:scale")
X.tilde <- scale(
  X.mat, attr(X.sc, "scaled:center"), attr(X.sc, "scaled:scale"))


epoch.vec <- seq(epoch+1, epoch+max.epochs)

epoch.dt.list <- list()
max.epochs <- 50
for(step.size in 10^seq(-2, 2)){
  epoch.vec <- 0:max.epochs
  for(batch.size in c(1, nrow(X.sc))){
    n.batches <- nrow(X.sc)/batch.size
    set.seed(1)
    V <- matrix(rnorm(ncol(X.sc)*n.hidden.units), ncol(X.sc), n.hidden.units)
    w <- rnorm(n.hidden.units)
    for(epoch in epoch.vec){
      ## train/validation error per epoch.
      A.mat <- X.tilde %*% V
      pred.vec <- sigmoid(A.mat) %*% w
      is.error <- ifelse(pred.vec > 0, 1, -1) != y.vec
      log.loss <- logistic.loss(pred.vec, y.vec)
      epoch.dt.list[[paste(step.size, batch.size, epoch)]] <- print(data.table(
        set=ifelse(is.train, "train", "validation"),
        is.error, 
        log.loss)[, list(
          step.size,
          batch.size,
          epoch,
          error.percent=mean(is.error)*100,
          mean.log.loss=mean(log.loss)
        ), by=list(set)])
      for(batch.i in 1:n.batches){
        batch.indices <- seq(batch.i, batch.i+batch.size-1)
        X.batch <- X.sc[batch.indices,,drop=FALSE]
        y.batch <- y.train[batch.indices]
        A <- X.batch %*% V
        Z <- sigmoid(A)
        b <- as.numeric(Z %*% w)
        dw <- -y.batch * sigmoid(-y.batch * b)
        A.deriv <- Z * (1-Z)
        dv <- unname(
          dw * A.deriv *
          matrix(w, nrow(A.deriv), ncol(A.deriv), byrow=TRUE))
        grad.w <- t(Z) %*% dw / nrow(X.batch)
        grad.V <- t(X.batch) %*% dv / nrow(X.batch)
        w <- w - step.size/n.batches * grad.w
        V <- V - step.size/n.batches * grad.V
      }
    }
  }
}
epoch.dt <- do.call(rbind, epoch.dt.list)
epoch.dt[, batchSize := factor(batch.size)]
epochs.tall <- melt(
  epoch.dt,
  measure.vars=c("error.percent", "mean.log.loss"))
min.tall <- epochs.tall[, .SD[which.min(value)], by=list(variable, set)]
set.colors <- c(
  train="black",
  validation="red")
gg <- ggplot()+
  ggtitle(paste(
    "Single layer neural network (57, 100, 1) for binary classification",
    "of spam data, N_train=2300, N_validation=2301, constant step size",
    sep="\n"))+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(variable ~ set, scales="free")+
  geom_line(aes(
    epoch, value, color=batchSize),
    data=epochs.tall)+
  scale_shape_manual(values=c(min=1))+
  geom_point(aes(
    epoch, value, color=batchSize, shape=Value),
    size=3,
    data=data.table(Value="min", min.tall))+
  ylab("")
dl <- direct.label(gg, "last.polygons")
dl

train.log <- epochs.tall[
  variable=="mean.log.loss" & set=="train"]
gg <- ggplot()+
  ggtitle(paste(
    "Single layer neural network (57, 100, 1) for binary classification",
    "of spam data, N_train=2300, N_validation=2301, constant step size",
    sep="\n"))+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_wrap("step.size", labeller=label_both)+
  geom_line(aes(
    epoch, value, color=batchSize),
    data=train.log)+
  scale_shape_manual(values=c(min=1))+
  ylab("")+
  coord_cartesian(ylim=c(NA, train.log[epoch==0, max(value)]))
dl <- direct.label(gg, "last.polygons")
dl






#####

epoch.dt.list <- list()
max.epochs <- 50
for(step.size in 10^seq(-2, 2)){
  epoch.vec <- 0:max.epochs
  for(batch.size in c(1, nrow(X.sc))){
    n.batches <- nrow(X.sc)/batch.size
    set.seed(1)
    V <- matrix(rnorm(ncol(X.sc)*n.hidden.units), ncol(X.sc), n.hidden.units)
    w <- rnorm(n.hidden.units)
    for(epoch in epoch.vec){
      ## train/validation error per epoch.
      A.mat <- X.tilde %*% V
      pred.vec <- sigmoid(A.mat) %*% w
      is.error <- ifelse(pred.vec > 0, 1, -1) != y.vec
      log.loss <- logistic.loss(pred.vec, y.vec)
      epoch.dt.list[[paste(step.size, batch.size, epoch)]] <- print(data.table(
        set=ifelse(is.train, "train", "validation"),
        is.error, 
        log.loss)[, list(
          step.size,
          batch.size,
          epoch,
          error.percent=mean(is.error)*100,
          mean.log.loss=mean(log.loss)
        ), by=list(set)])
      for(batch.i in 1:n.batches){
        batch.indices <- seq(batch.i, batch.i+batch.size-1)
        X.batch <- X.sc[batch.indices,,drop=FALSE]
        y.batch <- y.train[batch.indices]
        A <- X.batch %*% V
        Z <- sigmoid(A)
        b <- as.numeric(Z %*% w)
        dw <- -y.batch * sigmoid(-y.batch * b)
        A.deriv <- Z * (1-Z)
        dv <- unname(
          dw * A.deriv *
          matrix(w, nrow(A.deriv), ncol(A.deriv), byrow=TRUE))
        grad.w <- t(Z) %*% dw / nrow(X.batch)
        grad.V <- t(X.batch) %*% dv / nrow(X.batch)
        w <- w - step.size * grad.w
        V <- V - step.size * grad.V
      }
    }
  }
}
epoch.dt <- do.call(rbind, epoch.dt.list)
epoch.dt[, batchSize := factor(batch.size)]
epochs.tall <- melt(
  epoch.dt,
  measure.vars=c("error.percent", "mean.log.loss"))
train.log <- epochs.tall[
  variable=="mean.log.loss" & set=="train"]
gg <- ggplot()+
  ggtitle(paste(
    "Single layer neural network (57, 100, 1) for binary classification",
    "of spam data, N_train=2300, N_validation=2301, constant step size",
    sep="\n"))+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_wrap("step.size", labeller=label_both)+
  geom_line(aes(
    epoch, value, color=batchSize),
    data=train.log)+
  scale_shape_manual(values=c(min=1))+
  ylab("")+
  coord_cartesian(ylim=c(NA, train.log[epoch==0, max(value)]))
dl <- direct.label(gg, "last.polygons")
dl

