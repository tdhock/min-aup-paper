library(ggplot2)
library(data.table)
## define variables which are specific to each data set.
prefix <- "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/"
one.data <- list()
f <- "spam.data"
if(!file.exists(f)){
  u <- paste0(prefix, f)
  download.file(u, f)
}
full.dt <- data.table::fread(f)
label.col.id <- ncol(full.dt)
X.raw <- as.matrix(full.dt[, -label.col.id, with=FALSE])
y.vec <- full.dt[[label.col.id]]
is.01 <- y.vec == 0 | y.vec == 1
X.unsc <- X.raw[is.01, ]
is.constant <- apply(X.unsc, 2, sd) == 0
X.vary <- X.unsc[, !is.constant]
range.list <- list()
for(fn in c("min","max")){
  range.list[[fn]] <- matrix(
    apply(X.vary, 2, fn),
    nrow(X.vary),
    ncol(X.vary),
    byrow=TRUE)
}
X.all <- with(range.list, (X.vary-min)/(max-min)*2-1)
y.all <- y.vec[is.01]
set.seed(1)
n.folds <- 7
uniq.folds <- 1:n.folds
fold.dt <- data.table(label=y.all)
fold.dt[, fold := sample(rep(uniq.folds, l=.N)), by=label]
fold.dt[, table(fold, label)]
test.fold <- 1
set.vec <- ifelse(fold.dt$fold==test.fold, "test", "train")
for(set in unique(set.vec)){
  is.set <- set.vec==set
  one.data[[set]] <- list(
    x=unname(X.all[is.set,]),
    y=as.integer(y.all[is.set]))
}
X.eig <- eigen(t(X.all) %*% X.all)
str(X.eig)
## > with(X.eig, max(values)/min(values))
## [1] 727559

subset.list <- function(L, keep){
  with(L, list(x=x[keep,], y=y[keep]))
}

subset.n0.n1 <- function(L, n0, n1=n0){
  max.dt <- data.table(label=c(0,1), max.number=c(n0,n1))
  keep <- data.table(
    label=L$y,
    orig.index=seq_along(L$y)
  )[, label.number := 1:.N, by=label][
    max.dt, on="label"
  ][label.number <= max.number, orig.index]
  subset.list(L, keep)
}

get.loss.grad <- function(order.pred.vec, N.labels.vec, margin=1){
  stopifnot(N.labels.vec %in% c(0,1))
  labels.vec <- ifelse(N.labels.vec==1, 1, -1)
  label.tab <- table(labels.vec)
  denominator <- prod(label.tab)
  augmented.pred <- order.pred.vec+ifelse(N.labels.vec==1, 0, margin)
  grad.vec <- rep(NA_real_, length(order.pred.vec))
  sorted.indices <- order(augmented.pred)
  for(s in c(1, -1)){
    i <- if(s==1){
           sorted.indices
         }else{
           rev(sorted.indices)
         }
    pred.sorted <- order.pred.vec[i]
    labels.sorted <- labels.vec[i]
    I.coef <- ifelse(labels.sorted == s, 1, 0)
    z <- margin-s*pred.sorted
    quadratic <- cumsum(I.coef)
    linear <- cumsum(I.coef*s*2*z)
    constant <- cumsum(I.coef*z^2)
    ## below is only for gradient.
    grad.values <- 2*quadratic*pred.sorted + linear
    is.loss <- labels.sorted == -s
    grad.indices <- i[is.loss]
    grad.vec[grad.indices] <- grad.values[is.loss]
  }
  loss.values <- quadratic*pred.sorted^2 + linear*pred.sorted + constant
  list(
    loss=sum(loss.values[is.loss])/denominator,
    gradient=grad.vec/denominator)
}

learn <- function(subtrain, verbose=0, lambda=0, tol=1e-3){
  non.const.i <- which(apply(subtrain$x, 2, sd) > 0)
  full.weight <- rep(0, ncol(subtrain$x))
  X.subtrain <- subtrain$x[,non.const.i]
  weight.vec <- rep(0, ncol(X.subtrain))
  full.weight[non.const.i] <- weight.vec
  epoch <- 0
  weight.mat.list <- list()
  weight.mat.list[[paste(epoch)]] <- full.weight
  crit <- Inf
  step.size <- 1
  while(crit > tol){
    epoch <- epoch+1
    pred.vec <- X.subtrain %*% weight.vec
    loss.grad.list <- get.loss.grad(pred.vec, subtrain$y)
    loss.grad <- t(X.subtrain) %*% loss.grad.list$gradient
    pen.grad <- weight.vec * lambda/length(weight.vec)
    weight.grad <- loss.grad + pen.grad
    crit <- sum(abs(weight.grad))
    weight.after.step <- function(s)weight.vec - s * weight.grad
    cost.after.step <- function(s){
      w <- weight.after.step(s)
      loss <- get.loss.grad(X.subtrain %*% w, subtrain$y)$loss
      loss+0.5*sum(weight.vec^2)*lambda/length(weight.vec)
    }
    step.factor <- c(2, 1, 0.5, 0.1)
    step.candidates <- step.size*step.factor
    step.cost <- sapply(step.candidates, cost.after.step)
    step.size <- step.candidates[which.min(step.cost)]
    if(step.size<1e-3)step.size <- 1/lambda
    if(verbose)cat(sprintf(
      "epoch=%d cost=%f crit=%f best_step=%f\n",
      epoch,
      cost.after.step(0),
      crit,
      step.size))
    if(FALSE){
      roc.df <- WeightedROC::WeightedROC(pred.vec, subtrain$y)
      WeightedROC::WeightedAUC(roc.df)
    }
    weight.vec <- weight.after.step(step.size)
    full.weight[non.const.i] <- weight.vec
    weight.mat.list[[paste(epoch)]] <- full.weight
  }
  do.call(cbind, weight.mat.list)
}

test.y.counts <- table(one.data$test$y)
train.n <- 1000
test.dt.list <- list()
for(train.prop.pos in c(0.01, 0.5)){
  train.n1 <- train.n * train.prop.pos
  train.n0 <- train.n-train.n1
  prop.data.list <- list(
    test=subset.n0.n1(one.data$test, min(test.y.counts)),
    train=subset.n0.n1(one.data$train, train.n0, train.n1))
  sapply(prop.data.list, function(L)table(L$y))
  for(seed in 1:10){
    index.dt <- data.table(label=prop.data.list$train$y)
    sv <- c("subtrain", "validation")
    index.dt[, set := sample(rep(sv, l=.N)), by=label]
    index.dt[, .(count=.N), by=.(label,set)]
    tlist <- list()
    for(s in sv){
      is.set <- index.dt$set==s
      tlist[[s]] <- subset.list(prop.data.list$train, is.set)
    }
    pen.weight.mat.list <- list()
    for(penalty in 10^seq(3, -5, by=-0.5)){
      print(penalty)
      learn.mat <- learn(tlist$subtrain, verbose=1, lambda=penalty)
      pen.weight.mat.list[[paste(-log10(penalty))]] <- learn.mat[,ncol(learn.mat)]
    }
    reg.type.list <- list(
      "-log10(penalty)"=do.call(cbind, pen.weight.mat.list),
      epochs=learn(tlist$subtrain))
    metrics.list <- list(
      loss=function(pred, y)get.loss.grad(pred, y)$loss,
      auc=function(pred, y){
        roc.df <- WeightedROC::WeightedROC(pred, y)
        WeightedROC::WeightedAUC(roc.df)
      })
    metrics.dt.list <- list()
    for(reg.param in names(reg.type.list)){
      weight.mat <- reg.type.list[[reg.param]]
      for(set.name in names(tlist)){
        set.data <- tlist[[set.name]]
        pred.mat <- set.data$x %*% weight.mat
        for(valid.metric.name in names(metrics.list)){
          metrics.dt.list[[paste(
            reg.param, set.name, valid.metric.name
          )]] <- data.table(
            reg.param,
            set.name,
            valid.metric.name,
            reg.i=1:ncol(pred.mat),
            reg.value=as.numeric(colnames(pred.mat)),
            valid.metric.value=apply(
              pred.mat, 2, metrics.list[[valid.metric.name]], set.data$y))
        }
      }
    }
    metrics.dt <- do.call(rbind, metrics.dt.list)
    if(FALSE){
      ggplot()+
        geom_line(aes(
          reg.value, valid.metric.value,
          color=set.name),
          data=metrics.dt)+
        facet_grid(
          valid.metric.name ~ reg.param, scales="free", labeller=label_both)
    }
    seed.test.dt <- metrics.dt[set.name=="validation", {
      which.m <- if(valid.metric.name=="auc")which.max else which.min
      reg.i <- .SD[which.m(valid.metric.value), reg.i]
      weight.vec <- reg.type.list[[reg.param]][, reg.i]
      pred.vec <- one.data$test$x %*% weight.vec
      data.table(test.metric.name=names(metrics.list))[, {
        metric.fun <- metrics.list[[test.metric.name]]
        data.table(test.metric.value=metric.fun(pred.vec, one.data$test$y))
      }, by=test.metric.name]
    }, by=.(reg.param, valid.metric.name)]
    test.dt.list[[paste(train.prop.pos, seed)]] <- data.table(
      train.prop.pos, seed, seed.test.dt)
  }
}
test.dt <- do.call(rbind, test.dt.list)

data.table::fwrite(test.dt, "figure-l2-vs-early-stopping-spam-data.csv")

