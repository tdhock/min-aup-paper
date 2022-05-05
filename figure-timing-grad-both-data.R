library(data.table)

N.vec <- as.integer(10^seq(1, 7, by=0.5))
max.N <- max(N.vec)
all.labels.vec <- rep(c(-1,1), l=max.N)
set.seed(1)
all.pred.vec <- rnorm(max.N)
timing.dt.list <- list()
done.list <- list()
seconds.limit <- 1
do.sub <- function(...){
  mcall <- match.call()
  L <- as.list(mcall[-1])
  for(arg.name in names(L)){
    maybe.lang <- L[[arg.name]]
    if(is.language(maybe.lang)){
      L[[arg.name]] <- substitute(
        result.list[[NAME]] <- EXPR,
        list(NAME=arg.name, EXPR=maybe.lang))
    }
  }
  L
}
getNaive <- function(order.pred.vec, N.labels.vec, clip){
  is.positive <- N.labels.vec == 1
  pairs.dt <- data.table(expand.grid(
    positive=which(is.positive),
    negative=which(!is.positive)))
  pairs.dt[, diff := order.pred.vec[positive]-order.pred.vec[negative] ]
  pairs.dt[, diff.clipped := clip(margin-diff)]
  pairs.tall <- data.table::melt(
    pairs.dt,
    measure.vars=c("positive", "negative"),
    value.name="pred.i",
    variable.name="label")
  pairs.tall[, grad.sign := ifelse(label=="positive", -1, 1)]
  grad.dt <- pairs.tall[, .(
    gradient=sum(grad.sign*2*diff.clipped),
    loss=sum(diff.clipped^2)
  ), keyby=.(pred.i, label)]
  grad.dt[, pred := order.pred.vec]
  list(
    loss=pairs.dt[, sum(diff.clipped^2)],
    gradient=grad.dt[["gradient"]])
}

for(N in N.vec){
  print(N)
  print(done.list)
  N.pred.vec <- all.pred.vec[1:N]
  N.labels.vec <- sort(all.labels.vec[1:N])
  order.list <- list(unsorted=N.pred.vec)
  for(prediction.order in names(order.list)){
    order.pred.vec <- order.list[[prediction.order]]
    result.list <- list()
    margin <- 1
    m.args <- do.sub(Logistic={
      -N.labels.vec/(1+exp(order.pred.vec*N.labels.vec))
    }, Functional.square={
      grad.vec <- rep(NA_real_, length(order.pred.vec))
      for(s in c(1, -1)){
        I.coef <- ifelse(N.labels.vec == s, 1, 0)
        z <- margin-s*order.pred.vec
        quadratic <- sum(I.coef)
        linear <- sum(I.coef*s*2*z)
        constant <- sum(I.coef*z^2)
        grad.values <- 2*quadratic*order.pred.vec + linear
        is.loss <- N.labels.vec != s
        grad.vec[is.loss] <- grad.values[is.loss]
      }
      loss.values <- 
        quadratic*order.pred.vec^2 + linear*order.pred.vec + constant
      list(
        loss=sum(loss.values[is.loss]),
        gradient=grad.vec)
    }, Functional.squared_hinge={
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
        labels.sorted <- N.labels.vec[i]
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
        loss=sum(loss.values[is.loss]),
        gradient=grad.vec)
    }, Naïve.squared_hinge={
      getNaive(order.pred.vec, N.labels.vec, function(x)ifelse(x<0, 0, x))
    }, Naïve.square={
      getNaive(order.pred.vec, N.labels.vec, identity)
    },
    times=2)
    m.args[names(done.list[[prediction.order]])] <- NULL
    if(length(m.args) > 1){
      timing.df <- do.call(microbenchmark::microbenchmark, m.args)
      if("Naïve" %in% names(result.list)){
        stopifnot(with(result.list, all.equal(Naïve, Functional)))
      }
      N.dt <- data.table(timing.df)
      N.dt[, seconds := time/1e9]
      N.dt[, loss := paste(expr)]
      N.stats <- data.table::dcast(
        N.dt,
        loss ~ .,
        list(median, min, max, timings=length),
        value.var="seconds")
      done.pkgs <- N.stats[seconds_median > seconds.limit, paste(loss)]
      done.list[[prediction.order]][done.pkgs] <- TRUE
      timing.dt.list[[paste(N, prediction.order)]] <- 
        data.table(N, prediction.order, N.stats)
    }
  }
}
(timing.dt <- do.call(rbind, timing.dt.list))

data.table::fwrite(timing.dt, "figure-timing-grad-both-data.csv")

