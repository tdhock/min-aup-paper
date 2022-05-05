library(data.table)
library(ggplot2)
timing.dt <- data.table::fread("figure-timing-grad-square-data.csv")
stats.dt <- timing.dt[, .(
  max=max(seconds),
  median=median(seconds),
  min=min(seconds)
), by=.(N, algorithm=sub("_", " ", sub("[.]", "\n", algorithm)))]
algo.colors <- c(
  "Functional\nsquared hinge"="#A6CEE3",
  "Naïve\nsquared hinge"="#1F78B4",
  Logistic="#B2DF8A", #"#33A02C",
  "Functional\nsquare"="#FB9A99",
  "Naïve\nsquare"="#E31A1C")
stats.dt[, .SD[which.max(N)], by=algorithm][order(N)]
gg <- ggplot()+
  scale_color_manual(values=algo.colors)+
  scale_fill_manual(values=algo.colors)+
  theme(legend.position="none")+
  directlabels::geom_dl(aes(
    N, median, color=algorithm, label=algorithm),
    method="top.polygons",
    data=stats.dt)+
  geom_line(aes(
    N, median, color=algorithm),
    data=stats.dt)+
  geom_ribbon(aes(
    N, ymin=min, ymax=max, fill=algorithm),
    alpha=0.5,
    data=stats.dt)+
  scale_x_log10(
    "Number of labeled examples = elements in gradient vector",
    limits=stats.dt[, c(min(N), max(N)*4)],
    breaks=10^seq(1, 7))+
  ## directlabels::geom_dl(aes(
  ##   N, median, color=algorithm, label=algorithm),
  ##   method="right.polygons",
  ##   data=stats.dt[grepl("Functional", algorithm)])+
  scale_y_log10(
    "Time to compute gradient (seconds),
median line and min/max band over 10 timings",
limits=stats.dt[, c(min(min), max(max)*4)])
png("figure-timing-grad-square-big.png", width=8, height=4, units="in", res=200)
print(gg)
dev.off()

gg <- ggplot()+
  scale_color_manual(values=algo.colors)+
  scale_fill_manual(values=algo.colors)+
  theme(legend.position="none")+
  directlabels::geom_dl(aes(
    N, median, color=algorithm, label=algorithm),
    method=list(cex=0.7, "top.polygons"),
    data=stats.dt)+
  geom_line(aes(
    N, median, color=algorithm),
    data=stats.dt)+
  geom_ribbon(aes(
    N, ymin=min, ymax=max, fill=algorithm),
    alpha=0.5,
    data=stats.dt)+
  coord_cartesian(xlim=stats.dt[, c(1e2, max(N)*4)])+
  scale_x_log10(
    "Number of labeled examples = elements in gradient vector",
    breaks=10^seq(1, 7))+
  ## directlabels::geom_dl(aes(
  ##   N, median, color=algorithm, label=algorithm),
  ##   method="right.polygons",
  ##   data=stats.dt[grepl("Functional", algorithm)])+
  scale_y_log10(
    "Time (seconds),
median line 
and min/max band 
over 10 timings",
breaks=10^seq(-5, 0),
limits=stats.dt[, c(min(min), max(max)*10)])
png("figure-timing-grad-square.png", width=6, height=2.2, units="in", res=200)
print(gg)
dev.off()

