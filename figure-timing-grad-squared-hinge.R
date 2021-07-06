library(data.table)
library(ggplot2)
timing.dt <- data.table::fread("figure-timing-grad-squared-hinge-data.csv")
stats.dt <- timing.dt[, .(
  max=max(seconds),
  median=median(seconds),
  min=min(seconds)
), by=.(N, prediction.order, algorithm)]

func.only <- stats.dt[algorithm=="Functional"]
gg <- ggplot()+
  facet_grid(. ~ algorithm, labeller=label_both)+
  geom_line(aes(
    N, median, color=prediction.order),
    data=func.only)+
  geom_ribbon(aes(
    N, ymin=min, ymax=max, fill=prediction.order),
    alpha=0.5,
    data=func.only)+
  scale_x_log10(
    "Number of labeled examples = elements in gradient vector",
    limits=func.only[, c(min(N), max(N)*10)])+
  scale_y_log10(
    "Time to compute gradient (seconds),
median line and min/max band over 10 timings")
dl <- directlabels::direct.label(gg, "right.polygons")
png("figure-timing-grad-squared-hinge-sorted.png", width=6, height=4, units="in", res=200)
print(dl)
dev.off()

one <- stats.dt[prediction.order=="unsorted"]
one[, .SD[which.max(N)], by=algorithm][order(N)]
gg <- ggplot()+
  geom_line(aes(
    N, median, color=algorithm),
    data=one)+
  geom_ribbon(aes(
    N, ymin=min, ymax=max, fill=algorithm),
    alpha=0.5,
    data=one)+
  scale_x_log10(
    "Number of labeled examples = elements in gradient vector")+
  scale_y_log10(
    "Time to compute gradient (seconds),
median line and min/max band over 10 timings",
limits=one[, c(min(min), max(max)*1.3)])
dl <- directlabels::direct.label(gg, "top.polygons")
png("figure-timing-grad-squared-hinge.png", width=6, height=4, units="in", res=200)
print(dl)
dev.off()

