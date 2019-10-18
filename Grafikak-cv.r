library(ggplot2)

iterations <-c(15,30,60,100,150,400)
accuracy <- c(0.648,0.74,0.792,0.72,0.729,0.7)
falsepos <- c(0,0,1,3,2,1)
datuak <- data.frame(accuracy,iterations)
id<- c("blank")
datuak<-cbind(datuak,id)
datuak<-colnames(id)
p1<-ggplot(datuak, aes(factor(datuak$iterations), datuak$accuracy )) + ylim(0,100)+ 
  geom_point() + labs(x ="Iterations") + labs( y = "Accuracy")
p1

X11()
plot(p1)

# p2<-ggplot(datuak, aes(factor(datuak$iterations), datuak$falsepos)) + 
#   geom_point() + labs(x ="Iterations") + labs( y = "False positives")
# p2


