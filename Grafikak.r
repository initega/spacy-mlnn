library(ggplot2)

iterations <-c(30,40,50,60,70,80,90,100,110,120,130,140,150)
accuracy <- c(66,66,100,100,66,100,100,100,100,100,100,100,100)
falsepos <- c(0,0,1,3,2,1,0,0,0,0,0,0,1)
datuak <- data.frame(accuracy,falsepos,iterations)
id<- c("blank")
datuak<-cbind(datuak,id)
datuak<-colnames(id)
p1<-ggplot(datuak, aes(factor(datuak$iterations), datuak$accuracy )) + ylim(0,100)+ 
  geom_point() + labs(x ="Iterations") + labs( y = "Accuracy")
p1

p2<-ggplot(datuak, aes(factor(datuak$iterations), datuak$falsepos)) + 
  geom_point() + labs(x ="Iterations") + labs( y = "False positives")
p2


iterations <-c(30,40,50,60,70,80,90,100,110,120,130,140,150)
accuracy <- c(100,100,100,100,100,100,100,100,100,100,100,100,100)
falsepos <- c(3,0,1,1,1,1,0,0,2,0,0,1,2)
datuak2 <- data.frame(accuracy,falsepos,iterations)
id<- c("over")
datuak2<-cbind(datuak2,id)
datuak<-rbind(datuak,datuak2)
p1<-ggplot(datuak, aes(factor(datuak$iterations), datuak$accuracy2 )) +
    ylim(0,100) + geom_point() + labs(x ="Iterations") + labs(y = "Accuracy")
p1

p2<-ggplot(datuak, aes(datuak$iterations, datuak$falsepos, colour= datuak$id)) + 
  geom_line() + labs(x ="Iterations") + labs( y = "False positives")
  + labs(colour= "model")
p2
