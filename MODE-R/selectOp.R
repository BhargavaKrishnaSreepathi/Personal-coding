selectOp <- function(PopArray){
        
        popsize <- length(PopArray$var1)
        pool <- rep(0,1,popsize)
        randnum <- floor(runif(2*popsize,1,(popsize)))
        newpop<-PopArray
        
        j<- c(1)
        sq <- 2*popsize
        
        for (i in seq(1,sq,by=2)){
                
               p1 <- randnum[i]
               p2 <- randnum[i+1]
               
               if ((PopArray$rank[p1] < PopArray$rank[p2] )|| ((PopArray$rank[p1] == PopArray$rank[p2]) && (PopArray$distance[p1] > PopArray$distance[p2]))){
                       
                      result <-1
                       
               }else{
                       result <-0
               }
               
               if (result == 1){
                       pool[j] <- p1
               }else{
                       pool[j] <- p2
               }
               
               j <- j+1
                
        }
        
        for (i in 1:popsize){
                
                newpop$var1[[i]] <- PopArray$var1[[pool[i]]]
                newpop$obj[[i]] <- PopArray$obj[[pool[i]]]
                newpop$cons[[i]] <- PopArray$cons[[pool[i]]]
                newpop$rank[[i]] <- PopArray$rank[[pool[i]]]
                newpop$distance[[i]] <- PopArray$distance[[pool[i]]]
                newpop$prefDistance[[i]] <- PopArray$prefDistance[[pool[i]]]
                newpop$nViol[[i]] <- PopArray$nViol[[pool[i]]]
                newpop$violSum[[i]] <- PopArray$violSum[[pool[i]]]
        }
        
        
        return(newpop)
}