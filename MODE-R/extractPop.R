extractPop <- function(combinepop){
        
        # extract the best n individuals in combine pop
        
        nextpop <- c()
        distance <- c()
        popsize <- length(combinepop$var1) /2
        for (i in (1):(popsize)){
                
                nextpop$var1[[i]] <- combinepop$var1[[i]]
                nextpop$obj[[i]] <- combinepop$obj[[i]]
                nextpop$cons[[i]] <- combinepop$cons[[i]]
                nextpop$rank[[i]] <- combinepop$rank[[i]]
                nextpop$distance[[i]] <- combinepop$distance[[i]]
                nextpop$prefDistance[[i]] <- combinepop$prefDistance[[i]]
                nextpop$nViol[[i]] <- combinepop$nViol[[i]]
                nextpop$violSum[[i]] <- combinepop$violSum[[i]]
                
                
        }
        
        rankVector <- combinepop$rank
        
        n <- c(0)
        rank <- c(1)
        idx <- which(rankVector==rank)
        numInd <- length(idx)
        
        while ((n+numInd)<=popsize){
                
                for (i in (n+1):(n+numInd)){
                        j <- i-n
                        nextpop$var1[[i]] <- combinepop$var1[[idx[j]]]
                        nextpop$obj[[i]] <- combinepop$obj[[idx[j]]]
                        nextpop$cons[[i]] <- combinepop$cons[[idx[j]]]
                        nextpop$rank[[i]] <- combinepop$rank[[idx[j]]]
                        nextpop$distance[[i]] <- combinepop$distance[[idx[j]]]
                        nextpop$prefDistance[[i]] <- combinepop$prefDistance[[idx[j]]]
                        nextpop$nViol[[i]] <- combinepop$nViol[[idx[j]]]
                        nextpop$violSum[[i]] <- combinepop$violSum[[idx[j]]]
                        
                        
                }
                n <- n + numInd
                rank <- rank + 1
                
                idx <- which(rankVector == rank)
                numInd <- length(idx)
        }
        
        # if the number of individuals in the next front plus the number of individuals in the current front is greater than the size of
        # population size, then select the best individuals by crowding distance 
        
        #         if (n < popsize){
        #                 j <- n
        #                 for (i in n+1 : popsize){
        #                      
        #                         nextpop$var1[[i]] <- combinepop$var1[[i]]
        #                         nextpop$obj[[i]] <- combinepop$obj[[i]]
        #                         nextpop$cons[[i]] <- combinepop$cons[[i]]
        #                         nextpop$rank[[i]] <- combinepop$rank[[i]]
        #                         nextpop$distance[[i]] <- combinepop$distance[[i]]
        #                         nextpop$prefDistance[[i]] <- combinepop$prefDistance[[i]]
        #                         nextpop$nViol[[i]] <- combinepop$nViol[[i]]
        #                         nextpop$violSum[[i]] <- combinepop$violSum[[i]]
        #                         j <- j+1
        #                 }
        #                 n<-j
        #         }
        
        if (n < popsize && !is.null(idx)){
                
                distance <- matrix(c(0),length(idx),2)
                for (i in 1 : length(idx)){
                        distance[i,1] <- combinepop$distance[idx[i]] 
                        distance[i,2] <- idx[i]
                }
                
                #                 distance <- t(matrix(c(distance,idx),,))
                #                 distance <- distance[order(distance[,1]),]
                #                 distance <- matrix(distance,,2)
                distance <- apply(distance,2,rev)
                distance <- distance[order(distance[,1],distance[,2],decreasing=TRUE),]
                if (n<popsize){
                        idxSelect <- distance[1:(popsize-n),2]
                        
                        for (i in 1 : length(idxSelect)){
                                
                                nextpop$var1[[n+i]] <- combinepop$var1[[idxSelect[i]]]
                                nextpop$obj[[n+i]] <- combinepop$obj[[idxSelect[i]]]
                                nextpop$cons[[n+i]] <- combinepop$cons[[idxSelect[i]]]
                                nextpop$rank[[n+i]] <- combinepop$rank[[idxSelect[i]]]
                                nextpop$distance[[n+i]] <- combinepop$distance[[idxSelect[i]]]
                                nextpop$prefDistance[[n+i]] <- combinepop$prefDistance[[idxSelect[i]]]
                                nextpop$nViol[[n+i]] <- combinepop$nViol[[idxSelect[i]]]
                                nextpop$violSum[[n+i]] <- combinepop$violSum[[idxSelect[i]]]
                        }
                }
        }
        
        #         b<-unique(nextpop$obj)
        #         obj <- unlist(nextpop$obj)
        #         obj <- t(matrix(obj,2,))
        #         if(length(b)<popsize){
        #                 
        #                 for (i in 1:length(b)){
        #                         
        #                         c <- which(obj[,1]==b[[i]][1])
        #                         
        #                         for (j in 2:(length(c))){
        #                                 
        #                                 ku <- round(runif(1,0,1)*length(combinepop$var1))
        #                                 
        #                                 if (ku==0){ku<-1}
        #                                 
        #                                 nextpop$var1[[c[j]]] <- combinepop$var1[[ku]]
        #                                 nextpop$obj[[c[j]]] <- combinepop$obj[[ku]]
        #                                 nextpop$cons[[c[j]]] <- combinepop$cons[[ku]]
        #                                 nextpop$rank[[c[j]]] <- combinepop$rank[[ku]]
        #                                 nextpop$distance[[c[j]]] <- combinepop$distance[[ku]]
        #                                 nextpop$prefDistance[[c[j]]] <- combinepop$prefDistance[[ku]]
        #                                 nextpop$nViol[[c[j]]] <- combinepop$nViol[[ku]]
        #                                 nextpop$violSum[[c[j]]] <- combinepop$violSum[[ku]]
        #                         }
        #                         
        #                 }
        #                 
        #                 length(b)<-popsize
        #         }
        
        return(nextpop)
}