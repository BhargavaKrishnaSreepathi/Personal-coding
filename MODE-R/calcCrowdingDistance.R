calcCrowdingDistance <- function (combinepop,front){
        
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/ndsort.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/RunMyMOO.R')
        # Calculate the crowding distance
        
        pop <- combinepop
        front1 <- front
        numObj <- length(pop$obj[[1]])
        
        N <- length(pop$var1)
        for (i in 1 : N){
                
                pop$rank[[i]] <- c(0)
                pop$distance[[i]] <- c(0)
                pop$prefDistance[[i]] <- c(0)
                
        }
        
        
        for (fid in 1 : length(front1)){
                var1 <- c()
                obj <- c()
                cons <- c()
                rank <-c()
                distance <- c()
                prefDistance <- c()
                nViol <- c()
                violSum <- c()
                
                idx <- front1[[fid]]
                #                 idx <- idx[[1]]  # done for temporary solving
                if (!is.null(idx)){
                        for (i in 1:length(idx)){
                                
                                var1[[i]] <- pop$var1[[idx[i]]]
                                obj[[i]] <- pop$obj[[idx[i]]]
                                cons[[i]] <- pop$cons[[idx[i]]]
                                rank[[i]] <- pop$rank[[idx[i]]]
                                distance[[i]] <- pop$distance[[idx[i]]]
                                prefDistance[[i]] <- pop$prefDistance[[idx[i]]]
                                nViol[[i]] <- pop$nViol[[idx[i]]]
                                violSum[[i]] <- pop$violSum[[idx[i]]]
                                
                        }
                        
                        frontPop <- list(var1=var1,obj=obj,cons=cons,rank=rank,distance=distance,prefDistance=prefDistance,nViol=nViol,violSum=violSum)
                        
                        #                 frontPop <- list(var1=pop$var1[[idx]],obj=pop$obj[[idx]],cons=pop$cons[[idx]],rank=pop$rank[[idx]],distance=pop$distance[[idx]],prefDistance=pop$prefDistance[[idx]],nViol=pop$nViol[[idx]],violSum=pop$violSum[[idx]])
                        #                 frontPop <- pop[[idx]]
                        
                        numInd <- length(idx)
                        
                        obj <- unlist(obj)
                        obj <- t(matrix(obj,2,))
                        obj <- cbind((obj),idx)
                        obj <- matrix(obj,nrow=numInd,ncol=3)
                        
                        for (m in 1 : numObj){
                                
                                obj <- obj[order(obj[,1],obj[,2]),]
                                obj <- matrix(obj,nrow=numInd,ncol=3)
                                #                         print(obj)
                                #                         print(m)
                                colIdx <- numObj +1
                                pop$distance[[obj[1,colIdx]]] <- 4
                                pop$distance[[obj[numInd,colIdx]]] <- 4
                                
                                abh<-c(2) 
                                bbh <- c(1)
                                
                                #                         if (2 < (numInd-1)){
                                #                         for (bh in 2 : (numInd-1)){
                                #                                 
                                #                                 if (obj[bh,1]==obj[1,1] && obj[bh,2]==obj[1,2]){
                                #                                         
                                #                                         pop$distance[[obj[bh,colIdx]]] <- 4
                                #                                         abh<-abh+1
                                #                                 }
                                #                         }
                                # 
                                #                         for (bh in 2 : (numInd-1)){
                                #                                 
                                #                                 if (obj[bh,1]==obj[numInd,1] && obj[bh,2]==obj[numInd,2]){
                                #                                         
                                #                                         pop$distance[[obj[bh,colIdx]]] <- 4
                                #                                         bbh <- bbh+1
                                #                                 }
                                #                         }
                                #                         }
                                
                                minobj <- obj[1,m]     # the maximum of the objective m
                                maxobj <- obj[numInd,m] # the minimum of the objective m
                                
                                #                         if (abh < 2){ abh <- c(2)}
                                #                         if (bbh < 1){ bbh <- c(1)}
                                
                                if (abh < (numInd-bbh)){
                                        for (i in abh : (numInd-bbh)){
                                                
                                                id <- obj[i,colIdx]
                                                if(maxobj==minobj){
                                                        pop$distance[[id]] <- 0
                                                        #                                 }else if (minobj==obj[i,m] || maxobj==obj[i,m]){
                                                        #                                         pop$distance[[id]] <- Inf
                                                }else {
                                                        pop$distance[[id]] <- pop$distance[[id]] + (obj[i+1,m] - obj[i-1,m])/(maxobj - minobj)
                                                }
                                        }
                                }
                        }
                }
        }
        # 
        #         for (i in 1:N){
        #                 
        #                 for (j in 2:N){
        #                         
        #                         if (pop$obj[[i]][[1]] == pop$obj[[j]][[1]] && pop$obj[[i]][[2]] == pop$obj[[j]][[2]]){
        #                                 
        #                                 pop$distance[[j]] <- pop$distance[[i]]
        #                         }
        #                 }
        #         }
        
        return(pop$distance)
        
        
}