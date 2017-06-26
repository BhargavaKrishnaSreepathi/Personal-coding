ndsort <- function(PopArray12){
        
        # initialize variables
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/calcDominationMatrix.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/variables_file_ex3.R')
        
        PopArray <- PopArray12
        N <- length(PopArray$var1)
        
        for (i in 1 : N){
                
                PopArray$rank[[i]] <- c(0)
                PopArray$distance[[i]] <- c(0)
                PopArray$prefDistance[[i]] <- c(0)
                
        }
        
        sp <- vector(mode='list',N)
        np <- vector(mode='numeric',N)
        
        ind <- list(sp=sp,np=np)
             
        # fast non-dominated sort
        
        nViol <- rep(c(0),N)
        violSum <- rep(c(0),N)
                
        for (i in 1 : N) {
                
                nViol[i] <- PopArray$nViol[[i]]
                violSum[i] <- PopArray$violSum[[i]]
                
        }
        
#         for (i in 1:N){
#                 for (j in 1:N){
#                         if (i != j && PopArray$obj[[i]][[2]] == PopArray$obj[[j]][[2]]){
#                                 if (PopArray$obj[[i]][[1]]==PopArray$obj[[j]][[1]]){
#                                         PopArray$obj[[j]][[2]]<- PopArray$obj[[j]][[2]] + .5
#                                 }
#                         }
#                 }
#         }
        
       obj <- unlist(PopArray$obj)
       obj <- t(matrix(obj,2,N))
        
        domMat <- calcDominationMatrix(nViol,violSum,obj) # domination matrix for efficiency
        
        # compute np and sp of each individual
        
        for (p in 1 : (N-1)){
                
                for (q in (p+1):N){
                        
                        if (domMat[p,q] == 1){
                                
                                # p dominate q
#                                 np[q] <- np[q] + 1
#                                 sp[[p]] <- c(sp[[p]],q)
                                ind$np[[q]] <- ind$np[[q]] + 1
                                ind$sp[[p]] <- cbind(ind$sp[[p]],q)
                                
                        }else if (domMat[p,q] == -1){
                                
                                # q dominate p
#                                 np[p] <- np[p] + 1
#                                 sp[[q]] <- c(sp[[q]],p)
                                ind$np[[p]] <- ind$np[[p]] + 1
                                ind$sp[[q]] <- cbind(ind$sp[[q]],p)
                                
                        }
                }
        }
        
        front1 <- vector(mode='list',N) # first front (Rank = 1)
        
        for (i in 1 : N) {
                
                if (ind$np[[i]] == 0){
                        
                        PopArray$rank[[i]] <- 1
                        front1[[1]] <- c(front1[[1]],i)
                }
        }
        
        front_count <- 1
        
        while (length(front1[[front_count]]) != 0) {
                Q <- vector()
                for (p in front1[[front_count]]) {
                        for (q in ind$sp[[p]]) {
                                ind$np[q] <- ind$np[q] -1
                                if (ind$np[q] == 0) {
                                        PopArray$rank[[q]] <- front_count + 1
                                        Q <- c(Q,q) }
                        }  ## end q loop
                }  ## end p loop
                front_count <- front_count + 1
#                 print(front_count)
#                 print(Q)
                front1[[front_count]] <- Q
        }  ## end while loop
        
        front1[[front_count]]<-c()
        newlist <- list(front1=front1,rank=PopArray$rank)
        return(newlist)
               
#         PopArray <<- PopArray
        # calculate the distance
#         crowding()
#         if (is.null(refPoints)){
#                 
#                 PopArray <- calcCrowdingDistance[PopArray,fron1]
#         }else {
#                 
#                 PopArray <- calcPreferenceDistance[PopArray,front1]
#         }
        
        
        
        
        
}