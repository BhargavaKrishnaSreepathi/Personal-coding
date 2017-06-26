calcDominationMatrix <- function(nViol,violSum,obj){
        
        # calculate the domination matrix which specified the domination relation
        # between two individual using constrained domination
        # Return
        # domMat[N,N] : domination matrix
        # domMat[p,q] = 1 : p dominates q
        # domMat[p,q] = -1 : q dominates p
        # domMat[p,q] = 0 : non dominate
        
        N <- nrow(obj)
        numObj <- ncol(obj)
              
        domMat <- matrix(c(0),N,N)
        
        for (p in 1 : (N-1)){
#                 print(p)
                for (q in (p+1) : N){
                        
                        # 1. p and q are both feasible
                        
                        if (nViol[p] == 0 && nViol[q] == 0){
                                
                                pdomq <- FALSE
                                qdomp <- FALSE
                                
                                for (i in 1:numObj){
                                        
                                        if (obj[p,i] < obj[q,i]){
                                                
                                                pdomq <- TRUE
                                                
                                        }else if (obj[p,i] > obj[q,i]){
                                                
                                                qdomp <- TRUE
                                                
                                        }
                                }
                                
                                if (pdomq && !qdomp){
                                        
                                        domMat[p,q] <- c(1)
                                        
                                }else if (!pdomq && qdomp){
                                        
                                        domMat[p,q] <- c(-1)
                                        
                                }
                        }else if (nViol[p] == 0 && nViol[q] != 0){ # p is feasible, q is infeasible
                                
                                domMat[p,q] <- c(1)
                                
                        }else if (nViol[p] != 0 && nViol[q] == 0){ # p is infeasible, q is feasible
                                
                                domMat[p,q] <- c(-1)
                                
                        }else {
                                
                                # p and q are both infeasible
                                if (violSum[p] < violSum[q]){
                                        
                                        domMat[p,q] <- c(1)
                                        
                                }else if (violSum[p] > violSum[q]){
                                        
                                        domMat[p,q] <- c(-1)
                                        
                                }
                        }
                }
#                 print(p)
        }
        
      
        
        domMat <- domMat-t(domMat)
        
}