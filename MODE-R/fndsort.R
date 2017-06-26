## fast-non-dominated-sort

fnd_sort <- function(x) {
        
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/dominance.R')
        
        x <- unlist(PopArray$var1)
        x <- t(matrix(x,87,2*Numpop))
        npop <- length(x[,1])
        Sp <- vector(mode='list',npop)
        np <- vector(mode='numeric',npop)
        Fi <- vector(mode='list',npop)
        for (p in 1:npop) {
                for (q in 1:npop) {
                        dom_flag <- dominance(x[p,],x[q,])
                        if (dom_flag == 1) # if p dominates q
                        { Sp[[p]] <- c(Sp[[p]],q) }
                        if (dom_flag == 2) # if q dominates p
                        { np[p] <- np[p] + 1 }
                }  ## end of q loop
                if (np[p] == 0)
                { pRank = 1
                  Fi[[pRank]] <- c(Fi[[pRank]],p) }
        }  ## end of p loop
        #####################################################
        ## FILL OTHER FRONTS
        #####################################################
        front_count <- 1
        nq <- np
        while (length(Fi[[front_count]]) != 0) {
                Q <- vector()
                for (p in Fi[[front_count]]) {
                        for (q in Sp[[p]]) {
                                nq[q] <- nq[q] -1
                                if (nq[q] == 0) {
                                        qRank <- front_count + 1
                                        Q <- c(Q,q) }
                        }  ## end q loop
                }  ## end p loop
                front_count <- front_count + 1
                Fi[[front_count]] <- Q
        }  ## end while loop
        
        returnVal <- Fi[1:(front_count-1)]
        return(returnVal)
}  ## end of fnd_sort function

#############################################################################
## fast-non-dominated-sort

dominance <- function(ind1,ind2)  {## ind1 and ind2 are rows in the population matrix
        ##  set for minimization (change "<" in next 4 lines to maximize)
        pdq1 <- as.numeric(ind1 <= ind2)    # no worse in all objectives
        pdq2 <- as.numeric(ind1 < ind2)
        qdp1 <- as.numeric(ind2 <= ind1)    # no worse in all objectives
        qdp2 <- as.numeric(ind2 < ind1)
        ###    cat(pdq1,pdq2,qdp1,qdp2,"\n",file="pdq.txt",append=T)
        if (sum(pdq1) > 0 && sum(pdq2) > 0)  # if p dominates q
        {dflag <- 1}
        if (sum(qdp1) > 0 && sum(qdp2) > 0)  # if q dominates p
        {dflag <- 2}
        if (sum(pdq1) > 0 && sum(qdp1) > 0)  # p non-inferior to q
        {dflag <- 3}
        returnVal <- dflag
        return(returnVal)
}
