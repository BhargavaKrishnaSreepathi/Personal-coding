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