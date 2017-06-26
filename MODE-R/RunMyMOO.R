RunMyMOO <- function(x){
        
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/check_SM.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/temp_C.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/temp_H.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/variables_file_ex3.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/fn_IMODE.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/fill_f.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/crowding.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/extractPop.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/selectOp.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/calcCrowdingDistance.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/ndsort.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/calcCrowdingDistance.R')
        
        var1<-NULL
        obj<-NULL
        cons<-NULL
        rank<- NULL
        distance <- NULL
        prefDistance <- NULL
        PopArray <- NULL
        ks <- c(0)
        sum_a <- c(0)
        sum_l <- c(0)
        sum_ll <- c(0)
        newpop<-c()
        NumF <- c(2)
        NumX <- length(VLB)
        NumC <- 2*NHEX
        bestValue <- c(1E+200)
        
        XMin <- VLB
        XMax <- VUB
        FWeights <- c(1,1)
        counter1 <- c(1)
        counter2 <- c(1)
        counter3 <- c(1)
        
        N <- Numpop
        var1 <- vector(mode='list',N)
        obj <- vector(mode='list',N)
        cons <- vector(mode='list',N)
        rank <- vector(mode='numeric',N)
        distance <- vector(mode='numeric',N)
        prefDistance <- vector(mode='numeric',N)
        nViol <- vector(mode='numeric',N)
        violSum <- vector(mode='numeric',N)
        
        for (i in 1 : N){
                for(j in 1:NumX){
                        var1[[i]] <- c(var1[[i]],0)
                }
        }
        
        for (i in 1 : N){
                for(j in 1:NumF){
                        obj[[i]] <- c(obj[[i]],0)
                }
        }
        
        for (i in 1 : N){
                for(j in 1:NumC){
                        cons[[i]] <- c(cons[[i]],0)
                }
        }
        
        
        
        PopArray1 <- matrix(c(1),nrow=Numpop,ncol=NumX)
        
        # initialization of the population and calculation of its objective functions
        
        for (counter3 in 1:Numpop){                
                
                for (counter1 in 1:NumX){
                        
                        X <- (XMax[counter1] - XMin[counter1])*runif(1,0,1) + XMin[counter1]
                        
                        if (dVtype[counter1] == 1) {
                                
                                PopArray1[counter3,counter1] <- round(X)
                        }else {
                                
                                PopArray1[counter3,counter1] <- X
                        }
                        
                        if (PopArray1[counter3,counter1] < XMin[counter1] || PopArray1[counter3,counter1] > XMax[counter1]){
                                
                                PopArray1[counter3,counter1] <- (XMax[counter1] - XMin[counter1])*runif(1,0,1) + XMin[counter1]
                                
                        }
                        
                        #                 print(X)
                }
                
                var1[[counter3]] <- PopArray1[counter3,]
                
                fn_IMODE(var1[[counter3]])
                
                obj[[counter3]] <- c(investmentcost,utilitycost)
                
                currentvalue <- c(0)
                Npen <- c(0)
                
                # this is for constraint check, here there is only greater than the Delta, so the constraints which are positive are in violation and the ones which are zero are satisfying
                for (counter1 in 1:NumC){
                        
                        if (Cons[counter1] >0) {
                                
                                currentvalue <- currentvalue + Cons[counter1]
                                Npen <- Npen + 1
                                
                        }
                        
                }
                cons[[counter3]] <- Cons
                violSum[counter3] <- currentvalue 
                nViol[counter3] <- Npen
                
                
        }
        
        PopArray <- list(var1=var1,obj=obj,cons=cons,rank=rank,distance=distance,prefDistance=prefDistance,nViol=nViol,violSum=violSum)
        
        #                         newlist <- ndsort(PopArray)
        #                         distance <- calcCrowdingDistance(PopArray,newlist$front1)
        #                         PopArray$distance <- distance[]
        #                         PopArray$rank <- newlist$rank
        
        obj<-unlist(PopArray$obj)
        obj <- t(matrix(obj,2,Numpop))
        
        plot(obj[,1],obj[,2],main=1)
        # Initialize the tabu list - randomly
        TL <- matrix(c(1),nrow=TLS,ncol=NumX)
        itl <- 1
        for (i in 1 : TLS){
                
                for (j in 1 : NumX){
                        
                        itl <- round(0.5 + Numpop * runif(1,0,1))
                        TL[i,j] <- PopArray$var1[[itl]][[j]]
                }
        }
        
        par_c <- c(0.1)
        ktl <- c(1)
        ST_par <- c(0)
        
        NFE <- c(0)
        HGS <- c(0)
        new_individual <- matrix(c(1),nrow=1,ncol=NumX)
        new_individual2 <- matrix(c(1),nrow=1,ncol=NumX)
        Store_FCR <- matrix(c(1),nrow=2+NumF,ncol=Numpop)
        ComPF <- matrix(c(1),nrow=Numpop,ncol=2*NumF)
        ngen <- c(0)
        # NSGA2 approach
        
        while (ngen < MaxGen) {
                
                ngen <- ngen + 1
                #                 kk <-c(1)
                
                # create new population
                newpop <- PopArray
                
                #                 for (i in 1:Numpop){
                #         
                #                         newpop$var1[[i]] <- PopArray$var1[[i]]
                #                         newpop$obj[[i]] <- PopArray$obj[[i]]
                #                         newpop$cons[[i]] <- PopArray$cons[[i]]
                #                         newpop$rank[[i]] <- PopArray$rank[[i]]
                #                         newpop$distance[[i]] <- PopArray$distance[[i]]
                #                         newpop$prefDistance[[i]] <- PopArray$prefDistance[[i]]
                #                         newpop$nViol[[i]] <- PopArray$nViol[[i]]
                #                         newpop$violSum[[i]] <- PopArray$violSum[[i]]
                #                 }
                #                 newpop <- selectOp(PopArray)
                #                 newpop <- crossoverOp(newpop)
                #                 newpop <- mutationOp(newpop)
                
                if (ngen > 1) {
                        
                        sum_a <- c(0)
                        sum_l <- c(0)
                        sum_ll <- c(0)
                        if(ks!=0){
                                for (kk in 1 : ks){
                                        
                                        
                                        sum_a <- sum_a + Store_FCR[1,kk]
                                        sum_l <- sum_l + Store_FCR[2,kk]
                                        sum_ll <- sum_ll + (Store_FCR[2,kk])^2
                                        
                                }
                        }else{
                                sum_a <- sum_a
                                sum_l <- sum_l
                                sum_ll <- sum_ll
                        }
                        # sum_a <- sum_a + Store_FCR[1,kk]
                        # sum_l <- sum_l + Store_FCR[2,kk]
                        # sum_ll <- sum_ll + (Store_FCR[2,kk])^2
                        
                        
                        
                        if (ks == 0){
                                
                                ks <- c(1)
                                
                        }
                        
                        SCR <- sum_a/ks
                        
                        if (sum_l == 0) {
                                
                                sum_l <- c(1)
                                
                        }
                        
                        SF <- sum_ll/sum_l
                        
                        New_mu_CR <- (1 - par_c) * mu_CR + par_c * SCR
                        New_mu_F <- (1 - par_c) * mu_F + par_c * SF
                        mu_CR <- New_mu_CR
                        mu_F <- New_mu_F
                        
                }
                
                kont <- c(0)
                
                for (counter3 in 1 : Numpop){
                        
                        r1 <- c(1)
                        r2 <- c(1)
                        r3 <- c(1)
                        
                        while (r1 == r2 || r2 == r3 || r1 ==r3 || r1 <= 0 || r2 <= 0 || r3 <= 0){
                                
                                r1 <- floor(((Numpop-1+1) * runif(1,0,1))+1)
                                r2 <- floor(((Numpop-1+1) * runif(1,0,1))+1)
                                r3 <- floor(((Numpop-1+1) * runif(1,0,1))+1)
                                
                        }
                        
                        XOverP <- 0.5
                        MutP <- 0.5
                        if (XOverP < 0) {
                                
                                XOverP <- c(0.01)
                                
                        }
                        
                        if (XOverP > 1) {
                                
                                XOverP <- c(0.99)
                                
                        }
                        
                        if (MutP < 0) {
                                
                                MutP <- c(0.01)
                                
                        }
                        
                        if (MutP > 2) {
                                
                                MutP <- c(1.99)
                                
                        }
                        
                        for (counter1 in 1 : NumX) {
                                
                                XRand <- runif(1,0,1)
                                Jrand <- round((NumX-1+1) * runif(1,0,1)+1)
                                
                                if (XRand < XOverP || Jrand == counter1) {
                                        
                                        new_individual[1,counter1] <- newpop$var1[[r1]][[counter1]] + MutP * (newpop$var1[[r2]][[counter1]]-newpop$var1[[r3]][[counter1]])
                                        
                                } else {
                                        
                                        new_individual[1,counter1] <- newpop$var1[[counter3]][[counter1]]
                                }
                                
                                if (new_individual[1,counter1] < XMin[counter1] || new_individual[1,counter1] > XMax[counter1]){
                                        
                                        new_individual[1,counter1] <- (XMax[counter1] - XMin[counter1]) * runif(1,0,1) + XMin[counter1]
                                        
                                }
                                
                                if (dVtype[counter1]==1){
                                        
                                        new_individual[1,counter1] <- round(new_individual[1,counter1])
                                }
                                
                        }
                        
                        # Normalized Euclidean distance calculation
                        
                        ED <- c(1000)
                        
                        for (i in 1 : TLS) {
                                
                                ED0 <- c(0)
                                
                                for (j in 1 : NumX){
                                        
                                        if (dVtype[j]==1){
                                                
                                                ED0 <- ED0 + ((new_individual[1,j] - TL[i,j])/(XMax[j] - XMin[j]))^2
                                                
                                        }else{
                                                
                                                ED0 <- ED0 + ((new_individual[1,j] - TL[i,j])/(XMax[j] - XMin[j]))^2
                                        }
                                        
                                        
                                }
                                
                                Ed_new <- (ED0)^0.5
                                
                                if (Ed_new < ED) {
                                        
                                        ED <- Ed_new
                                }
                                
                        }
                        
                        if (ED > TR ) {
                                
                                kont <- kont + 1
                                Store_FCR[1,kont] <- XOverP
                                Store_FCR[2,kont] <- MutP
                                
                                for (pp in 1 : NumF) {
                                        
                                        Store_FCR[2+pp,kont] <- FWeights[pp] * newpop$obj[[counter3]][[pp]]
                                        
                                }
                                
                                for (j in 1 : NumX) {                       
                                        
                                        
                                        
                                        if (dVtype[j] ==1) {
                                                
                                                new_individual[1,j] <- round(new_individual[1,j])
                                                
                                        }
                                        TL[ktl,j] <- new_individual[1,j]        #Update Tabu list
                                        
                                        PopArray$var1[[counter3]][[j]] <- new_individual[1,j]
                                        
                                }
                                
                                if (ktl < Numpop/2) {
                                        
                                        ktl <- ktl + 1
                                }else{
                                        ktl <- c(1)
                                        
                                }
                                
                                NFE <- NFE + 1
                                fn_IMODE(PopArray$var1[[counter3]])
                                
                                PopArray$obj[[counter3]] <- c(investmentcost,utilitycost)
                                
                                
                                currentvalue <-c(0)
                                Npen <- c(0)
                                
                                for (counter1 in 1:NumC){
                                        
                                        if (Cons[counter1] >0) {
                                                
                                                currentvalue <- currentvalue + Cons[counter1]
                                                Npen <- Npen + 1
                                                
                                        }
                                        
                                }
                                PopArray$cons[[counter3]] <- Cons
                                PopArray$violSum[[counter3]] <- currentvalue
                                PopArray$nViol[[counter3]] <- Npen
                                
                                
                        }else{
                                
                                PopArray$violSum[[counter3]] <- c(10000)
                                PopArray$nViol[[counter3]] <- NumC
                                
                                
                        }
                        
                }
                combinepop <- c()
                
                for (i in 1:Numpop){
                        
                        combinepop$var1[[i]]<-newpop$var1[[i]]
                        combinepop$var1[[Numpop+i]] <- PopArray$var1[[i]]
                        combinepop$obj[[i]]<-newpop$obj[[i]]
                        combinepop$obj[[Numpop+i]] <- PopArray$obj[[i]]
                        combinepop$cons[[i]]<-newpop$cons[[i]]
                        combinepop$cons[[Numpop+i]] <- PopArray$cons[[i]]
                        combinepop$rank[[i]]<-newpop$rank[[i]]
                        combinepop$rank[[Numpop+i]] <- PopArray$rank[[i]]
                        combinepop$distance[[i]]<-newpop$distance[[i]]
                        combinepop$distance[[Numpop+i]] <- PopArray$distance[[i]]
                        combinepop$prefDistance[[i]]<-newpop$prefDistance[[i]]
                        combinepop$prefDistance[[Numpop+i]] <- PopArray$prefDistance[[i]]
                        combinepop$nViol[[i]]<-newpop$nViol[[i]]
                        combinepop$nViol[[Numpop+i]] <- PopArray$nViol[[i]]
                        combinepop$violSum[[i]]<-newpop$violSum[[i]]
                        combinepop$violSum[[Numpop+i]] <- PopArray$violSum[[i]]
                }
                
                newlist <- ndsort(combinepop)
                distance <- calcCrowdingDistance(combinepop,newlist$front1)
                combinepop$distance <- distance[]
                combinepop$rank <- newlist$rank
                nextpop <- extractPop(combinepop)
                PopArray <- nextpop
                #                 PopArray <- extractPop(PopArray)
                
                obj<-unlist(PopArray$obj)
                obj <- t(matrix(obj,2,Numpop))
                
                for (ge in seq(50,MaxGen,by=50)){
                        
                        if (ge==ngen){
                                
                                plot(obj[,1],obj[,2],main=ngen)
                        }
                }
                
                for (ge1 in seq(50,MaxGen,by=50)){
                        
                        if (ge1==ngen){
                                
                                a<-matrix(c(0),Numpop,NumX)
                                b<-matrix(c(0),Numpop,NumF)
                                c<-matrix(c(0),Numpop,2)
                                for(i in 1:Numpop){
                                        
                                        a[i,] <- PopArray$var1[[i]]
                                        b[i,] <- PopArray$obj[[i]]
                                        c[i,1] <- PopArray$nViol[[i]]
                                        c[i,2] <- PopArray$violSum[[i]]
                                }
                                d<-cbind(a,b,c)
                                setwd("C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/Results")
                                write.table(d,paste(ngen,".txt",sep=""))
                        }
                }
                
                
                ks <- c(0)
                #         print(kk)
                
                if (kont !=0){
                        for (i in 1 : kont){
                                
                                for (j in 1 : Numpop){
                                        
                                        P_break <- c(0)
                                        
                                        if (Store_FCR[2+1,i] == PopArray$obj[[j]][1] && Store_FCR[2+2,i] == PopArray$obj[[j]][2]){
                                                
                                                ks <- ks+1
                                                Store_FCR[1,ks] <- Store_FCR[1,i]
                                                Store_FCR[2,ks] <- Store_FCR[2,i]
                                                P_break <- c(1)
                                                Pj <- c()
                                                if (P_break == 1){
                                                        Pj <- break
                                                }
                                                Pj
                                                
                                        }
                                        Pj <-c()
                                        if (P_break ==1){
                                                Pj <- break
                                        }
                                        Pj
                                        
                                }
                        }
                }
                
        }
        
        # starting of generation loop

        PopArray <<- PopArray
        
         

        
        
        
        
        print(PopArray)
}