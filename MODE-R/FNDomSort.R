FNDomSort <- function (NumF, NumO, NumC, MyPopArray, Numpop, NDCUT_I){
        
        NDCUT_I <- c(0)
        aColNum <- NumF + NumO + NumC
        MyPopArray <- cbind(PopArray,rep(1,Numpop))
        tGenRange2 <- MyPopArray
        
        for (i in 1 : Numpop) {
                
                print(i)
                 while (MyPopArray[i,aColNum+2]!=0){                
                        
                        j<-c(0)
                        MyPopArray[i,aColNum+3] <- Numpop*2
                        
                        i <- i + 1
                        if (i == Numpop+1){
                                
                                j <- break
                        }
                        j
                        print(i)
                                       
                }
                if (i == Numpop+1) {
                        
                        j <- break
                        
                }
                
                j
                print(i)
                NdomCount <- c(0)
                
                for (ii in 1 : Numpop){
                        
                        j <- c(0)
                        
                        if (ii == 1) {
                                
                                ii <- ii + 1
                                
                        }
                        
                        if (ii == Numpop+1){
                                
                                j <- break
                                
                        }
                        j
                        
                        while (MyPopArray[ii,aColNum+2] != 0){
                                
                                ii <- ii+1
                                
                                if (ii == Numpop+1){
                                        
                                        j <- break
                                }
                                j
                        }
                        if (ii == Numpop+1){
                                
                                j <- break
                        }
                        j
                        switchequal <- c(0)
                        switchless <- c(0)
                        
                        for (iii in 1 : NumO) {
                                
                                if (MyPopArray[ii,NumF+iii] <= MyPopArray[i,NumF+iii]){
                                        
                                        switchequal <- switchequal + 1
                                        
                                }
                                
                                if (MyPopArray[ii,NumF+iii] < MyPopArray[i,NumF+iii]){
                                        
                                        switchless <- switchless + 1
                                        
                                }
                                
                        }
                        
                }
                
                if (Ndomcount == 0){
                        
                        NDCUT_I <- NDCUT_I +1
                        
                }
        
        }
        
        tGenRange2 <- MyPopArray
       
        
}