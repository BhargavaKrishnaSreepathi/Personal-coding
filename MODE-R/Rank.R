Rank <- function(NumF,NumO,NumC,cGenRange,Numpop){
        
        aColNum <- c()
        aColNum <- NumF + NumO + NumC
        
        ii <- c(1)
        
        for (i in 1:Numpop){
                
                if (cGenRange[i-1,aColNum+2] > cGenRange[i-2,aColNum+2]) {
                        
                        ii <- ii + 1
                        
                }
                
                cGenRange[i-1,aColNum+3] <- ii
                
                
        }
        
        
}