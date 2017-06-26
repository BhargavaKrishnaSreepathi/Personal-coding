
temp_C <- function(x,AlfaC,HnodT,CnodT,CstreamN,CnodeN,NHEX){
        
#         source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/check_SM.R', echo=TRUE)
        
        
#         x<-c(1,  2, -1,  2,  2,  3,  4,  6 , 1,  2,  1,  2,  2,  3,  2,  3)
#         x<-check_SM(x)
#         AlfaC<-c(1,1,1,1)
#         CstreamN<-1
#         CnodeN<-2
#         HnodT<-t(matrix(c(298,290,290,290,250,245,245,245),4,2))
#         CnodT<-t(matrix(c(14,20,20,20,25,25,25,25),4,2))
#         #NHEX<-4
        Ex<-0
        
        # Calculate the stream temperature at each node
        # See if the node has a exchanger connected to it, if yes then calculate
        # the temperature using alfah (equation given by Jezowski), else assign the
        # temperature of the inlet hot stream
        
        SM <- x
        Temp_In <- CnodT[CstreamN,CnodeN]
        
        for (i in 1:NHEX){
                
                if (CstreamN == SM[i,3] && CnodeN == SM[i,4]){
                        
                        Ex <- i
                        break
                        
                }
        }
        
        if (Ex == 0){
                
                Temp_C <- Temp_In                
        }
        else {
                
                alpha <- AlfaC[Ex]
                hstream <- SM[Ex,1]
                hnode <- SM[Ex,2]
                
                if (hnode == 0){
                        
                        Temp_C <- Temp_In
                        
                }
                else {
                        
                        Temp_H <- HnodT[hstream,hnode]
                        Temp_C <- (alpha*Temp_In) + (1-alpha)*Temp_H
                }
        }
#         print(Temp_C)
        temp_C <-Temp_C
}