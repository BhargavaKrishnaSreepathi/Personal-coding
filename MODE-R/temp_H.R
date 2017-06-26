
temp_H <- function(x,AlfaH,HnodT,CnodT,HstreamN,HnodeN,NHEX){
        
#         source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/check_SM.R', echo=TRUE)
        
        
#         x<-c(1,  2, -1,  2,  2,  3,  4,  6 , 1,  2,  1,  2,  2,  3,  2,  3)
#         x<-check_SM(x)
#         AlfaH<-c(1,1,1,1)
#         HstreamN<-1
#         HnodeN<-2
#         HnodT<-t(matrix(c(298,290,290,290,250,245,245,245),4,2))
#         CnodT<-t(matrix(c(14,20,20,20,25,25,25,25),4,2))
#         NHEX<-4
        Ex<-0
#         print(AlfaH)
        
        # Calculate the stream temperature at each node
        # See if the node has a exchanger connected to it, if yes then calculate
        # the temperature using alfah (equation given by Jezowski), else assign the
        # temperature of the inlet hot stream
#         print(HnodeN)
#         print(HstreamN)
#         print(x)
        SM <- x
        Temp_In <- HnodT[HstreamN,HnodeN]
        
        for (i in 1:NHEX){
                
                if (HstreamN == SM[i,1] && HnodeN == SM[i,2]){
                        
                        Ex <- i
                        break
                        
                }
        }
        
        if (Ex == 0){
                
                Temp_H <- Temp_In                
        }
        else {
                
                alpha <- AlfaH[Ex]
                cstream <- SM[Ex,3]
                cnode <- SM[Ex,4]
                
                if (cnode == 0){
                        
                        Temp_H <- Temp_In
                        
                }
                else {
                        
                        Temp_C <- CnodT[cstream,cnode]
                        Temp_H <- (alpha*Temp_In) + (1-alpha)*Temp_C
                }
#                 print(alpha)
        }
        
#         print(Temp_H)
        temp_H <-Temp_H
}