

check_SM <- function(x){
        
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/variables_file_ex3.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/fn_IMODE.R')       
#         x<- c(9, 2, 3, 11, 1, 8, 3, 10, 2, 2, 3, 4, 1, 1, 1, 2, 4, 7, 3, 1, 6, 6, 3, 11, 1, 6, 3, 11, 1, 3, 3, 4, 9, 5, 4, 2, 11, 2, 1, 10, 11, 0, 1, 6, 4, 4, 4, 0, 4, 7, 1, 4, 12, 1, 3, 2, 2, 8, 1, 11, 1273.62044531526, 35.4766843839836, 829.247598855291, 818.060445340785, 6.59536944642476, 625.515040920701, 301.476850670235, 235.007434614707, 429.515167880085, 940.2647014027, 1148.63676090501, 989.144507654556, 934.604804462067, 255.992116870015, 327.717450519284, 0.845860439485781, 0.620651194231628, 0.81516034261892, 0.789592614090366, 0.262723685462704, 0.467861149799087, 0.796793236355329, 0.271444554544165, 0.603880653141923, 0.370322598732215, 0.447683074462249, 0.75026973992943, 109941.223444849, 15875816.1600962, 0, 0)
        
        x<-x[1:(4*NHEX)]
        #print(x)
        y<-matrix(x,4,NHEX)
        SM<-t(y)
        #print(y)
        for (i in 1:NHEX){
                
                # Checking the element in SM doesn't exceed the number of hot streams possible
                # Hot Stream Check
                
                if (SM[i,2]==0 || SM[i,4]==0){
                        
                        SM[i,2]<-c(0)
                        SM[i,4]<-c(0)
                }
                
                if (SM[i,1] > length(NODH) || SM[i,1]<=0){
                        SM[i,1]<-sample(1:length(NODH),1)                        
                }
                
                #COld stream check
                
                if (SM[i,3] > length(NODC) || SM[i,3]<=0){
                        SM[i,3]<-sample(1:length(NODC),1)                        
                }
                
                # positive node check
                
                for (j in 1:length(NODH)){
                        if (SM[i,2]<0){
                                SM[i,2]<-sample(0:NODH[j],1)
                        }
                }
                
                for (j in 1:length(NODC)){
                        if (SM[i,4]<0){
                                SM[i,4]<-sample(0:NODC[j],1)
                        }
                }
                
                #Hot stream node check
                
                for (j in 1:length(NODH)){
                        if (SM[i,1]==j){
                                if (SM[i,2] > NODH[j] || SM[i,2]<0){
                                        SM[i,2]<-sample(0:NODH[j],1)
                                }
                        }
                }
                
                # Cold stream node check
                
                for (j in 1:length(NODC)){
                        if (SM[i,3]==j){
                                if (SM[i,4] > NODC[j] || SM[i,4]<0){
                                        SM[i,4]<-sample(0:NODC[j],1)
                                }
                        }
                }
                
                #print(y)
                
                # if any node contains zero, then the hot and cold stream nodes are both turned zero
                
                if (SM[i,2]==0 || SM[i,4]==0){
                        SM[i,2]<-0
                        SM[i,4]<-0
                }
                
        }
        
        #print(y)
#         print(SM)
        
        # to check that same node is not assigned to different units
        
        for (i in 2:NHEX){
                for(j in 1:(i-1)){
                        #print(i,j)
                        if (SM[i,1] == SM[j,1] && SM[i,2] == SM[j,2]){
                                SM[i,2]<-0
                                SM[i,4]<-0
                        }
                        
                        if (SM[i,3] == SM[j,3] && SM[i,4] == SM[j,4]){
                                SM[i,2]<-0
                                SM[i,4]<-0
                        }
                }
        }
        
        # checks that there is no split or merge node on the same node as heat exchanger
        
        # hot stream
        
        if (!is.null(SPLH)){
                for (j in 1:nrow(SPLH)){
                        for (i in 1:NHEX){
                                if (SM[i,1]==SPLH[j,1]){
                                        if (SM[i,2] == SPLH[j,2] || SM[i,2] == SPLH[j,3]){
                                                SM[i,2]<-0
                                                SM[i,4]<-0
                                        }
                                }
                        }
                }
        }
        
        # cold stream
        
        if (!is.null(SPLC)){
                for (j in 1:nrow(SPLC)){
                        for (i in 1:NHEX){
                                if (SM[i,3]==SPLC[j,1]){
                                        if (SM[i,4] == SPLC[j,2] || SM[i,4] == SPLC[j,3]){
                                                SM[i,2]<-0
                                                SM[i,4]<-0
                                        }
                                }
                        }
                }
        }
        
#         print(SM)
        SM <<- SM
}
