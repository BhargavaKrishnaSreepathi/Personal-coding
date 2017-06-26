
fn_IMODE <- function(x){
        
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/check_SM.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/temp_C.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/temp_H.R')
        source('C:/Users/a0091335/Desktop/Work/PhD/Manuscripts/Manuscript-MODE and NSGA/IMODE/HEN retrofitting using IMODE and R/variables_file_ex3.R')
        
#         x<- c(9, 2, 3, 11, 1, 8, 3, 10, 2, 2, 3, 4, 1, 1, 1, 2, 4, 7, 3, 1, 6, 6, 3, 11, 1, 6, 3, 11, 1, 3, 3, 4, 9, 5, 4, 2, 11, 2, 1, 10, 11, 0, 1, 6, 4, 4, 4, 0, 4, 7, 1, 4, 12, 1, 3, 2, 2, 8, 1, 11, 1273.62044531526, 35.4766843839836, 829.247598855291, 818.060445340785, 6.59536944642476, 625.515040920701, 301.476850670235, 235.007434614707, 429.515167880085, 940.2647014027, 1148.63676090501, 989.144507654556, 934.604804462067, 255.992116870015, 327.717450519284, 0.845860439485781, 0.620651194231628, 0.81516034261892, 0.789592614090366, 0.262723685462704, 0.467861149799087, 0.796793236355329, 0.271444554544165, 0.603880653141923, 0.370322598732215, 0.447683074462249, 0.75026973992943, 109941.223444849, 15875816.1600962, 0, 0)
        Area <- rep(0,NHEX)
        pos <- c()
        delT1 <- c()
        delT2 <- c()
        lmtd  <- c() 
        bhar <- c(0)
        penalty <- c(0)
        for (i in (4*NHEX):(5*NHEX)){
                
                Area[i-(4*NHEX)] <- x[i]
        
        }
        
        SM <-check_SM(x)
#         print(SM)
        
        Q1 <- c()
        Q2 <- c()
        Q3 <- c()
        Q  <- c()
        HEXload <- c()
        HnodoutT <- c()
        Hload <- c()
        CnodoutT <- c()
        Cload <- c()
        HEXarea <- c()
        Harea <- c()
        Carea <- c()
        allhex <- c()
        allhex_strt3 <- c()
        area_strt3 <- c()
   
        
        for (i in 1:NHEX){
                
                if (SM[i,2] == 0 || SM[i,4] ==0){
                        
                        Area[i] <-0
                        
                }
        }
        
        SpltrH <- c()
        SpltrC <- c()
        
        for (i in (5*NHEX+1):(5*NHEX+nrow(HstreamT))){
                
                SpltrH <- c(SpltrH,x[i])
                
        }
        
        for (i in ((5*NHEX+nrow(HstreamT))+1):(5*NHEX+nrow(HstreamT)+nrow(CstreamT))){
                
                SpltrC <- c(SpltrC,x[i])
        }
        
#         print(SpltrH)
#         print(SpltrC)
        
        convflag <- 1
        
        h <- 0
        g <- 0
        f <- 0
        bhar <- 0
        CPH <- c()
        CPC <- c()
        AlfaH <- c()
        AlfaC <- c()
        GamH <- c()
        GamC <- c()
        
        HnodTconv <- HnodT + 20
        CnodTconv <- CnodT + 20
        
#         print(SM)
        
        while (convflag ==1){
                
                for (i in 1:NHEX){
                        
                        for (j in 1:nrow(HstreamT)){
                                
                                if (SM[i,1] == j && SM[i,2]!=0){
                                        
                                        CPH[i] <- hca[j] + (hcb[j]/2)*(HnodT[j,SM[i,2]] + HnodT[j,SM[i,2]+1]) + (hcc[j]/3)*(HnodT[j,SM[i,2]]^2 + HnodT[j,SM[i,2]+1]^2 + HnodT[j,SM[i,2]]*HnodT[j,SM[i,2]+1])+(hcd[j]/4)*(HnodT[j,SM[i,2]]^2+HnodT[j,SM[i,2]+1]^2)*(HnodT[j,SM[i,2]]+HnodT[j,SM[i,2]+1])+(hce[j]/5)*(HnodT[j,SM[i,2]]^4+(HnodT[j,SM[i,2]]^3)*HnodT[j,SM[i,2]+1]+(HnodT[j,SM[i,2]]*HnodT[j,SM[i,2]+1])^2+HnodT[j,SM[i,2]]*HnodT[j,SM[i,2]+1]^3+HnodT[j,SM[i,2]+1]^4)
#                                         CPH[i] <- hca[j]
                                        
                                }else if (SM[i,1] == j && SM[i,2] == 0){
                                        
                                        CPH[i] <- 0
                                        
                                }
                        }
                        
                                
                        for (k in 1:nrow(SPLH)){
                                
                                j <- SM[i,1]
                                
                                if (SM[i,1] == SPLH[k,1] && SM[i,2] != 0){
                                        
                                        if (SM[i,2] > SPLH[k,2] && SM[i,2] < SPLH[k,3]){
                                                
                                                CPH[i] <- (hca[j]+(hcb[j]/2)*(HnodT[j,SM[i,2]]+HnodT[j,SM[i,2]+1])+(hcc[j]/3)*(HnodT[j,SM[i,2]]^2+HnodT[j,SM[i,2]+1]^2+HnodT[j,SM[i,2]]*HnodT[j,SM[i,2]+1])+(hcd[j]/4)*(HnodT[j,SM[i,2]]^2+HnodT[j,SM[i,2]+1]^2)*(HnodT[j,SM[i,2]]+HnodT[j,SM[i,2]+1])+(hce[j]/5)*(HnodT[j,SM[i,2]]^4+(HnodT[j,SM[i,2]]^3)*HnodT[j,SM[i,2]+1]+(HnodT[j,SM[i,2]]*HnodT[j,SM[i,2]+1])^2+HnodT[j,SM[i,2]]*HnodT[j,SM[i,2]+1]^3+HnodT[j,SM[i,2]+1]^4))*(1-SpltrH[SPLH[k,1]])
                                                #CPH[i] <- hca[j] 
                                                
                                        }
                                }else if (SM[i,1] == SPLH[k,1] && SM[i,2] ==0){
                                        
                                        CPH[i] <-0
                                }
                        }
                        
                        for (l in (nrow(HstreamT)+1):(nrow(HstreamT) + nrow(SPLH))){
                                
                                j <- l-nrow(HstreamT)
                                #print (j)
                                j <- SPLH[j,1]
                                
                                if (SM[i,1] == l && SM[i,2] != 0){
                                        
                                        CPH[i] <- ((hca[j]+(hcb[j]/2)*(HnodT[l,SM[i,2]]+HnodT[l,SM[i,2]+1])+(hcc[j]/3)*(HnodT[l,SM[i,2]]^2+HnodT[l,SM[i,2]+1]^2+HnodT[l,SM[i,2]]*HnodT[l,SM[i,2]+1])+(hcd[j]/4)*(HnodT[l,SM[i,2]]^2+HnodT[l,SM[i,2]+1]^2)*(HnodT[l,SM[i,2]]+HnodT[l,SM[i,2]+1])+(hce[j]/5)*(HnodT[l,SM[i,2]]^4+(HnodT[l,SM[i,2]]^3)*HnodT[l,SM[i,2]+1]+(HnodT[l,SM[i,2]]*HnodT[l,SM[i,2]+1])^2+HnodT[l,SM[i,2]]*HnodT[l,SM[i,2]+1]^3+HnodT[l,SM[i,2]+1]^4)))*SpltrH[SPLH[l-nrow(HstreamT),1]]
                                                                              
                                        
                                }else if (SM[i,1] == l && SM[i,2] == 0){
                                        
                                        CPH[i] <- 0
                                        
                                }
                        }
                        
                        for (j in 1:nrow(CstreamT)){
                                
                                if (SM[i,3] == j && SM[i,4] != 0){
                                        
                                        CPC[i] <- cca[j]+(ccb[j]/2)*(CnodT[j,SM[i,4]]+CnodT[j,SM[i,4]+1])+(ccc[j]/3)*(CnodT[j,SM[i,4]]^2+CnodT[j,SM[i,4]+1]^2+CnodT[j,SM[i,4]]*CnodT[j,SM[i,4]+1])+(ccd[j]/4)*(CnodT[j,SM[i,4]]^2+CnodT[j,SM[i,4]+1]^2)*(CnodT[j,SM[i,4]]+CnodT[j,SM[i,4]+1])+(cce[j]/5)*(CnodT[j,SM[i,4]]^4+(CnodT[j,SM[i,4]]^3)*CnodT[j,SM[i,4]+1]+(CnodT[j,SM[i,4]]*CnodT[j,SM[i,4]+1])^2+CnodT[j,SM[i,4]]*CnodT[j,SM[i,4]+1]^3+CnodT[j,SM[i,4]+1]^4)
                                        
                                }else if (SM[i,3] == j && SM[i,4] == 0){
                                        
                                        CPC[i] <- 0
                                }
                        }
                        
                        for (k in 1:nrow(SPLC)){
                                
                                j <- SM[i,3]
                                
                                if (SM[i,3] == SPLC[k,1] && SM[i,4] != 0){
                                        
                                        if (SM[i,4] > SPLC[k,2] && SM[i,4] < SPLC[k,3]){
                                                
                                                CPC[i] <- (cca[j]+(ccb[j]/2)*(CnodT[j,SM[i,4]]+CnodT[j,SM[i,4]+1])+(ccc[j]/3)*(CnodT[j,SM[i,4]]^2+CnodT[j,SM[i,4]+1]^2+CnodT[j,SM[i,4]]*CnodT[j,SM[i,4]+1])+(ccd[j]/4)*(CnodT[j,SM[i,4]]^2+CnodT[j,SM[i,4]+1]^2)*(CnodT[j,SM[i,4]]+CnodT[j,SM[i,4]+1])+(cce[j]/5)*(CnodT[j,SM[i,4]]^4+(CnodT[j,SM[i,4]]^3)*CnodT[j,SM[i,4]+1]+(CnodT[j,SM[i,4]]*CnodT[j,SM[i,4]+1])^2+CnodT[j,SM[i,4]]*CnodT[j,SM[i,4]+1]^3+CnodT[j,SM[i,4]+1]^4))*(1-SpltrC[SPLC[k,1]])
                                                                      
                                        }
                                        
                                }else if (SM[i,3] == SPLC[k,1] && SM[i,4] == 0){
                                        
                                        CPC[i] <- 0
                                }
                                
                                
                        }
                        
                        for (l in (nrow(CstreamT)+1) : (nrow(CstreamT)+nrow(SPLC))){
                                
                                j <- l-nrow(CstreamT)
                                j <- SPLC[j,1]
                                
                                if (SM[i,3] == l && SM[i,4] != 0){
                                        
                                        CPC[i] <- (cca[j]+(ccb[j]/2)*(CnodT[l,SM[i,4]]+CnodT[l,SM[i,4]+1])+(ccc[j]/3)*(CnodT[l,SM[i,4]]^2+CnodT[l,SM[i,4]+1]^2+CnodT[l,SM[i,4]]*CnodT[l,SM[i,4]+1])+(ccd[j]/4)*(CnodT[l,SM[i,4]]^2+CnodT[l,SM[i,4]+1]^2)*(CnodT[l,SM[i,4]]+CnodT[l,SM[i,4]+1])+(cce[j]/5)*(CnodT[l,SM[i,4]]^4+(CnodT[l,SM[i,4]]^3)*CnodT[l,SM[i,4]+1]+(CnodT[l,SM[i,4]]*CnodT[l,SM[i,4]+1])^2+CnodT[l,SM[i,4]]*CnodT[l,SM[i,4]+1]^3+CnodT[l,SM[i,4]+1]^4))*SpltrC[SPLC[l-nrow(CstreamT),1]];
                                        
                                }else if (SM[i,3] == l && SM[i,4] == 0){
                                        
                                        CPC[i] <-0
                                }
                                
                                
                        }
                        
                        # Calculating the overall U and then using the simplified equation provided by Jezowski et al for the caluclation of the alpha, which in turn is used
                        # to find the outlet temperature
                        # formula is Touthot = alpha*Tinhot + (1-alpha)*Tincold
                        # split streams also will be taken care because the CPH and CPC are formulated accordingly
                        
#                         print(CPH)
#                         print(CPC)
#                         print(SM)
                        
                        if (CPC[i] > 10^10 || CPH[i] > 10^10){
                                
                                CP_error <- c(CPC(i), CPH(i))
                                SM_error <- SM
                        }
                      
                        
                        if (CPH[i] != 0){
                                
                                if (CPC[i] != 0){
                                        
                                        GamH[i] <- (HstreamU[SM[i,1]]*CstreamU[SM[i,3]]/(HstreamU[SM[i,1]]+CstreamU[SM[i,3]]))*Area[i]/CPH[i]
                                        GamC[i] <- (HstreamU[SM[i,1]]*CstreamU[SM[i,3]]/(HstreamU[SM[i,1]]+CstreamU[SM[i,3]]))*Area[i]/CPC[i]
                                        
                                        if ((GamC[i]-GamH[i]*exp(GamH[i]-GamC[i])) !=0){
                                                
                                                AlfaH[i] <- (GamC[i]-GamH[i])/(GamC[i]-GamH[i]*exp(GamH[i]-GamC[i]))
                                                AlfaC[i] <- (GamH[i]-GamC[i])/(GamH[i]-GamC[i]*exp(GamC[i]-GamH[i]))
                                        }else if (GamC[i]==GamH[i] & GamC[i] !=0) {
                                                
                                                AlfaH[i] <- 1/(GamH[i]+1)
                                                AlfaC[i] <- 1/(GamH[i]+1)
                                                
                                        }
                                        else if (GamC[i]==0 | GamH[i]==0) {
                                        
                                                AlfaH[i] <- c(1)
                                                AlfaC[i] <- c(1)
                                        }}
                                else {
                                        AlfaH[i] <- c(1)
                                        AlfaC[i] <- c(1)  
                                }
                        }else {
                                
                                AlfaH[i] <- c(1)
                                AlfaC[i] <- c(1)
                        }
                        
                }
#                 convflag <- 0
#                 print(AlfaH)
#                 print(AlfaC)
#                 print(CPH)
#                 print(CPC)

#Temperature of Hot Stream Nodes 
        
        for (i in 1 : (nrow(HstreamT)+nrow(SPLH))){
                
                             
                for (j in 1 : NODH[i]) {
                        
                        HstreamN <- i
                        HnodeN <- j
                     
                        
                        HnodT[i,j+1] <- temp_H(SM,AlfaH,HnodT,CnodT,HstreamN,HnodeN,NHEX)
#                         print(HnodT)
                        
                        for (l in 1 : (nrow(HstreamT)+nrow(SPLH))) {
                                
                                if (is.null(SPLH)==FALSE){
                                        
                                        cd <- which(SPLH[,1]==l,arr.ind=FALSE)
                                        
                                        if (length(cd) ==0) {
                                                
                                                cd <- c(0)
                                        }
#                                         print(cd)
                                        
                                        pos[l] <- c(cd)
                                }
                                
                        
                        }
#                         print(pos)
                        # Calculating the node temperatures at the point of merging
                        
                        for (k in 1 : nrow(SPLH)){
                               if (i == SPLH[k,1] & j == SPLH[k,3]){
                                        
                                        HnodT[i,j+1] <- HnodT[SPLH[k,1],SPLH[k,3]]*(1-SpltrH[SPLH[k,1]])+(HnodT[(pos[SPLH[k,1]]+nrow(HstreamT)),NODH[pos[SPLH[k,1]]+nrow(HstreamT)]+1])*SpltrH[SPLH[k,1]]
                                        
                                }
#                                 print(i)
                        }
#                         print(HnodT)
                        
                        # Calculating the node temperature at the point of split
                        
                        for (k in 1 : nrow(SPLH)){
                                
#                                 print((pos[SPLH[k,1]]+nrow(HstreamT)))
                                
                                if (i==(pos[SPLH[k,1]]+nrow(HstreamT))){
                                        
                                        HnodT[i,1] <- HnodT[SPLH[k,1],SPLH[k,2]]
                                }
                        }
                        
                                             
                } 

        
        }

#Temperature of Cold Stream Nodes

        for (i in 1 : (nrow(CstreamT)+nrow(SPLC))){
        
                for (j in 1 : NODC[i]) {
                
                        CstreamN <- i
                        CnodeN <- j
                        CnodT[i,j+1] <- temp_C(SM,AlfaC,HnodT,CnodT,CstreamN,CnodeN,NHEX)
#                         print(CnodT)
                
                        for (l in 1 : (nrow(CstreamT)+nrow(SPLC))) {
                        
                                if (is.null(SPLC)==FALSE){
                                
                                        cd <- which(SPLC[,1]==l,arr.ind=FALSE)
                                
                                        if (length(cd) ==0) {
                                        
                                                cd <- c(0)
                                        }
#                                         print(cd)
                                
                                        pos[l] <- c(cd)
                                }
                        
                        
                        }
                
                # Calculating the node temperatures at the point of merging
                
                        for (k in 1 : nrow(SPLC)){
                                if (i == SPLC[k,1] & j == SPLC[k,3]){
                                
                                        CnodT[i,j+1] <- CnodT[SPLC[k,1],SPLC[k,3]]*(1-SpltrC[SPLC[k,1]])+(CnodT[(pos[SPLC[k,1]]+nrow(CstreamT)),NODC[pos[SPLC[k,1]]+nrow(CstreamT)]+1])*SpltrC[SPLC[k,1]]
                                
                                }
#                                 print(i)
                        }
                
                # Calculating the node temperature at the point of split
                
                        for (k in 1 : nrow(SPLC)){
                        
                                if (i==(pos[SPLC[k,1]]+nrow(CstreamT))){
                                
                                        CnodT[i,1] <- CnodT[SPLC[k,1],SPLC[k,2]]
                                }
                        }
                
                
                }      
        
        }
# print(HnodT)
# print(CnodT)

        for (i in 1 : NHEX) {
                
                if (SM[i,2] != 0) {
                        
                        Q1[i]    <- CPC[i]*(CnodT[SM[i,3],SM[i,4]+1] - CnodT[SM[i,3],SM[i,4]])
                        Q2[i]    <- CPH[i]*(HnodT[SM[i,1],SM[i,2]] - HnodT[SM[i,1],SM[i,2]+1])
                        delT1[i] <- HnodT[SM[i,1],SM[i,2]] - CnodT[SM[i,3],SM[i,4]+1]
                        delT2[i] <- HnodT[SM[i,1],SM[i,2]+1] - CnodT[SM[i,3],SM[i,4]]
                        lmtd[i]  <- (delT1[i]-delT2[i])/(log(delT1[i]/delT2[i]))
                        Q3[i]    <- ((HstreamU[SM[i,1]])*CstreamU[SM[i,3]])/(HstreamU[SM[i,1]]+CstreamU[SM[i,3]])*lmtd[i]*Area[i]
#                         print(Q3)
                }
        }
        bhar <- bhar + 1
        if ((norm(HnodT-HnodTconv)+norm(CnodT-CnodTconv))<= 0.1)  {
                
                convflag <-c(0)
        }else {
                
                HnodTconv <- HnodT
                CnodTconv <- CnodT
                
        }

        if (bhar > 500) {
                
                convflag <- c(0)
        }

       
        }

        
        for (i in 1 : nrow(HstreamT)) {
                
                HnodoutT[i] <- HnodTconv[i,NODH[i]+1]
                Hload[i] <- (HnodoutT[i]-HstreamT[i,2])*hca[i] + (HnodoutT[i]^2 - HstreamT[i,2]^2)*hcb[i]/2 + (HnodoutT[i]^3 - HstreamT[i,2]^3)*hcc[i]/3  + (HnodoutT[i]^4 - HstreamT[i,2]^4)*hcd[i]/4 + (HnodoutT[i]^5 - HstreamT[i,2]^5)*hce[i]/5                          
                
                if (Hload[i] > 0) {
                        
                        delT1[i] <- HstreamT[i,2] - CU[1]
                        delT2[i] <- HnodoutT[i] - CU[2]
                        lmtd[i] <- (delT1[i] - delT2[i])/(log(delT1[i]/delT2[i]))
                        Carea[i] <- abs(Hload[i])/((HstreamU[i]*CU[3])/(HstreamU[i] + CU[3]))/lmtd[i]
                        
                        if (delT1[i] >= 2 && delT2[i] >=2){
                                
                                penalty <- penalty + 0
                        } else {
                                
                                penalty <- penalty + 100000
                        }
                }
                
                if (Hload[i]<0) {
                        
                        delT1[i] <- HstreamT[i,2] - HU[1]
                        delT2[i] <- HnodoutT[i] - HU[2]
                        lmtd[i] <- (delT1[i] - delT2[i])/log(delT1[i]/delT2[i])
                        Carea[i] <- abs(Hload[i])/((HstreamU[i]*CU[3])/(HstreamU[i] + CU[3]))/lmtd[i]
                        
                        if (delT1[i] >= 2 && delT2[i] >=2){
                                
                                penalty <- penalty + 0
                        } else {
                                
                                penalty <- penalty + 100000
                        }
                        
                }
                
                if (abs(Hload[i])<1) {
                        
                        Carea[i] <- c(0)
                        Hload[i] <- c(0)
                } else {
                        
                        HnodoutT[i] <- HstreamT[i,2]
                }
        }
# print(Harea)
# print(Carea)

        for (i in 1 : nrow(CstreamT)){
                
                CnodoutT[i] <- CnodTconv[i,NODC[i]+1]
                Cload[i] <- (CnodoutT[i]-CstreamT[i,2])*cca[i] + (CnodoutT[i]^2 - CstreamT[i,2]^2)*ccb[i]/2 + (CnodoutT[i]^3 - CstreamT[i,2]^3)*ccc[i]/3 + (CnodoutT[i]^4 - CstreamT[i,2]^4)*ccd[i]/4 + (CnodoutT[i]^5 - CstreamT[i,2]^5)*cce[i]/5
                
                if (Cload[i]<0){
                        
                        delT1[i] <- CstreamT[i,2] - HU[1]
                        delT2[i] <- CnodoutT[i] - HU[2]
                        lmtd[i] <- abs((delT1[i]-delT2[i])/(log(delT1[i]/delT2[i])))
                        Harea[i] <- abs(Cload[i])/((CstreamU[i]*HU[3])/(CstreamU[i]+HU[3]))/lmtd[i]
                        
                        if (delT1[i] >=2 && delT2[i] >=2) {
                                
                                penalty <- penalty  + 0
                        } else {
                                
                                penalty <- penalty + 100000
                        }
#                         print(Harea)
#                         print(Cload)
                }
                
                
                if (Cload[i]>0){
                        
                        delT1[i] <- CstreamT[i,2]-CU[1]
                        delT2[i] <- CnodoutT[i]-CU[2]
                        lmtd[i] <- abs((delT1[i]-delT2[i])/(log(delT1[i]/delT2[i])))
                        Harea[i] <- abs(Cload[i])/((CstreamU[i]*HU[3])/(CstreamU[i]+HU[3]))/lmtd[i]
                        
                        if (delT1[i] >=2 && delT2[i] >=2) {
                                
                                penalty <- penalty  + 0
                        } else {
                                
                                penalty <- penalty + 100000
                        }
                        
                }
                
                if (abs(Cload[i])<1){
                        
                        Harea[i] <- 0
                        Carea[i] <- 0
                }
#                 } else {
#                         
#                         CnodoutT[i] <- CstreamT[i,2]
#                 }
        }

# Calculation of the heat exchanger area
        
        for (i in 1 : NHEX) {
                
                if (SM[i,2] != 0){
                        
                        Area[i] <- Area[i]
                        HEXarea[i] <- Area[i]
                } else {
                        
                        HEXarea[i] <- c(0)
                }
        }

# print(Carea)
# print(Harea)

#SOrting of heat exchanger areas to compare with the existing areas, ERS-4 is used

allhex <- matrix(c(HEXarea,Carea,Harea),1,)

allhex <- sort(allhex, decreasing =TRUE)

allhex_strt3 <- allhex
area_strt3 <- ExitHEX
match_strt3 <- c()
newcost_strt3 <- c(0)
addcost_strt3 <- c(0)
b<-c()
# print(allhex_strt3)
# print(area_strt3)

for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] >= area_strt3[i] - 0.01*area_strt3[i] && allhex_strt3[j] < area_strt3[i]){
                                
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}

for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] <= area_strt3[i] + 0.01*area_strt3[i] && allhex_strt3[j] >= area_strt3[i] ){
                                
#                                 addcost_strt3 <- addcost_strt3 + area_cost_2*(allhex_strt3[j]-area_strt3[i])^exponent_2
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}

# print(allhex_strt3)
# print(area_strt3)
# print(match_strt3)

for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] >= area_strt3[i] - 0.05*area_strt3[i] && allhex_strt3[j] < area_strt3[i] ){
                                
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}

# for (i in 1 : length(area_strt3)) {
#         
#         j_ind <- length(allhex_strt3)
#         
#         for (j in 1 : j_ind) {
#                 
#                 if (allhex_strt3[j] != 0) {
#                         
#                         if (allhex_strt3[j] <= area_strt3[i] + 0.05*area_strt3[i] && allhex_strt3[j] >= area_strt3[i] ){
#                                 
#                                 addcost_strt3 <- addcost_strt3 + area_cost_2*(allhex_strt3[j]-area_strt3[i])^exponent_2
#                                 allhex_strt3[j] <- c(0)
#                                 area_strt3[i] <- c(0)
#                                 b <- matrix(c(i,j),2,1)
#                                 match_strt3 <- cbind(match_strt3,b)
#                         }
#                 }
#         }
# }

for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] >= area_strt3[i] - 0.1*area_strt3[i] && allhex_strt3[j] < area_strt3[i] ){
                                
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}

for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] <= area_strt3[i] + 0.1*area_strt3[i] && allhex_strt3[j] >= area_strt3[i] ){
                                
                                addcost_strt3 <- addcost_strt3 + area_cost_2*(allhex_strt3[j]-area_strt3[i])^exponent_2
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}

for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] >= area_strt3[i] - 0.15*area_strt3[i] && allhex_strt3[j] < area_strt3[i] ){
                                
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}

for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] <= area_strt3[i] + 0.15*area_strt3[i] && allhex_strt3[j] >= area_strt3[i] ){
                                
                                addcost_strt3 <- addcost_strt3 + area_cost_2*(allhex_strt3[j]-area_strt3[i])^exponent_2
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}

for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] >= area_strt3[i] - 0.25*area_strt3[i] && allhex_strt3[j] < area_strt3[i] ){
                                
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}

# for (i in 1 : ncol(area_strt3)) {
#         
#         j_ind <- ncol(allhex_strt3)
#         
#         for (j in 1 : j_ind) {
#                 
#                 if (allhex_strt3[j] != 0) {
#                         
#                         if (allhex_strt3[j] <= area_strt3[i] + 0.25*area_strt3[i] & allhex_strt3[j] >= area_strt3[i] ){
#                                 
#                                 addcost_strt3 <- addcost_strt3 + area_cost_2*(allhex_strt3[j]-area_strt3[i])^exponent_2
#                                 allhex_strt3[j] <- c(0)
#                                 area_strt3[i] <- c(0)
#                                 b <- matrix(c(i,j),2,1)
#                                 match_strt3 <- cbind(match_strt3,b)
#                         }
#                 }
#         }
# }

for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] >= area_strt3[i] - 0.5*area_strt3[i] && allhex_strt3[j] < area_strt3[i] ){
                                
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}


for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] >= area_strt3[i] - 0.75*area_strt3[i] && allhex_strt3[j] < area_strt3[i] ){
                                
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}


for (i in 1 : length(area_strt3)) {
        
        j_ind <- length(allhex_strt3)
        
        for (j in 1 : j_ind) {
                
                if (allhex_strt3[j] != 0) {
                        
                        if (allhex_strt3[j] >= area_strt3[i] - 0.9*area_strt3[i] && allhex_strt3[j] < area_strt3[i] ){
                                
                                allhex_strt3[j] <- c(0)
                                area_strt3[i] <- c(0)
                                b <- matrix(c(i,j),2,1)
                                match_strt3 <- cbind(match_strt3,b)
                        }
                }
        }
}

allhex_strt3[allhex_strt3==0] <- NA
allhex_strt3 <- allhex_strt3[!is.na(allhex_strt3)]

area_strt3[area_strt3] <- NA
area_strt3 <- area_strt3[!is.na(area_strt3)]

area_strt3 <- sort(area_strt3, decreasing = TRUE)

#print(length(allhex_strt3))
if (length(allhex_strt3>0)){
for (i in 1 : length(allhex_strt3)){
        
        if (allhex_strt3[i] > 0.5){
                
                newcost_strt3 <- newcost_strt3 + new_area_capital + area_cost_1*allhex_strt3[i]^exponent_1
        }else{
                
                newcost_strt3 <- newcost_strt3
        }
        #print(newcost_strt3)
}
}

for (i in 1 : nrow(HstreamT)){
        
        if (Hload[i]<0){
                
                Cload <- c(Cload,Hload[i])
                penalty <- penalty + 100000
                Hload[i] <- c(0)
        }
}

for (i in 1 : nrow(CstreamT)){
        
        if (Cload[i] > 0) {
                
                Hload <- c(Hload,Cload[i])
                penalty <- penalty + 100000
                Cload[i] <- c(0)
        }
}

H_utility <- sum (abs(Cload))
C_utility <- sum(abs(Hload))

# Utility Cost

utilitycost <- CU[4]*C_utility + HU[4]*H_utility
Investmentcost_strt3 <- addcost_strt3 + newcost_strt3

Cons <- c()

for (i in 1 : NHEX) {
        
        if (SM[i,2]!=0){
                
                a <- (HnodT[SM[i,1],SM[i,2]]-CnodT[SM[i,3],SM[i,4]+1]) - Delta
                b <- (HnodT[SM[i,1],SM[i,2]+1]-CnodT[SM[i,3],SM[i,4]]) - Delta
                
                if (a<0) {
                        
                        Cons[i] <- abs(a)
                }else{
                        
                        Cons[i] <- c(0)
                }
                
                if (b<0) {
                        
                        Cons[i+NHEX] <- abs(b)
                }else {
                        Cons[i+NHEX] <- c(0)
                }
        }else{
                
                Cons[i] <- c(0)
                Cons[i+NHEX] <- c(0)
        }
}

# print(HnodT)
# print(HnodTconv)
# print(bhar)  
# print(addcost_strt3)
# print(match_strt3)
# print(newcost_strt3)
# print(Cons)
# print(Investmentcost_strt3)
# print(utilitycost)
# print(H_utility)
# print(C_utility)
        
utilitycost <<- utilitycost
investmentcost <<- Investmentcost_strt3
Cons <<- Cons        
        
        
        
        
        
        
        
        
        
        
        
        
        
}