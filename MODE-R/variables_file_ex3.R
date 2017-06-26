# Initialization

SPLH <- c()
SPLC <- c()
HnodoutT <- c()
CnodoutT <- c()

#--------------
# x<-c(5,3,4,6,6,1,3,11,11,8,4,11,2,7,1,5,6,5,4,7,3,6,4,8,2,1,1,12,5,1,4,3,1,4,1,8,10,4,2,4,4,2,4,5,4,0,2,0,7,0,1,0,7,4,1,1,10,0,1,0,291.02646,256.8190984,1450.775288,1468.175449,236.8531001,1209.227841,284.8089044,271.7002855,519.4961359,174.9071855,85.36413533,286.4847292,87.0592617,1476.376269,1495.2814,0.456744761,0.879918329,0.20776334,0,0,0,0,0,0.692119229,0.504205683,0,0)

# x<-c(1,2,1,2,2,4,2,4,3,3,3,3,4,4,4,4,5,5,4,5,6,6,4,6,7,7,1,7,8,8,2,8,9,9,3,9,10,7,4,10,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,rep(10,15),rep(0.5,3),rep(0,5),0.5,0.5,0,0)

NHEX <- 15
HstreamT <- t(matrix(c(298,268,339,100,250,200,257,50,170,150,282,40,100,77,77,40,189,40),2,9))
CstreamT <- t(matrix(c(25,365,271,282,182,189),2,3))
SPLH <- t(matrix(c(1,2,5,2,2,5,3,2,5,9,2,5),3,4))
SPLC <- matrix(c(1,2,10),1,3)
HstreamU <- rep(1,13)*1
CstreamU <- rep(1,4)*1

NODH <- rep(1,13)*8
NODC <- rep(1,4)*12

# Initial guess of the nodal temperatures
# hot streams

h1 <- c(298,290,280,277,276,270,270,270,270)
h2 <- c(339,259,259,259,259,259,259,259,259)
h3 <- c(250,245,245,245,245,245,245,245,245)
h4 <- c(257,250,250,250,250,250,250,250,250)
h5 <- c(170,160,160,160,160,160,160,160,160)
h6 <- c(282,230,230,230,230,230,230,230,230)
h7 <- c(100,100,100,100,100,100,100,100,100)
h8 <- c(77,77,77,77,77,77,77,77,77)
h9 <- c(189,180,180,180,180,180,180,180,180)

HnodT <- t(matrix(c(h1,h2,h3,h4,h5,h6,h7,h8,h9,h1,h2,h3,h9),9,13))


# cold streams

c1 <- c(25,255,255,255,255,255,255,255,255,255,255,255,255)
c2 <- c(271,271,271,271,278,278,278,278,278,278,278,278,278)
c3 <- c(182,182,182,182,182,182,182,182,182,182,182,182,182)

CnodT <- t(matrix(c(c1,c2,c3,c1),13,4))

# Utility details

HU <- c(1500,800,2,306.8)
CU <- c(10,40,2.5,5.25)

# Details of existing heat exchangers in decreasing order of areas

Exithex <- c(292,285,280,278,273,161,156,37,20,19,16,14,2,135,16,11,1054,258,116,85,61,55,24,20)
ExitHEX <- sort(Exithex,decreasing = TRUE)

# Details of the variable heat capacity values 
# hot streams

hca <- c(0.4276,137.9925828/1000,0.3604,18.52370708/1000,0.55875,0.06106491,2.081087,0.035703,0.03254988,0.4276,137.9925828/1000,0.3604,0.03254988)*1000
hcb <- c(0,0.187222233/1000,0,2.21E-05,0,6.75E-05,0,0.00E+00,3.39E-05,0,0.187222233/1000,0,3.39E-05)*1000*2
hcc <- c(0,-5.70E-08,0,-2.54E-09,0.00E+00,1.12E-08,0.00E+00,0.00E+00,2.67E-08,0,-5.70E-08,0,2.67E-08)*1000*3
hcd <- rep(0,13)*(10^-7)*10^-2
hce <- rep(0,13)*10^-12

# cold streams

cca <- c(0.195638207,0.798455,0.946429,0.195638207)*1000
ccb <- c(0.000926819,0,0,0.000926819)*2*1000
ccc <- c(-7.25589E-07,0.00E+00,0.00E+00,-7.25589E-07)*3*1000
ccd <- rep(0,4)
cce <- rep(0,4)


# Cost data

int <- 0        # interest rate
noy <- 5        # Plant lifetime
new_area_capital <- 94093 # cost for new heat exchanger excluding the area term
area_cost_1 <- 1127     # area cost which is the same for the new exchanger or the  addition in the existing heat exchanger
exponent_1 <- 0.9887    # exponent of the cost estimation
area_cost_2 <- 9665     # area cost for difference in new exchanger and existing heat exchanger
exponent_2 <- 0.68      # exponent for difference in costs

# BOunds for the decision variables. Decision variables include the structural variables (integer), area variables and split ratios (both continuous)

VL_1 <- c(1,0,1,0)      # corresponding the structural matrix
VL_2 <- c(1.5)          # corresponding to areas of heat exchangers
VL_3 <- c(0)            # corresponding to split ratios

VLB <- c(rep(VL_1,NHEX),rep(VL_2,NHEX),rep(VL_3,nrow(HstreamT)+nrow(CstreamT)))


VU_1 <- c(13.4,8.4,4.4,12.4)      # corresponding the structural matrix
VU_2 <- c(1500)                   # corresponding to areas of heat exchangers
VU_3 <- c(0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9)            # corresponding to split ratios


VUB <- c(rep(VU_1,15),rep(VU_2,15),VU_3)

Delta <- c(30)

MaxGen <- c(1000)
Numpop <- c(100)
dVtype <- c(rep(1,4*NHEX),rep(2,NHEX+nrow(HstreamT)+nrow(CstreamT)))  # 1 for integer, 2 for continuos variable
TLS <- Numpop/2
TR <- c(0.01)
mu_CR <- c(0.5)
mu_F <- c(0.5)
refPoints <- c()
refWeight <- c()

















