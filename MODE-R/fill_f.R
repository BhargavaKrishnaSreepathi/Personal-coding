fill_f <- function(x,npop)  {
        Pt1 <- vector()
        i <- 1
        tot_len <- 0
        while ((tot_len + length(x[[i]])) <= npop) {
                Pt1 <- c(Pt1,x[[i]])
                tot_len <- length(Pt1)
                i <- i + 1
        }
        
        returnVal <- list(Pt1=Pt1,i=i)
        return(returnVal)
}