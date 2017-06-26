## crowding distance assignment

crowding <- function(front) {
        eye_d <- vector(mode='numeric',length(front[,1]))
        l <- length(front[,1])
        nobj <- length(front[1,])
        flist <- attributes(front)$dimnames[[1]]
        names(eye_d) <- flist
        for (m in 1:nobj) {
                eye <- front
                eye_order <- flist[order(front[,m])]
                eye_d[eye_order[1]] <- pi/0
                eye_d[eye_order[l]] <- pi/0
                if (l > 2) {
                        eye_count <- 1
                        for (i in eye_order[2:(l-1)]) {
                                eye_count <- eye_count + 1
                                eye_d[i] <- eye_d[i] + (eye[eye_order[eye_count+1],m] 
                                                        - eye[eye_order[eye_count-1],m])/(diff(range(eye,m)))
                        }
                }
        }
        returnVal <- eye_d
        return(returnVal)
}