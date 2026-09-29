#' Recover missing data points via imputation or prediction
#'
#' Main function to recover missing points. Used internally, but can be re-purposed by the user.
#'
#' @param data_list A list of datasets (matrix or tensor etc.)
#' @param code Code parameters from gcode
#' @param main.parameters Main parameters from gcode
#' @param config Configuration parameters from gcode
#' @param recover Recover list from gcode
#'
#' @return  Recovered data from imputation or prediction, with the design matrices and any user input parameters and functions
#' @export
recover_points <- function(data_list,
                           main.code,
                           main.parameters,
                           config,
                           join,
                           recover,
                           references){
  
  for (task in recover$task){
    
    if ("regression" %in% task){
      
      for (method in recover$method){
        
        
        for (i in which(references$data_list==1)){
          
          if (!is.null(recover$design.list[[join$complete$data_list[i]]])){
            
            if ("decode" %in% method){
              
              missing_points <- which(as.matrix(recover$design.list[[i]]==1),arr.ind = T)
              row_with_missing_points <- unique(missing_points[,1])
              column_with_missing_points <- unique(missing_points[,2])
              
              main.data <- data_list[[join$complete$data_list[i]]]
              pred <- main.parameters$intercept[[i]]+t(main.parameters$alpha[[join$complete$alpha[i]]])%*%main.code$code[[join$complete$code[i]]]%*%t(main.parameters$beta[[join$complete$beta[i]]])
              
              main.data[row_with_missing_points,column_with_missing_points]  <- pred[row_with_missing_points,column_with_missing_points]
              data_list[[join$complete$data_list[i]]] <- recover$predict.list[[join$complete$data_list[i]]] <- main.data
              
            }
          }
        }
      }
      
    }
    
    if ("classification" %in% task){
      
      
      for (method in recover$method){
        
        if ("label.projection" %in% method){
          
          labels <- recover$labels
          
          recover$encoded_covariate <- lapply(c(1:length(data_list)),function(X){
            transformed.data <- as.matrix(data_list[[X]])%*%(main.parameters$beta[[join$beta[X]]])
            return(transformed.data)
          })
          
          decoded_covariate <- do.call('cbind',lapply(c(1:length(recover$encoded_covariate)),function(X){
            recover$encoded_covariate[[X]]
          }))
          
          unlabel.decoded_covariate <- decoded_covariate[is.na(labels),]
          label.decoded_covariate <- decoded_covariate[!is.na(labels),]
          
          
          label_assign <- (unlabel.decoded_covariate)%*%t(label.decoded_covariate)
          recover$labels[is.na(labels)] <- apply(label_assign,1,function(X){(labels[!is.na(labels)])[which(X==max(X))]})
          
        }
      }
    }
    
  }
  
  return(list(recover=recover,
              data_list=data_list))
}
