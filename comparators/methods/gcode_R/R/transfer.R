#' Transfer model from one to another
#'
#' Main function to transferring models 
#' 
#' @param model_list A list of gcode models 
#' @param data_list A list of datasets (matrix or tensor etc.)
#' @param config Configuration parameters (required, default provided)
#' @param transfer Transferring pre-trained model parameters (not required)
#' @param join Join parameters that share the same axes to be similar (not required)
#'
#' @return  All models to be learned or updated
#' @export
transfer <- function(model_list,
                     data_list,
                     config = gcode::extract_config(verbose = F)
){
  
  
  runtime.start <- Sys.time()
  
  set.seed(config$seed)
  
  convergence.parameters <- list(count=0,score.vec=c())
  
  # Set up transfer of model
  transfer_parameters <- list()
  transfer_code <- gcode::extract_transfer_framework(F)
  
  join_list <- list()
  
  
  for (set.id in c(1:length(data_list))){
    
    transfer_parameters[[set.id]] <- gcode::extract_transfer_framework(F)
    join_list[[set.id]] <- gcode::extract_join_framework(F)
    
    transfer_parameters[[set.id]]$main.code <- model_list[[set.id]]$main.code
    transfer_parameters[[set.id]]$main.parameters <- model_list[[set.id]]$main.parameters
    join_list[[set.id]] <- model_list[[set.id]]$meta.parameters$join

  }
  
  gcode.model_list <- transfer_parameters
  
  while (T){
    
    prev_param <- transfer_parameters
    
    for (set.id in c(1:length(data_list))){

      if (config$verbose){
        print(paste("Updating model ...  ",set.id,sep=""))
      }
      
      config$verbose <- F
      gcode.model_list[[set.id]] <- gcode(data_list = data_list[[set.id]],
                                            config = config,
                                            transfer = transfer_parameters[[set.id]],
                                            join = join_list[[set.id]])
      config$verbose <- T
      
      
      for (intenral.set.id in 1:length(data_list)){
        transfer_parameters[[intenral.set.id]]$main.code <- 
          gcode.model_list[[intenral.set.id]]$main.code <- 
          gcode.model_list[[set.id]]$main.code
      }
      
      transfer_parameters[[set.id]]$main.parameters <- gcode.model_list[[set.id]]$main.parameters
      
    
    }
    
    total.mse <- Reduce("+",lapply(c(1:length(data_list)),function(set.id){
      matrix.residuals <- Reduce("+",transfer_parameters[[set.id]]$main.code$encode) - Reduce("+",prev_param[[set.id]]$main.code$encode)
      total.mse <- mean(abs(matrix.residuals))
      return(total.mse)
    }))
    
    # Check convergence
    convergence.parameters$score.vec <- c(convergence.parameters$score.vec, total.mse)
    MSE <- mean(tail(convergence.parameters$score.vec,2))
    prev.MSE <- mean(tail(convergence.parameters$score.vec,3)[1:2])
    
    if ( convergence.parameters$count > ( 3 ) ){
      if (config$verbose == T){
        print(paste("Iteration: ",convergence.parameters$count," with Tolerance of: ", abs(prev.MSE - MSE),sep=""))
      }
    } else {
      if (config$verbose){
        print(paste("Iteration: ",convergence.parameters$count," ... initialising ... ",sep=""))
      }
    }
    
    if (convergence.parameters$count > config$min_iter){
      if ((convergence.parameters$count > config$max_iter ) | abs(prev.MSE - MSE) < config$tol){
        break
      }
    }
    
    convergence.parameters$count = convergence.parameters$count + 1
    
  }

  
  runtime.end <- Sys.time()
  
  return(list(model = gcode.model_list,
              convergence.parameters = convergence.parameters,
              run_time = runtime.end - runtime.start
              )
  )
  
}