#' @export
dgcode <- function(data_list,
                   config = gcode::extract_config(verbose = F),
                   transfer = gcode::extract_transfer_framework(verbose = F),
                   recover = gcode::extract_recovery_framework(verbose = F),
                   join = gcode::extract_join_framework(verbose=F),
                   references = gcode::extract_references_framework(F)
){
  
  dynamic_list_of_models <- list()
  
  combin <- cbind(rep(config$i_dim,length(config$j_dim)),rep(config$j_dim,length(config$i_dim)))
  
  for (i in 1:dim(combin)[1]){
    
    config$i_dim <- combin[i,1]
    config$j_dim <- combin[i,2]
    
    config$verbose <- F
    print(paste("Bootstrap: iteration   ",i,"   of   ",dim(combin)[1],"   ...",sep=""))
    
    gcode.model <- gcode::gcode(data_list = data_list, config = config, join = join, transfer = transfer, recover = recover, references = references)
    dynamic_list_of_models <- c(dynamic_list_of_models,list(gcode.model))
    
    data_list <- lapply(c(1:length(join$complete$data_list)),function(X){
      data_list[[join$complete$data_list[X]]] - t(gcode.model$main.parameters$alpha[[join$complete$alpha[X]]])%*%gcode.model$main.code$code[[join$complete$code[X]]]%*%t(gcode.model$main.parameters$beta[[join$complete$beta[X]]]) - gcode.model$main.parameters$intercept[[join$complete$data_list[X]]]
    })
    
  }
  
  return(dynamic_list_of_models)
}
