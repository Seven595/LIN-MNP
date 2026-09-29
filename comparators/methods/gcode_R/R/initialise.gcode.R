#' @export
initialise.gcode <- function(data_list,
                             config,
                             transfer,
                             join
){

  if (config$verbose){
    print(c(paste("Initialising alpha_sample with : ",config$init[1],sep=""),
            paste("Initialising beta_sample with : ",config$init[2],sep=""))
    )
  }

  data_list <- lapply(data_list,function(X){
    X <- gcode::prepare_data(X,
                        log=config$transform$log,
                        center=config$transform$center,
                        scale.norm=config$transform$norm
                        )
  })

  main.code <- list(code=lapply(unique(join$complete$code),function(X){NULL}),encode=lapply(unique(join$complete$code),function(X){NULL}))
  main.intercept <- list(rintercept=lapply(unique(join$complete$rintercept),function(X){NULL}),cintercept=lapply(unique(join$complete$cintercept),function(X){NULL}))
  main.parameters <- list(alpha_sample = lapply(unique(join$complete$alpha_sample),function(X){NULL}), beta_sample = lapply(unique(join$complete$beta_sample),function(X){NULL}),
                          alpha_signal = lapply(unique(join$complete$alpha_signal),function(X){NULL}), beta_signal = lapply(unique(join$complete$beta_signal),function(X){NULL}))

  
  
  names(main.code$code) <- unique(join$complete$code)
  names(main.code$encode) <- unique(join$complete$code)
  
  names(main.parameters$alpha_sample) <- unique(join$complete$alpha_sample)
  names(main.parameters$beta_sample) <- unique(join$complete$beta_sample)
  
  names(main.parameters$alpha_signal) <- unique(join$complete$alpha_signal)
  names(main.parameters$beta_signal) <- unique(join$complete$beta_signal)
  
  names(main.intercept$rintercept) <- unique(join$complete$rintercept)
  names(main.intercept$cintercept) <- unique(join$complete$cintercept)
    
  
  
  
  config.code <- config
  config.code$i_dim <- config.code$k_dim
  config.code$j_dim <- config.code$k_dim

  for (i in 1:length(join$complete$data_list)){

    if (is.null(main.parameters$alpha_sample[[join$complete$alpha_sample[i]]])){
      if (!is.null(transfer$main.parameters$alpha_sample[[join$complete$alpha_sample[i]]])){
        main.parameters$alpha_sample[[join$complete$alpha_sample[i]]] <- transfer$main.parameters$alpha_sample[[join$complete$alpha_sample[i]]]
      } else {
        main.parameters$alpha_sample[[join$complete$alpha_sample[i]]] <- as.matrix(initialise.parameters(x = data_list[[join$complete$data_list[i]]],config = config, param.type = "alpha_sample"))
      }
    }

    if (is.null(main.parameters$beta_sample[[join$complete$beta_sample[i]]])){
      if (!is.null(transfer$main.parameters$beta_sample[[join$complete$beta_sample[i]]])){
        main.parameters$beta_sample[[join$complete$beta_sample[i]]] <- transfer$main.parameters$beta_sample[[join$complete$beta_sample[i]]]
      } else {
        main.parameters$beta_sample[[join$complete$beta_sample[i]]] <- as.matrix(initialise.parameters(x = data_list[[join$complete$data_list[i]]],config = config, param.type = "beta_sample"))
      }
    }

    if (is.null(main.code$encode[[join$complete$encode[i]]])){
      if (!is.null(transfer$main.code$encode[[join$complete$encode[i]]])){
        main.code$encode[[join$complete$encode[i]]] <- transfer$main.code$encode[[join$complete$encode[i]]]
      } else {
        main.code$encode[[join$complete$encode[i]]] <- main.parameters$alpha_sample[[join$complete$alpha_sample[i]]]%*%as.matrix(data_list[[join$complete$data_list[i]]])%*%main.parameters$beta_sample[[join$complete$beta_sample[i]]]
      }
    }

    if (is.null(main.code$code[[join$complete$code[i]]])){
      if (!is.null(transfer$main.code$code[[join$complete$code[i]]])){
        main.code$code[[join$complete$code[i]]] <- transfer$main.code$code[[join$complete$code[i]]]
      } else {
        main.code$code[[join$complete$code[i]]] <- as.matrix(MASS::ginv(main.parameters$alpha_sample[[join$complete$alpha_sample[i]]]%*%t(main.parameters$alpha_sample[[join$complete$alpha_sample[i]]]))%*%main.code$encode[[join$complete$encode[i]]]%*%MASS::ginv(t(main.parameters$beta_sample[[join$complete$beta_sample[i]]])%*%main.parameters$beta_sample[[join$complete$beta_sample[i]]]))
      }
    }

    if (is.null(main.intercept$cintercept[[join$complete$cintercept[i]]])){
      if (!is.null(transfer$main.intercept$cintercept[[join$complete$cintercept[i]]])){
        main.intercept$cintercept[[join$complete$cintercept[i]]] <- transfer$main.intercept$cintercept[[join$complete$cintercept[i]]]
      } else {
        main.intercept$cintercept[[join$complete$cintercept[i]]] <- colMeans(t(main.intercept$alpha_sample[[join$complete$alpha_sample[i]]])%*%main.code$code[[join$complete$code[i]]]%*%t(main.intercept$beta_sample[[join$complete$beta_sample[i]]]))
      }
    }
    
    if (is.null(main.intercept$rintercept[[join$complete$rintercept[i]]])){
      if (!is.null(transfer$main.intercept$rintercept[[join$complete$rintercept[i]]])){
        main.intercept$rintercept[[join$complete$rintercept[i]]] <- transfer$main.intercept$rintercept[[join$complete$rintercept[i]]]
      } else {
        main.intercept$rintercept[[join$complete$rintercept[i]]] <- rowMeans(t(main.intercept$alpha_sample[[join$complete$alpha_sample[i]]])%*%main.code$code[[join$complete$code[i]]]%*%t(main.intercept$beta_sample[[join$complete$beta_sample[i]]]))
      }
    }
    

  }


  return(
    list(
      main.parameters = main.parameters,
      main.code = main.code,
      main.intercept = main.intercept
      )
  )

}


#' @export
initialise.parameters <- function(x,config,param.type){

  if (param.type=="beta_sample"){
    main.param <- if (config$init[[2]]=="rnorm"){
      array(rnorm(dim(x)[2]*config$j_dim),dim=c(dim(x)[2],config$j_dim))
    } else if (config$init[[2]]=="runif"){
      array(runif(dim(x)[2]*config$j_dim),dim=c(dim(x)[2],config$j_dim))
    } else if (config$init[[2]]=="irlba") {
      (irlba::irlba(as.matrix(x), nv = config$j_dim, maxit = 1)$v)
    } else if (config$init[[2]]=="rsvd") {
      (rsvd::rsvd(as.matrix(x), nv = config$j_dim)$v)
    }
  }

  if (param.type=="alpha_sample"){
    main.param <- if (config$init[[1]]=="rnorm") {
      array(rnorm(config$i_dim*dim(x)[1]),dim=c(config$i_dim,dim(x)[1]))
    } else if (config$init[[1]]=="runif") {
      array(runif(config$i_dim*dim(x)[1]),dim=c(config$i_dim,dim(x)[1]))
    } else if (config$init[[1]]=="irlba") {
      t(irlba::irlba(as.matrix(x), nu = config$i_dim, maxit = 1)$u)
    } else if (config$init[[1]]=="rsvd") {
      t(rsvd::rsvd(as.matrix(x), nu = config$i_dim)$u)
    }
  }

  return(main.param)

}
