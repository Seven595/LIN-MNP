#' Extract configuration parameters of gcode
#'
#' @param i_dim Dimension reduction for samples (assumed to be along rows)
#' @param j_dim Dimension reduction for features (assumed to be along columns)
#' @param min_iter Minimum iteration of gcode
#' @param max_iter Maximum iteration of gcode
#' @param n_cores Number of CPU cores, used for prediction only
#' @param tol Tolerance threshold for convergence (metric: Root Mean Squared Error)
#' @param verbose Print statements?
#' @param init Initialisation method for the model ("random","eigen-quick","eigen-dense","svd-quick","svd-dense")
#' @return  Configuration parameters for gcode
#' @export
extract_config <- function(verbose=T){
  config <- list(
    init=list(alpha_sample="irlba",beta_sample="irlba"),
    transform=list(log=F,center=F,norm=F),
    regularise=list(a=0,l=0),
    dimension_reduction=FALSE,
    i_dim = 30,
    j_dim = 30,
    max_iter=350,
    seed = 1,
    tol=1,
    verbose = T
  )

  if (verbose == T){
    print(config)
  }

  return(config)
}

#' Extract anchor framework to put into gcode
#'
#' Transfers learned parameters from a pre-trained model.
#' NOTE: This is an empty framework that the user must fill in.
#'
#' @param code Transfer the code
#' @param fix Fix anchors the code parameters, otherwise it will be updated

#' @return  Anchor framework for gcode
#' @export
extract_transfer_framework <- function(verbose=T){
  transfer <- list(
    main.code = list(code = NULL, encode = NULL),
    main.intercept = list(rintercept = NULL, cintercept = NULL),
    main.parameters = list(alpha_signal = NULL, beta_signal = NULL, alpha_sample = NULL, beta_sample = NULL, intercept = NULL),
    fix = list(code = F, encode = F, alpha_sample = F, beta_sample = F, alpha_signal = F, beta_signal = F, intercept = F)
    )
  
  if (verbose == T){
    print(transfer)
  }
  
  return(transfer)
}


#' Extract recovery framework to put into gcode
#'
#' Can recover data points by imputing or predicting missing values
#'
#' @param task Allows user to specify either a regression, classification, or imputation task
#' @param method The algorithm for the task (Options are regression/imputation: "knn","matrix.projection", -- provide your own --   ;   classification: "label.projection")
#' @param design.list A list of design structures where each element is given a 1 to indicate the test set, 0 indicates the train set.
#' @param labels For classification, these are the pre-defined labels
#' @return  Prediction framework for gcode
#' @export
extract_recovery_framework <- function(verbose=T){
  recover <- list(
    task = c("regression"),
    method = c("matrix.projection"),
    link_function = c("identity","identity"),
    design.list = NULL,
    labels = NULL
  )
  
  if (verbose == T){
    print(recover)
  }
  
  return(recover)
}






#' Extract join framework to put into gcode
#'
#' Join data to improve modelling capacity for similar axes
#' @param alpha_sample Joining the alpha_sample parameters. A vector of integers, where identical integers indicate same the data axis to be joined.
#' @param beta_sample Joining the beta_sample parameters. A vector of integers, where identical integers indicate same the data axis to be joined.
#' @param code Joining the code parameters. A vector of integers, where identical integers indicate same the data axis to be joined.
#' @export
extract_join_framework <- function(verbose=T){
  join <- list(complete=list(data_list=NULL,alpha_sample=NULL,beta_sample=NULL,code=NULL,encode=NULL,alpha_signal=NULL,beta_signal=NULL,rintercept=NULL,cintercept=NULL),
               labels=list(alpha_sample=NULL,beta_sample=NULL)
  )
  
  if (verbose == T){
    print(join)
  }
  
  return(join)
}







#' Extract join framework to put into gcode
#'
#' Join data to improve modelling capacity for similar axes
#' @param alpha_sample Joining the alpha_sample parameters. A vector of integers, where identical integers indicate same the data axis to be joined.
#' @param beta_sample Joining the beta_sample parameters. A vector of integers, where identical integers indicate same the data axis to be joined.
#' @param code Joining the code parameters. A vector of integers, where identical integers indicate same the data axis to be joined.
#' @export
extract_references_framework <- function(verbose=T){
  references <- list(data_list=NULL)
  
  if (verbose == T){
    print(references)
  }
  
  return(references)
}
