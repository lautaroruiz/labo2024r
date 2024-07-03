# Ensemble de arboles de decision
# utilizando el naif metodo de Arboles Azarosos
# entreno cada arbol utilizando un subset distinto de atributos del dataset

# mandatoriamente debe correr en Google Cloud
# sube automaticamente a Kaggle

# limpio la memoria
rm(list = ls()) # Borro todos los objetos
gc() # Garbage Collection

require("data.table")
require("rpart")

# parametros experimento
PARAM <- list()
PARAM$experimento <- 3910

# parametros rpart

# cambiar aqui por SUS corridas 
#  segun lo que indica la  Planilla Colaborativa
#PARAM$corridas <- data.table(
#  "cp" = rep(-1, 8),
#  "minsplit" = c(50, 100, 100, 250, 500, 500, 1000, 1000), 
#  "minbucket" = c(5, 5, 50, 20, 5, 50, 10, 100), 
#  "maxdepth" = c(10, 10, 8, 8, 6, 6, 10, 10) 
#)
#PARAM$corridas <- data.table(
#  "cp" = c(-1),
#  "minsplit" = c(100), 
#  "minbucket" = c(10), 
#  "maxdepth" = c(10) 
#)
PARAM$corridas <- data.table(
  "cp" = c(-0.59224974),
  "minsplit" = c(1054), 
  "minbucket" = c(525), 
  "maxdepth" = c(8) 
)

#-0.59224974	1054	525	8 1

# parametros  arbol
# entreno cada arbol con solo 50% de las variables variables
#  por ahora, es fijo
PARAM$feature_fraction <- 0.5


# voy a generar 500 arboles,
#  a mas arboles mas tiempo de proceso y MEJOR MODELO,
#  pero ganancias marginales
PARAM$num_trees_max <- 500

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui comienza el programa

setwd("C:/Users/lauta/Desktop/Lautaro/maestria_ciencia_datos/labo_1/") # Establezco el Working Directory

# cargo MI semilla, que esta en MI bucket
#tabla_semillas <- fread( "./datasets//mis_semillas.txt" )
ksemilla_azar <- 100019#tabla_semillas[ 1, semilla ]  # 1 es mi primer semilla

# cargo los datos
dataset <- fread("./datasets/dataset_pequeno.csv")


# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
carpeta_experimento <- paste0("./exp/KA", PARAM$experimento, "/")
dir.create(paste0("./exp/KA", PARAM$experimento, "/"),
           showWarnings = FALSE
)

setwd(carpeta_experimento)


# que tamanos de ensemble grabo a disco, pero siempre debo generar los 500
grabar <- c(1, 5, 10, 50, 100, 200, 500)


# defino los dataset de entrenamiento y aplicacion
dtrain <- dataset[foto_mes == 202107]
dapply <- dataset[foto_mes == 202109]

# arreglo clase_ternaria por algun distraido ""
dapply[, clase_ternaria := NA ]

# elimino lo que ya no utilizo
rm(dataset)
gc()

# Establezco cuales son los campos que puedo usar para la prediccion
# el copy() es por la Lazy Evaluation
campos_buenos <- copy(setdiff(colnames(dtrain), c("clase_ternaria")))



# Genero las salidas
for( icorrida in seq(nrow(PARAM$corridas)) ){
  
  cat( "Corrida ", icorrida, " ; " )
  
  # aqui se va acumulando la probabilidad del ensemble
  dapply[, prob_acumulada := 0]
  
  # los parametros que voy a utilizar para rpart
  param_rpart <- PARAM$corridas[ icorrida ]
  
  set.seed(ksemilla_azar) # Establezco la semilla aleatoria
  
  for (arbolito in seq(PARAM$num_trees_max) ) {
    qty_campos_a_utilizar <- as.integer(length(campos_buenos)
                                        * PARAM$feature_fraction)
    
    campos_random <- sample(campos_buenos, qty_campos_a_utilizar)
    
    # paso de un vector a un string con los elementos
    # separados por un signo de "+"
    # este hace falta para la formula
    campos_random <- paste(campos_random, collapse = " + ")
    
    # armo la formula para rpart
    formulita <- paste0("clase_ternaria ~ ", campos_random)
    
    # genero el arbol de decision
    modelo <- rpart(formulita,
                    data = dtrain,
                    xval = 0,
                    control = param_rpart
    )
    
    # aplico el modelo a los datos que no tienen clase
    prediccion <- predict(modelo, dapply, type = "prob")
    
    dapply[, prob_acumulada := prob_acumulada + prediccion[, "BAJA+2"]]
    
    if (arbolito %in% grabar) {
      # Genero la entrega para Kaggle
      umbral_corte <- (1 / 40) * arbolito
      entrega <- as.data.table(list(
        "numero_de_cliente" = dapply[, numero_de_cliente],
        "Predicted" = as.numeric(dapply[, prob_acumulada] > umbral_corte)
      )) # genero la salida
      
      nom_arch_kaggle <- paste0(
        "KA", PARAM$experimento, "_",
        icorrida, "_2_", #le agrego el 2 para la optimizacion bayesiana
        sprintf("%.3d", arbolito), # para que tenga ceros adelante
        ".csv"
      )
      
      # grabo el archivo 
      fwrite(entrega,
             file = nom_arch_kaggle,
             sep = ","
      )
      
    }
    
    cat(arbolito, " ")
  }
}
