#!/usr/bin/env Rscript

# Workflow  Data Drifting repair

# inputs
#  * gran dataset
#  * metodo para tratar el data drifting
# output  
#   gran dataset :
#     misma cantidad de registros
#     misma cantidad de columnas
#     los valores de los campos SE modifican para corregir el data drifting


# limpio la memoria
rm(list = ls(all.names = TRUE)) # remove all objects
gc(full = TRUE) # garbage collection

require("data.table")
require("yaml")

#cargo la libreria
# args <- c( "~/labo2024r" )
args <- commandArgs(trailingOnly=TRUE)
source( paste0( args[1] , "/src/lib/action_lib.r" ) )

#------------------------------------------------------------------------------
# deflaciona por IPC
# momento 1.0  31-dic-2020 a las 23:59

drift_deflacion <- function(campos_monetarios) {
  cat( "inicio drift_deflacion()\n")
  vfoto_mes <- c(
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 202104, 202105, 202106,
    202107, 202108, 202109
  )

  vIPC <- c(
    1.9903030878, 1.9174403544, 1.8296186587,
    1.7728862972, 1.7212488323, 1.6776304408,
    1.6431248196, 1.5814483345, 1.4947526791,
    1.4484037589, 1.3913580777, 1.3404220402,
    1.3154288912, 1.2921698342, 1.2472681797,
    1.2300475145, 1.2118694724, 1.1881073259,
    1.1693969743, 1.1375456949, 1.1065619600,
    1.0681100000, 1.0370000000, 1.0000000000,
    0.9680542110, 0.9344152616, 0.8882274350,
    0.8532444140, 0.8251880213, 0.8003763543,
    0.7763107219, 0.7566381305, 0.7289384687
  )

  tb_IPC <- as.data.table( list( vfoto_mes, vIPC) )

 colnames( tb_IPC ) <- c( envg$PARAM$dataset_metadata$periodo, "IPC" )

  dataset[tb_IPC,
    on = c(envg$PARAM$dataset_metadata$periodo),
    (campos_monetarios) := .SD * i.IPC,
    .SDcols = campos_monetarios
  ]

  cat( "fin drift_deflacion()\n")
}

#------------------------------------------------------------------------------
# deflaciona por UVA
# momento 1.0  31-dic-2020 a las 23:59

drift_uva <- function(campos_monetarios) {
  cat( "inicio drift_uva()\n")
  vfoto_mes <- c(
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 202104, 202105, 202106,
    202107, 202108, 202109
  )
  
  vUVA <- c(
    2.001408838932958, 1.950325472789153, 1.89323032351521,
    1.8247220405493787, 1.746027787673673, 1.6871348409529485,
    1.6361678865622313, 1.5927529755859773, 1.5549162794128493,
    1.4949100586391746, 1.4197729500774545, 1.3678188186372326,
    1.3136508617223726, 1.2690535173062818, 1.2381595983200178,
    1.211656735577568, 1.1770808941405335, 1.1570338657445522,
    1.1388769475653255, 1.1156993751209352, 1.093638313080772,
    1.0657171590878205, 1.0362173587708712, 1.0,
    0.9669867858358365, 0.9323750098728378, 0.8958202912590305,
    0.8631993702994263, 0.8253893405524657, 0.7928918905364516,
    0.7666323845128089, 0.7428976357662823, 0.721615762047849,
    0.7027397112961563, 0.6800365611054359, 0.6575083343953959
  )
  
  tb_UVA <- as.data.table( list( vfoto_mes, vUVA) )
  
  colnames( tb_UVA ) <- c( envg$PARAM$dataset_metadata$periodo, "UVA" )
  
  dataset[tb_UVA,
          on = c(envg$PARAM$dataset_metadata$periodo),
          (campos_monetarios) := .SD * i.UVA,
          .SDcols = campos_monetarios
  ]
  
  cat( "fin drift_uva()\n")
}

#------------------------------------------------------------------------------
# dolar blue
# momento 1.0  31-dic-2020 a las 23:59

drift_dol_blue <- function(campos_monetarios) {
  cat( "inicio drift_dol_blue()\n")
  vfoto_mes <- c(
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 202104, 202105, 202106,
    202107, 202108, 202109
  )
  
  vDolBlue <- c(
    39.045455, 38.402500, 41.639474, 44.274737, 46.095455,
    45.063333, 43.983333, 54.842857, 61.059524, 65.545455,
    66.750000, 72.368421, 77.477273, 78.191667, 82.434211,
    101.087500, 126.236842, 125.857143, 130.782609, 133.400000,
    137.954545, 170.619048, 160.400000, 153.052632, 157.900000,
    149.380952, 143.615385, 146.250000, 153.550000, 162.000000,
    178.478261, 180.878788, 184.357143
  )
  
  tb_DolBlue <- as.data.table( list( vfoto_mes, vDolBlue) )
  
  colnames( tb_DolBlue ) <- c( envg$PARAM$dataset_metadata$periodo, "DolBlue" )
  
  dataset[tb_DolBlue,
          on = c(envg$PARAM$dataset_metadata$periodo),
          (campos_monetarios) := .SD * i.DolBlue,
          .SDcols = campos_monetarios
  ]
  
  cat( "fin drift_dol_blue()\n")
}

#------------------------------------------------------------------------------
# dolar oficial minorista
# momento 1.0  31-dic-2020 a las 23:59

drift_dol_mino <- function(campos_monetarios) {
  cat( "inicio drift_dol_mino()\n")
  vfoto_mes <- c(
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 202104, 202105, 202106,
    202107, 202108, 202109
  )
  
  vDolMino <- c(
    38.430000, 39.428000, 42.542105, 44.354211, 46.088636,
    44.955000, 43.751429, 54.650476, 58.790000, 61.403182,
    63.012632, 63.011579, 62.983636, 63.580556, 65.200000,
    67.872000, 70.047895, 72.520952, 75.324286, 77.488500,
    79.430909, 83.134762, 85.484737, 88.181667, 91.474000,
    93.997778, 96.635909, 98.526000, 99.613158, 100.619048,
    101.619048, 102.569048, 103.781818, 104.766842, 105.831905, 107.438500
  )
  
  tb_DolMino <- as.data.table( list( vfoto_mes, vDolMino) )
  
  colnames( tb_DolMino ) <- c( envg$PARAM$dataset_metadata$periodo, "DolMino" )
  
  dataset[tb_DolMino,
          on = c(envg$PARAM$dataset_metadata$periodo),
          (campos_monetarios) := .SD * i.DolMino,
          .SDcols = campos_monetarios
  ]
  
  cat( "fin drift_dol_mino()\n")
}

#------------------------------------------------------------------------------

drift_rank_simple <- function(campos_drift) {
  
  cat( "inicio drift_rank_simple()\n")
  for (campo in campos_drift)
  {
    cat(campo, " ")
    dataset[, paste0(campo, "_rank") :=
      (frank(get(campo), ties.method = "random") - 1) / (.N - 1), by = eval(envg$PARAM$dataset_metadata$periodo)]
    dataset[, (campo) := NULL]
  }
  cat( "fin drift_rank_simple()\n")
}
#------------------------------------------------------------------------------
# El cero se transforma en cero
# los positivos se rankean por su lado
# los negativos se rankean por su lado

drift_rank_cero_fijo <- function(campos_drift) {
 
  cat( "inicio drift_rank_cero_fijo()\n")
  for (campo in campos_drift)
  {
    cat(campo, " ")
    dataset[get(campo) == 0, paste0(campo, "_rank") := 0]
    dataset[get(campo) > 0, paste0(campo, "_rank") :=
      frank(get(campo), ties.method = "random") / .N, by = eval(envg$PARAM$dataset_metadata$periodo)]

    dataset[get(campo) < 0, paste0(campo, "_rank") :=
      -frank(-get(campo), ties.method = "random") / .N, by = eval(envg$PARAM$dataset_metadata$periodo)]
    dataset[, (campo) := NULL]
  }
  cat("\n")
  cat( "fin drift_rank_cero_fijo()\n")
}
#------------------------------------------------------------------------------

drift_estandarizar <- function(campos_drift) {

  cat( "inicio drift_estandarizar()\n")
  for (campo in campos_drift)
  {
    cat(campo, " ")
    dataset[, paste0(campo, "_normal") := 
      (get(campo) -mean(campo, na.rm=TRUE)) / sd(get(campo), na.rm=TRUE),
      by = eval(envg$PARAM$dataset_metadata$periodo)]

    dataset[, (campo) := NULL]
  }
  cat( "fin drift_estandarizar()\n")
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui comienza el programa
cat( "1401_DR_corregir_drifting_009.r  START\n")
action_inicializar() 

# cargo el dataset donde voy a entrenar
# esta en la carpeta del exp_input y siempre se llama  dataset.csv.gz
# cargo el dataset
envg$PARAM$dataset <- paste0( "./", envg$PARAM$input, "/dataset.csv.gz" )
envg$PARAM$dataset_metadata <- read_yaml( paste0( "./", envg$PARAM$input, "/dataset_metadata.yml" ) )

cat( "lectura del dataset\n")
action_verificar_archivo( envg$PARAM$dataset )
cat( "Iniciando lectura del dataset\n" )
dataset <- fread(envg$PARAM$dataset)
cat( "Finalizada lectura del dataset\n" )

GrabarOutput()

# ordeno dataset
setorderv(dataset, envg$PARAM$dataset_metadata$primarykey)

# por como armé los nombres de campos,
#  estos son los campos que expresan variables monetarias
campos_monetarios <- colnames(dataset)
campos_monetarios <- campos_monetarios[campos_monetarios %like%
  "^(m|Visa_m|Master_m|vm_m)"]

# aqui aplico un metodo para atacar el data drifting
# hay que probar experimentalmente cual funciona mejor
switch(envg$PARAM$metodo,
  "ninguno"        = cat("No hay correccion del data drifting"),
  "rank_simple"    = drift_rank_simple(campos_monetarios),
  "rank_cero_fijo" = drift_rank_cero_fijo(campos_monetarios),
  "deflacion"      = drift_deflacion(campos_monetarios),
  "estandarizar"   = drift_estandarizar(campos_monetarios),
  "uva"            = drift_uva(campos_monetarios),
  "dol_blue"       = drift_dol_blue(campos_monetarios),
  "dol_mino"       = drift_dol_mino(campos_monetarios)
)


#------------------------------------------------------------------------------
# grabo el dataset
cat( "escritura del dataset\n")
cat( "Iniciando grabado del dataset\n" )
fwrite(dataset,
  file = "dataset.csv.gz",
  logical01 = TRUE,
  sep = ","
)
cat( "Finalizado grabado del dataset\n" )

# copia la metadata sin modificar
cat( "escritura de metadata\n")
write_yaml( envg$PARAM$dataset_metadata, 
  file="dataset_metadata.yml" )

#------------------------------------------------------------------------------

# guardo los campos que tiene el dataset
tb_campos <- as.data.table(list(
  "pos" = 1:ncol(dataset),
  "campo" = names(sapply(dataset, class)),
  "tipo" = sapply(dataset, class),
  "nulos" = sapply(dataset, function(x) {
    sum(is.na(x))
  }),
  "ceros" = sapply(dataset, function(x) {
    sum(x == 0, na.rm = TRUE)
  })
))

fwrite(tb_campos,
  file = "dataset.campos.txt",
  sep = "\t"
)

#------------------------------------------------------------------------------
cat( "Fin del programa\n")

envg$OUTPUT$dataset$ncol <- ncol(dataset)
envg$OUTPUT$dataset$nrow <- nrow(dataset)
envg$OUTPUT$time$end <- format(Sys.time(), "%Y%m%d %H%M%S")
GrabarOutput()

#------------------------------------------------------------------------------
# finalizo la corrida
#  archivos tiene a los files que debo verificar existen para no abortar

action_finalizar( archivos = c("dataset.csv.gz","dataset_metadata.yml")) 
cat( "1401_DR_corregir_drifting_009.r  END\n")