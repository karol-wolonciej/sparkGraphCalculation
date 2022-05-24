#!/bin/bash

export SPARK_MASTER_OPTS="-Dspark.deploy.defaultCores=4"

spark-submit --master spark://172.17.0.2:7077 --properties-file "spark.conf" analiza.py