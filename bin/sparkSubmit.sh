#!/bin/bash

spark-submit --master spark://172.17.0.2:7077 --properties-file "spark.conf" oblicz.py "/home/karol/pdd/duzeZadanie2/grafy/parameters.json" 2>/dev/null