#!/bin/bash


# reset cluster
sudo docker stop master
sudo docker start master

# create dockers containers table
SLAVES_COUNT=12

dockers=("master")

for (( c=1; c<=$SLAVES_COUNT; c++ ))
do
	dockers+=("slave$c")
done

printf '%s\n' "${dockers[@]}"

SETS_PATH="/home/karol/pdd/duzeZadanie2/sets/*"
PROJECT_PATH=$(pwd)

IFS=:
for p in $SETS_PATH; do
    PARAMETERS_FILE="$p/parameters.json"
    NAME=$(basename $p)0
    CONFIG_FILE="$PROJECT_PATH/config/spark.conf"
    OBLICZ_FILE="$PROJECT_PATH/bin/oblicz.py"
    PREVIOUS_DICT="$p/$NAME.pkl"
    echo "spark-submit --master spark://172.17.0.2:7077 --name $NAME --properties-file $CONFIG_FILE $OBLICZ_FILE $PARAMETERS_FILE 2>/dev/null &"
    rm $PREVIOUS_DICT 2>/dev/null
    spark-submit --master spark://172.17.0.2:7077 --name $NAME --properties-file $CONFIG_FILE $OBLICZ_FILE $PARAMETERS_FILE 2>/dev/null &
done

wait