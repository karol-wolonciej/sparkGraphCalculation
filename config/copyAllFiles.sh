#!/bin/bash


dockers=("master" "slave1" "slave2" "slave3" "slave4")

pathInDockers="/home/karol/pdd/duzeZadanie2/grafy"



for dockerId in "${dockers[@]}"
do
	sudo docker exec -d $dockerId mkdir -p /home/karol/pdd/duzeZadanie2/grafy
	sudo docker cp . "$dockerId:/$pathInDockers"
done
