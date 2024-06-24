docker-compose down
docker-compose up -d

echo "Ожидание запуска контейнеров..."
sleep 40

docker cp /Users/sonya.cherepennikova/Documents/spark_lab/combined_ny_taxi.csv namenode:/

docker exec -it namenode bash
hdfs dfs -mkdir -p /data/ny_taxi/
hdfs dfs -put /tmp/combined_ny_taxi.csv /data/ny_taxi/
exit

docker exec -it spark-worker-1 bash
pip install pyspark
exit

docker cp /Users/sonya.cherepennikova/Documents/spark_lab/app.py spark_master:/
docker exec -it master bash

spark/bin/spark-submit spark_app.py