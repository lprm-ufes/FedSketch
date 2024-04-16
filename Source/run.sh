num_rounds=50

dataset="FAMNIST"
for d in 12 25 37 50
do
    echo "\n*Iniciando execução Sketch random\n"
    python3 server_publish_sketch_pytorch.py $num_clients $d $num_rounds "random" "increasing" &
    sleep 5

    for ((i=1; i<=num_clients; i++)); 
    do 
        i=$((i))
        python3 client_publish_sketch_pytorch.py 500 $d $dataset & 
    done
    echo "\n*Finalizando execução Sketch random\n"
    wait
done

echo "\n*Iniciando execução Sketch client selection sketch\n"
python3 server_publish_sketch_pytorch.py $num_clients $num_clients $num_rounds "sketch" "increasing" &
sleep 5
for ((i=1; i<=$num_clients; i++)); 
do 
    i=$((i))
    echo $i
    python3 client_publish_sketch_pytorch.py 500 $num_clients $dataset & 
done
wait
echo "\n*Finalizando execução Sketch client selection sketch\n"

echo "\n*Iniciando execução Sketch client selection accuracy\n"
python3 server_publish_sketch_pytorch.py $num_clients $num_clients $num_rounds "accuracy" "increasing" &
sleep 5
for ((i=1; i<=$num_clients; i++)); 
do 
    i=$((i))
    python3 client_publish_sketch_pytorch.py 500 $num_clients $dataset & 
done
echo "\n*Finalizando execução Sketch client selection sketch\n"
wait

dataset="MNIST"
for d in 12 25 37 50
do
    echo "\n*Iniciando execução Sketch random\n"
    python3 server_publish_sketch_pytorch.py $num_clients $d $num_rounds "random" "increasing" &
    sleep 5

    for ((i=1; i<=num_clients; i++)); 
    do 
        i=$((i))
        python3 client_publish_sketch_pytorch.py 500 $d $dataset & 
    done
    echo "\n*Finalizando execução Sketch random\n"
    wait
done

echo "\n*Iniciando execução Sketch client selection sketch\n"
python3 server_publish_sketch_pytorch.py $num_clients $num_clients $num_rounds "sketch" "increasing" &
sleep 5
for ((i=1; i<=$num_clients; i++)); 
do 
    i=$((i))
    echo $i
    python3 client_publish_sketch_pytorch.py 500 $num_clients $dataset & 
done
wait

echo "\n*Finalizando execução Sketch client selection sketch\n"

echo "\n*Iniciando execução Sketch client selection accuracy\n"
python3 server_publish_sketch_pytorch.py $num_clients $num_clients $num_rounds "accuracy" "increasing" &
sleep 5
for ((i=1; i<=$num_clients; i++)); 
do 
    i=$((i))
    python3 client_publish_sketch_pytorch.py 500 $num_clients $dataset & 
done
echo "\n*Finalizando execução Sketch client selection sketch\n"
wait

dataset="CIFAR10"

for d in 12 25 37 50
do
    echo "\n*Iniciando execução Sketch random\n"
    python3 server_publish_sketch_pytorch.py $num_clients $d $num_rounds "random" "increasing" &
    sleep 5

    for ((i=1; i<=num_clients; i++)); 
    do 
        i=$((i))
        python3 client_publish.py 1000 $d $dataset & 
    done
    echo "\n*Finalizando execução Sketch random\n"
    wait
done


echo "\n*Iniciando execução Sketch client selection sketch\n"
python3 server_publish_sketch_pytorch.py $num_clients $num_clients $num_rounds "sketch" "increasing" &
sleep 5
for ((i=1; i<=$num_clients; i++)); 
do 
    i=$((i))
    echo $i
    python3 client_publish_sketch_pytorch.py 1000 $num_clients $dataset & 
done
wait
echo "\n*Finalizando execução Sketch client selection sketch\n"

echo "\n*Iniciando execução Sketch client selection accuracy\n"
python3 server_publish_sketch_pytorch.py $num_clients $num_clients $num_rounds "accuracy" "increasing" &
sleep 5
for ((i=1; i<=$num_clients; i++)); 
do 
    i=$((i))
    python3 client_publish_sketch_pytorch.py 1000 $num_clients $dataset & 
done
echo "\n*Finalizando execução Sketch client selection sketch\n"
wait


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
