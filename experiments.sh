 #!/bin/bash

for EMB_SIZE in 256 512 1024
do
    for H_SIZE in 256 512 1024
    do
        python main.py --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch 350 --batch_size 512 --eval_batch_size 1024
    done
done

python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --teacher_forcing_ratio 0 --batch_size 512 --eval_batch_size 1024
python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --teacher_forcing_ratio 0.2 --batch_size 512 --eval_batch_size 1024
python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --teacher_forcing_ratio 0.5 --batch_size 512 --eval_batch_size 1024
python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --teacher_forcing_ratio 1 --batch_size 512 --eval_batch_size 1024

python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --bidirectional --batch_size 512 --eval_batch_size 1024
python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --n_layers 2 --batch_size 512 --eval_batch_size 1024
python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --n_layers 3 --batch_size 512 --eval_batch_size 1024

python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --attention 'pre-rnn' --attention_method 'dot' --full_focus --batch_size 512 --eval_batch_size 1024
python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --attention 'pre-rnn' --attention_method 'mlp' --full_focus --batch_size 512 --eval_batch_size 1024
python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --attention 'pre-rnn' --attention_method 'concat' --full_focus --batch_size 512 --eval_batch_size 1024

python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --attention 'post-rnn' --attention_method 'dot' --full_focus --batch_size 512 --eval_batch_size 1024
python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --attention 'post-rnn' --attention_method 'mlp' --full_focus --batch_size 512 --eval_batch_size 1024
python main.py --embedding_size 512 --hidden_size 512 --epoch 350 --attention 'post-rnn' --attention_method 'concat' --full_focus --batch_size 512 --eval_batch_size 1024
