 #!/bin/bash

python main.py --bidirectional --epoch 200 --embedding_size 512 --hidden_size 500
python main.py --bidirectional --epoch 200 --embedding_size 512 --hidden_size 500 --positional_attention
python main.py --bidirectional --epoch 200 --embedding_size 512 --hidden_size 500 --positional_attention --attention none