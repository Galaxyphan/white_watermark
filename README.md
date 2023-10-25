Embedding Watermarks into Deep Neural Networks
====
This code is the pytorch implementation of "Embedding Watermarks into Deep Neural Networks" [1]. It embeds a digital watermark into deep neural networks in training the host network. This embedding is achieved by a parameter regularizer.

## Usage
Embed a watermark in training a host network:

```sh
# train the host network while embedding a watermark
# In the configuration file for training, train_random_min.json only trains three rounds (epoch=3) and train_random.json training for 200 rounds
python train_watermark.py 

# extract the embedded watermark Note: This file can only be run on the command line. 
# And the file with the second parameter is the ~w.npy ending (watermark matrix)
python val_watermark.py result/filename[you_should_change_it].pth result/filename[you_should_change_it]_w.npy result/random
```

Train the host network *without* embedding:

```sh
# train the host network without embedding
# train with config/train_non_min.json(target_blk_id=0)
python train_wrn.py  

# extract the embedded watermark (meaningless because no watermark was embedded)
python val_watermark.py result/.pth result/.npy result/non

# visualize the embedded watermark
python draw_histogram_signature.py config/draw_histogram_non.json hist_signature_non.png
```

## References
[1] Y. Uchida, Y. Nagai, S. Sakazawa, and S. Satoh, "Embedding Watermarks into Deep Neural Networks," ICMR, 2017.