wget -O 'ckpt.hdf5' 'https://www.dropbox.com/s/ydk7wgcor8sbagk/ckpt.hdf5?dl=1'
wget -O 'word2vec.model' 'https://www.dropbox.com/s/rr5bsf4q9t11ncq/word2vec.model?dl=1'
wget -O 'word2vec.model.trainables.syn1neg.npy' 'https://www.dropbox.com/s/gtxg3n97zqj7v6p/word2vec.model.trainables.syn1neg.npy?dl=1'
wget -O 'word2vec.model.wv.vectors.npy' 'https://www.dropbox.com/s/d168if4cs5czmiw/word2vec.model.wv.vectors.npy?dl=1'
python3 test.py $1 $2 $3