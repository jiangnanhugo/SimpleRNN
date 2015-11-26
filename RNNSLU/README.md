Investigation of Recurrent Neural Network Architectures and Learning Methods for Spoken Language Understanding
==============================================================================================================

### Code for RNN and Spoken Language Understanding

Based on the Interspeech '13 paper:

[Grégoire Mesnil, Xiaodong He, Li Deng and Yoshua Bengio - **Investigation of Recurrent Neural Network Architectures and Learning Methods for Spoken Language Understanding**](http://www.iro.umontreal.ca/~lisa/pointeurs/RNNSpokenLanguage2013.pdf)

We also have a follow-up IEEE paper:

[Grégoire Mesnil, Yann Dauphin, Kaisheng Yao, Yoshua Bengio, Li Deng, Dilek Hakkani-Tur, Xiaodong He, Larry Heck, Gokhan Tur, Dong Yu and Geoffrey Zweig - **Using Recurrent Neural Networks for Slot Filling in Spoken Language Understanding**](http://www.iro.umontreal.ca/~lisa/pointeurs/taslp_RNNSLU_final_doubleColumn.pdf)

# 实验操作流程：

## 数据预处理 

>下载 ATIS dataset: 

>>training set : 4978 sentences/ 56590 words;

>>testing set: 893 sentences/ 9198 words.

>统计training set里面的单词，建立 vocabulary dictionary; 

>>对 training set只出现一次的单词标记为 `<UNK>`;
 对不在training set中，却出现在testing set中的word,将其转换成`<UNK>`;

>>对整个dataset中的数字，全部转换成DIGIT;最后再更新vocabulary dictionary;

> 将每个word 转换成 "one-hot" representation的向量，其维度为vocabulary dictionary的length.

> 下一步就是建立context windows:

>>对于第i个word, 其词向量为 x(i) ,其对应的context representation 为 [x(i-n),...,x(i),...,x(i+n)], context window size=n

>> 对于边界单词 如x(0),其对应的左边界为x(-1), 所以可以表示为 [x(-1),x(-1)...,x(0),x(1),....,x(n)]


## 实验结果
NEW BEST: epoch 0 valid F1 85.71 best test F1 79.89                     
NEW BEST: epoch 1 valid F1 91.78 best test F1 89.42                     
NEW BEST: epoch 2 valid F1 93.54 best test F1 90.82                     
NEW BEST: epoch 3 valid F1 95.12 best test F1 92.27                     
NEW BEST: epoch 4 valid F1 95.4 best test F1 92.33                     
[learning] epoch 5 >> 100.00% completed in 319.63 (sec) <<
[learning] epoch 6 >> 100.00% completed in 321.81 (sec) <<
NEW BEST: epoch 7 valid F1 95.58 best test F1 92.58                     
[learning] epoch 8 >> 100.00% completed in 357.68 (sec) <<
NEW BEST: epoch 9 valid F1 96.18 best test F1 93.63                     
NEW BEST: epoch 10 valid F1 96.56 best test F1 92.65                     
[learning] epoch 11 >> 100.00% completed in 360.72 (sec) <<
[learning] epoch 12 >> 100.00% completed in 355.01 (sec) <<
[learning] epoch 13 >> 100.00% completed in 354.26 (sec) <<
[learning] epoch 14 >> 100.00% completed in 364.73 (sec) <<
[learning] epoch 15 >> 100.00% completed in 355.45 (sec) <<
[learning] epoch 16 >> 100.00% completed in 357.52 (sec) <<
[learning] epoch 17 >> 100.00% completed in 353.49 (sec) <<
[learning] epoch 18 >> 100.00% completed in 348.97 (sec) <<
[learning] epoch 19 >> 100.00% completed in 342.46 (sec) <<
[learning] epoch 20 >> 100.00% completed in 343.87 (sec) <<
NEW BEST: epoch 21 valid F1 96.63 best test F1 93.0                     
[learning] epoch 22 >> 100.00% completed in 345.02 (sec) <<
[learning] epoch 23 >> 100.00% completed in 342.41 (sec) <<
NEW BEST: epoch 24 valid F1 96.73 best test F1 93.64                     
NEW BEST: epoch 25 valid F1 97.15 best test F1 94.75                     
[learning] epoch 26 >> 100.00% completed in 349.28 (sec) <<
[learning] epoch 27 >> 100.00% completed in 348.13 (sec) <<
[learning] epoch 28 >> 100.00% completed in 348.53 (sec) <<
NEW BEST: epoch 29 valid F1 97.16 best test F1 93.92                     
[learning] epoch 30 >> 100.00% completed in 343.07 (sec) <<
[learning] epoch 31 >> 100.00% completed in 341.08 (sec) <<
[learning] epoch 32 >> 100.00% completed in 341.87 (sec) <<
[learning] epoch 33 >> 100.00% completed in 343.02 (sec) <<
[learning] epoch 34 >> 100.00% completed in 342.53 (sec) <<
[learning] epoch 35 >> 100.00% completed in 342.30 (sec) <<
NEW BEST: epoch 36 valid F1 97.54 best test F1 93.9                     
[learning] epoch 37 >> 100.00% completed in 341.54 (sec) <<
[learning] epoch 38 >> 100.00% completed in 343.68 (sec) <<
[learning] epoch 39 >> 100.00% completed in 345.23 (sec) <<
[learning] epoch 40 >> 100.00% completed in 342.62 (sec) <<
[learning] epoch 41 >> 100.00% completed in 343.35 (sec) <<
[learning] epoch 42 >> 100.00% completed in 341.36 (sec) <<
[learning] epoch 43 >> 100.00% completed in 341.72 (sec) <<
[learning] epoch 44 >> 100.00% completed in 351.70 (sec) <<
[learning] epoch 45 >> 100.00% completed in 350.74 (sec) <<
[learning] epoch 46 >> 100.00% completed in 351.31 (sec) <<
[learning] epoch 47 >> 100.00% completed in 348.79 (sec) <<
[learning] epoch 48 >> 100.00% completed in 349.97 (sec) <<
[learning] epoch 49 >> 100.00% completed in 349.11 (sec) <<
BEST RESULT: epoch 49 valid F1 97.54 best test F1 93.9 with the model elman-forward



## Code

This code allows to get state-of-the-art results and a significant improvement
(+1% in F1-score) with respect to the results presented in the paper.

In order to reproduce the results, make sure Theano is installed and the
repository is in your `PYTHONPATH`, e.g run the command
`export PYTHONPATH=/path/where/is13/is:$PYTHONPATH`. Then, run the following
commands:


For running the elman architecture:

```
python elman-forward.py
```

## ATIS Data: 
already downloaded & conlleval.pl script.






`dicts` is a python dictionnary that contains the mapping from the labels, the
name entities (if existing) and the words to indexes used in `train` and `test`
lists. Refer to this [tutorial](http://deeplearning.net/tutorial/rnnslu.html) for more details. 

Running the following command can give you an idea of how the data has been preprocessed:

```
python data/load.py
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Recurrent Neural Network Architectures for Spoken Language Understanding</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Grégoire Mesnil</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/mesnilgr/is13" rel="dct:source">https://github.com/mesnilgr/is13</a>.



