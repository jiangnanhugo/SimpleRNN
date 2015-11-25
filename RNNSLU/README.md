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

## ATIS Data: already downloaded & conlleval.pl script.




`dicts` is a python dictionnary that contains the mapping from the labels, the
name entities (if existing) and the words to indexes used in `train` and `test`
lists. Refer to this [tutorial](http://deeplearning.net/tutorial/rnnslu.html) for more details. 

Running the following command can give you an idea of how the data has been preprocessed:

```
python data/load.py
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Recurrent Neural Network Architectures for Spoken Language Understanding</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Grégoire Mesnil</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/mesnilgr/is13" rel="dct:source">https://github.com/mesnilgr/is13</a>.



