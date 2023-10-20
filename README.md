# RoBERTa
## REF
>Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., Grangier, D., & Auli, M. (2019). fairseq: A Fast, Extensible Toolkit for Sequence Modeling. Proceedings of NAACL-HLT 2019: Demonstrations.

## 相关大文件下载
1. conda package下载：https://pan.seu.edu.cn:443/link/313EB706AB4CBCE67E506B12961BA25F
有效期限：2028-10-18
访问密码：kFky
2. 示例数据集下载：https://pan.seu.edu.cn:443/link/8D2B0CC3B6B3E4AC825250123BAF0745
有效期限：2028-10-18
访问密码：IEea
3. 下载完成后请直接放置在当前文件夹，使用`pwd`命令查看当前目录，应该为`....../RoBERTa`
## conda 环境解压

1. 创建目录 `RoBERTa`，并将环境解压至该目录：

    ```shell
    mkdir -p RoBERTa
    tar -xzf RoBERTa.tar.gz -C RoBERTa
    ```

2. 使用 Python 而不激活或修复前缀。大多数 Python 库可以正常工作，但需要处理前缀的部分将失败：

    ```shell
    ./RoBERTa/bin/python
    ```

3. 激活环境，同时这步操作会将路径 `RoBERTa/bin` 添加到环境变量 path：

    ```shell
    source RoBERTa/bin/activate
    ```

4. 在环境中运行 Python：

    ```shell
    (RoBERTa) $ python
    ```
   
5. 使用完成后，停用环境以将其从环境变量 path 中删除
   ```shell
   (RoBERTa) $ source RoBERTa/bin/deactivate
   ```
## 预训练
### 1.  预处理数据

数据应该按照语言建模格式进行预处理，即每个文档之间应该用空行分隔（只在使用 --sample-break-mode complete_doc 时有用）。在训练期间，这些行将被连接成一个一维文本流。

这里使用 WikiText-103 数据集 来演示如何使用 GPT-2 BPE 预处理原始文本数据。当然，这个数据集相当小，因此生成的预训练模型性能将较差，但它可以提供一个一般的概念。

#### 首先移动并解压数据集
   ```bash
  mv wikitext-103-raw-v1.zip ./fairseq
  cd fairseq
  unzip wikitext-103-raw-v1.zip
   ```
####  使用 GPT-2 BPE 进行编码
```bash
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
```
更换数据时，请将数据放入一个文件夹内，如创建`mydata`文件夹，将`data.train.raw` `data.test.raw`和`data.valid.raw`放入该文件夹，并将`inputs`参数行改为`--inputs mydata/data.${SPLIT}.raw \`，对应`outputs`参数行改为`--outputs mydata/data.${SPLIT}.bpe \`

#### 使用 GPT-2 fairseq 字典进行预处理/二进制化数据
```bash
wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60
```
此处`trainpref` `validpref` `testpref`和`destdir`参数注意更改，与上一步同理
### 2. 预训练模型
这里指定`DATA_DIR`相对路径可能会出问题，建议使用绝对路径，注意更改`DATA_DIR`
```bash
DATA_DIR=data-bin/wikitext-103

fairseq-hydra-train -m --config-dir examples/roberta/config/pretraining \
--config-name base task.data=$DATA_DIR
```
### 3. 加载预训练模型
```python
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'path/to/data')
assert isinstance(roberta.model, torch.nn.Module)
```

