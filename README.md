# Aspect-Based-Sentiment-Analysis

`Original Repository`: [URL](https://github.com/lixin4ever/BERT-E2E-ABSA)

BERT-E2E-ABSA 모델을 따라 작성하였습니다. [Origin Paper](https://arxiv.org/pdf/1910.00883.pdf)는 2019년 발표된 논문으로 ABSA 연구에 BERT를 이용하여 End-to-End 방식으로 ABSA를 수행하였습니다. 한글로 코드를 뜯어보며 어떻게 작동되는지 알아보고자 합니다.


# SemEval

`SemEval` 데이터는 ABSA 연구에서 주로 사용되는 데이터셋이며, Restaurant, Laptop에 관련된 리뷰


| **Dataset** | **Train** | **Valid** | **Test** |
|--------:|:--------:|:--------:|:--------:|
| **#Total Sentence** | 5,959 | 851 | 1,703 |
| **#Positive** | 3,050 | 422 | 744 |
| **#Negatvie** | 1,181 | 137 | 290 |
| **#Neutral** | 673 | 20 | 60 |
| **#Total** | 4,904 | 579 | 1,094 |

# Docker

1.clone this repository

```
git clone https://github.com/ceo21ckim/Aspect-Based-Sentiment-Analysis.git
cd Aspect-Based-Sentiment-Analysis
```

2.build Dockerfile

```
docker build --tag [filename]
```

3.execute

```
docker run -itd --gpus all --name [NAME] -p 8888:8888 -v [PATH]:/workspace [filename] /bin/bash
```

4.use jupyter notebook

```
docker exec -it [NAME] bash 

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

# Execute

```
python absa_train.py

```
