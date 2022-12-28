# Aspect-Based-Sentiment-Analysis

`Original Repository`: [URL](https://github.com/lixin4ever/BERT-E2E-ABSA)

BERT-E2E-ABSA 모델을 따라 작성하였습니다. [Origin Paper](https://arxiv.org/pdf/1910.00883.pdf)는 2019년 발표된 논문으로 ABSA 연구에 BERT를 이용하여 End-to-End 방식으로 ABSA를 수행하였습니다. 한글로 코드를 뜯어보며 어떻게 작동되는지 알아보고자 합니다.

`Dockerfile`: Docker를 사용하시는 분들은 `Dockerfile`을 호출해 동일한 환경에서 사용이 가능합니다. cuda 버전을 확인하세요.

`absa_bert`: BERT-E2E-ABSA 기법 중 Attention Network 기법이 가장 우수한 성능을 발휘하고 있기 때문에, SAN을 기반으로 구축한 모델입니다. 

`parser`: 학습에 필요한 파라미터들을 설정값을 담고 있습니다.

`run.py`: SemEval 데이터의 성능을 testing하는 코드입니다. 

`train.py`: BERT-E2E-ABSA 모델을 학습하는 코드입니다.

`settings.py`: 실험에 필요한 경로들을 설정하였습니다. 

`run.ipynb`: SemEval 데이터로 학습한 후 Yelp.com 데이터를 이용해 inference하는 코드입니다. 

`utils.py`: 실험에 필요한 함수들을 저장해두었습니다. 


# SemEval

`SemEval` 데이터는 ABSA 연구에서 주로 사용되는 데이터셋이며, Restaurant, Laptop에 관련된 리뷰로 구성되어 있으나, Restaurant 관련 리뷰만 사용하였습니다. 추가로 활용된 `Yelp.com` 데이터는 2018년도 데이터이며 샘플로 5만개를 추출하여 사용하였습니다. 


| **Dataset** | **Train** | **Valid** | **Test** |
|--------:|:--------:|:--------:|:--------:|
| **#Total Sentence** | 5,959 | 851 | 1,703 |
| **#Positive** | 3,050 | 422 | 744 |
| **#Negatvie** | 1,181 | 137 | 290 |
| **#Neutral** | 673 | 20 | 60 |
| **#Total** | 4,904 | 579 | 1,094 |

# Docker

**1.clone this repository**

```
git clone https://github.com/ceo21ckim/Aspect-Based-Sentiment-Analysis.git
cd Aspect-Based-Sentiment-Analysis
```

**2.build Dockerfile**

```
docker build --tag [filename]
```

**3.execute**

```
docker run -itd --gpus all --name [NAME] -p 8888:8888 -v [PATH]:/workspace [filename] /bin/bash
```

**4.use jupyter notebook**

```
docker exec -it [NAME] bash 

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

# Execute

```
python absa_train.py

```
