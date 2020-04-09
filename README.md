# Model Zoo

[![Build Status](https://travis-ci.com/dangne/model-zoo.svg?branch=master)](https://travis-ci.com/github/dangne/model-zoo) [![GitHub issues](https://img.shields.io/github/issues/dangne/model-zoo.svg)](https://GitHub.com/dangne/model-zoo/issues/) ![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)



This is an educative project beneficial for those who want to learn how to implement ML models either from scratch or using libraries and/or frameworks.

Each model has 3 different implementations with increasing level of complexity:

- **Level 1**: Use high-level API or library to build models.
- **Level 2**: Custom models with Model subclassing and custom training loop with GradientTape.
- **Level 3**: Completely from scratch with pure numpy code :)



## Install dependencies

Currently, this project only supports scikit-learn and Tensorflow. 

```bash
pip install -r requirements.txt
```



## Models

| Models               | Level 1 | Level 2 | Level 3 |
| -------------------- | :-----: | :-----: | :-----: |
| Linear Regression    |         |         |         |
| Logistics Regression |         |         |         |
| SVM                  |         |         |         |
| Decision Tree        |         |         |         |
| PCA                  |         |         |         |
| Naive Bayes          |         |         |         |
| Random Forest        |         |         |         |
| Deep NN              |    ✓    |         |         |
| CNN                  |    ✓    |         |         |
| RNN                  |         |         |         |
| LSTM                 |         |         |         |
| GRU                  |         |         |         |
| BD-RNN               |         |         |         |
| Attention            |         |         |         |
| ...                  |         |         |         |



# License

This project is licensed under the [MIT](https://github.com/dangne/model-zoo/blob/master/LICENSE) License.