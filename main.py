import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# follow the https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough?hl=zh-cn

train_data = "adult.data"
test_data = "adult.test"


def pack_features_vector(features, labels):
    """将特征打包到一个数组中"""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


# 定义损失和梯度函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y):
  y_ = model(x)
  return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


if __name__ == '__main__':
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    train_data_fp = tf.keras.utils.get_file(fname=train_data, origin="file:///D:/workspace/py/tf_start/adult.data.csv")

    # 检查数据
    dataframe = pd.read_csv(train_data)
    print(dataframe.head())

    # CSV文件中列的顺序
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
    feature_names = column_names[:-1]
    label_name = column_names[-1]

    class_name = []

    # 创建一个 tf.data.Dataset
    batch_size = 32

    train_dataset = tf.data.experimental.make_csv_dataset(
        train_data_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1)

    features, labels = next(iter(train_dataset))

    print(features)

    # https://www.tensorflow.org/tutorials/structured_data/feature_columns?hl=zh-cn
    # train_dataset = train_dataset.map(pack_features_vector)
    #
    # features, labels = next(iter(train_dataset))
    #
    # print(features[:5])
