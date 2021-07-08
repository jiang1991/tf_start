import os
import tensorflow as tf
from tensorflow import feature_column
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# follow the https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough?hl=zh-cn
# https://www.tensorflow.org/tutorials/structured_data/feature_columns?hl=zh-cn#%E5%88%86%E7%B1%BB%E5%88%97

train_data = "adult.data.csv"
test_data = "adult.test.csv"


# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


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

    # train_data_fp = tf.keras.utils.get_file(fname=train_data, origin="file:///D:/workspace/py/tf_start/adult.data.csv")
    # train_data_fp = tf.keras.utils.get_file(fname=train_data, origin="file:///D:/python/tf_start/adult.data.csv")
    # test_data_fp = tf.keras.utils.get_file(fname=train_data, origin="file:///D:/python/tf_start/adult.test.csv")

    # 检查数据
    dataframe = pd.read_csv(train_data)
    print(dataframe.head())
    train, val = train_test_split(dataframe, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')

    test = pd.read_csv(test_data)
    print(len(test), 'test examples')

    batch_size = 5
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    for feature_batch, label_batch in train_ds.take(1):
        print('Every feature:', list(feature_batch.keys()))
        print('A batch of ages:', feature_batch['age'])
        print('A batch of targets:', label_batch)

    # 我们将使用该批数据演示几种特征列
    example_batch = next(iter(train_ds))[0]


    # 用于创建一个特征列
    # 并转换一批次数据的一个实用程序方法
    def demo(feature_column):
        feature_layer = layers.DenseFeatures(feature_column)
        print(feature_layer(example_batch).numpy())


    # # CSV文件中列的顺序
    # column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']
    # feature_names = column_names[:-1]
    # label_name = column_names[-1]

    # # class_name = []
    #
    # # 创建一个 tf.data.Dataset
    # batch_size = 32
    #
    # train_dataset = tf.data.experimental.make_csv_dataset(
    #     train_data_fp,
    #     batch_size,
    #     column_names=column_names,
    #     label_name=label_name,
    #     num_epochs=1)
    #
    # features, labels = next(iter(train_dataset))
    #
    # print(features)

    # https://www.tensorflow.org/tutorials/structured_data/feature_columns?hl=zh-cn
    # train_dataset = train_dataset.map(pack_features_vector)
    #
    # features, labels = next(iter(train_dataset))
    #
    # print(features[:5])

    feature_columns = []

    for header in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
        feature_columns.append(feature_column.numeric_column(header))

    workclass = feature_column.categorical_column_with_vocabulary_list(
        'workclass',
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
         'Never-worked']
    )
    workclass_one_hot = feature_column.indicator_column(workclass)
    feature_columns.append(workclass_one_hot)

    race = feature_column.categorical_column_with_vocabulary_list(
        'race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    )
    race_one_hot = feature_column.indicator_column(race)
    feature_columns.append(race_one_hot)

    sex = feature_column.categorical_column_with_vocabulary_list(
        'sex', ['Female', 'Male']
    )
    sex_one_hot = feature_column.indicator_column(sex)
    feature_columns.append(sex_one_hot)

    # target = feature_column.categorical_column_with_vocabulary_list(
    #     'target', ['>50K', '<=50K']
    # )
    # target_one_hot = feature_column.indicator_column(target)
    # feature_columns.append(target_one_hot)
    demo(feature_column.numeric_column("age"))
    # demo(workclass_one_hot)
    demo(race_one_hot)
    demo(sex_one_hot)

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(10, activation=tf.nn.relu),  # 需要给出输入的形式
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=5)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)

    # 使用 Keras 创建模型
