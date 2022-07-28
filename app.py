from typing import Any
from flask import Flask, jsonify, request
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from flask_mysqldb import MySQL
import MySQLdb.cursors


app = Flask(__name__)


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_kemiskinan'


mysql = MySQL(app)


def preprocess(data: pd.DataFrame):
    data.fillna(0, inplace=False)
    scalar = MinMaxScaler()
    scale = scalar.fit_transform(
        data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14']])
    data_scale = pd.DataFrame(scale, columns=[
                              'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14'])
    return data_scale


@app.route('/silhoutte')
def silhoutte():
    response = {}
    # df = pd.read_csv('data.csv')
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM clustering')
    data = cursor.fetchall()

    # df = pd.read_csv('data.csv')
    df = pd.DataFrame(data)
    # df = df.drop('id_penduduk', 1)
    df = df.drop('id', 1)
    df = df.drop('cluster', 1)
    # df = df.append(data, ignore_index=True)
    data = preprocess(df)
    km = KMeans(n_clusters=3)
    y_predict = km.fit_predict(
        data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14']])
    y_predict
    labels = km.labels_
    for i in range(2, 5):
        labels = cluster.KMeans(
            n_clusters=i, random_state=200).fit(data).labels_

        response[i] = metrics.silhouette_score(
            data, labels, metric='euclidean', sample_size=1000, random_state=200)

    return jsonify(response)


@app.route('/postInputanSurvey')
def postDataClustering():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # # executing query
    cursor.execute('SELECT * FROM clustering')
    # fetching all records from database
    data = cursor.fetchall()

    # df = pd.read_csv('data.csv')
    df = pd.DataFrame(data)
    # df = df.drop('id_penduduk', 1)
    df = df.drop('id', 1)
    df = df.drop('cluster', 1)
    # df = df.append(data, ignore_index=True)
    data = preprocess(df)

    k_means = KMeans(n_clusters=3, random_state=0)
    y_predict = k_means.fit_predict
    k_means.fit(data)
    y_predict = k_means.fit_predict(
        data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14']])
    labels = k_means.labels_
    data["cluster"] = labels
    # data["id_penduduk"] = labels
    # data.to_csv('hasilclustering.csv')
    # print(data)
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    data = data.iloc[-1]
    data_penduduk = df.iloc[-1]
    # id_penduduk = df('id_penduduk')
    # return data
    print(data)
    cur.execute("INSERT INTO hasil_clustering (X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14,cluster,id_penduduk) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (data['X1'], data['X2'], data['X3'], data['X4'], data['X5'], data['X6'], data['X7'], data['X8'], data['X9'], data['X10'], data['X11'], data['X12'], data['X13'], data['X14'], data['cluster'], data_penduduk['id_penduduk']))
    mysql.connection.commit()

    return jsonify(data.to_json())


# @app.route('/hasilclustering')
# def hasilclustering():
#     cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#     data =

if __name__ == '__main__':
    app.run(debug=True)
