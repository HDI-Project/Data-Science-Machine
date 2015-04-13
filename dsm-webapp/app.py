from flask import Flask
import flask
from flask import request
from flask import render_template
import config
import datetime
import json
import static_data
import random
import web_utils

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from features import database, table, column, utils, predict

import sys
sys.modules['database'] = database
sys.modules['table'] = table
sys.modules['column'] = column

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('connect.html')

@app.route('/connect')
def connect():
    return render_template('connect.html')

@app.route('/entity')
def entity_route():
    entities = []
    for t in db.tables:
        entities.append({
            "name" : t,
            "count" : db.tables[t].num_rows
            })

    template_data = {
        'entities' : entities
    }
    return render_template('entity.html', **template_data)


@app.route('/analyze')
def analyze():
    entity_name = request.args.get('entity', None)
    entity = db.tables[entity_name]

    predictable_features = [{"name":c.metadata["real_name"], "id":c.name}  for c in predict.get_predictable_features(entity)]
    features = [{"name":c.metadata["real_name"], "id":c.name} for c in entity.get_column_info()]

    # random.shuffle(features)

    template_data = {
        "entity_name" : entity_name,
        "predictable_features": predictable_features,
        'features' : features,
        'features_str' : json.dumps(features),
        'num_features': len(features),
        'num_rec_features' : 2+random.randint(0,4)
    }
    return render_template('analyze.html', **template_data)

@app.route('/model')
def model():
    entity_name = request.args.get('entity', None)
    entity = db.tables[entity_name]

    using_names = request.args.getlist('using[]')
    using = [entity.get_col_by_name(c) for c in using_names]

    target_feature_name = request.args.get('target', None)
    target_feature = entity.get_col_by_name(target_feature_name)

    if using == []:
        score, using = predict.best_model(target_feature)
    else: 
        score, using = predict.model(target_feature, using)

    print using
    using = [{"name":u.metadata["real_name"], "id":u.name, "weight": weight} for (u,weight) in using]

    return flask.jsonify(score=score, using=using)

@app.route('/scatter')
def scatter():
    entity_name = request.args.get('entity', None)
    entity = db.tables[entity_name]

    predictable_features = [{"name":c.metadata["real_name"], "id":c.name}  for c in predict.get_predictable_features(entity)]
    print dir(utils)
    features = web_utils.get_scatter_features(entity)

    template_data = {
        "entity_name" : entity_name,
        'features' : features,
        'features_str' : json.dumps(features),
        'num_features': len(features),
    }
    return render_template('scatter.html', **template_data)



@app.route('/scatter-data')
def scatter_data():
    entity_name = request.args.get('entity', None)
    entity = db.tables[entity_name]

    x = request.args.get('x', None)
    feature_x = entity.get_col_by_name(x)

    y = request.args.get('y', None)
    feature_y = entity.get_col_by_name(y)

    data = entity.get_rows([feature_x,feature_y])
    data = web_utils.iterSample(data, 999)

    graph_data = []
    for (x,y) in data:
        if x == None:
            x = 0

        if y == None:
            y = 0

        graph_data.append([x,y])

    payload = {
        "data": graph_data,
        "title": feature_x.metadata["real_name"] + " vs " + feature_y.metadata["real_name"],
        "x_axis": feature_x.metadata["real_name"],
        "y_axis": feature_y.metadata["real_name"]
    }

    return flask.jsonify(**payload)


@app.route('/cluster')
def cluster():
    entity_name = request.args.get('entity', None)
    entity = db.tables[entity_name]

    template_data = {
        "entity_name" : entity_name,
    }
    return render_template('cluster.html', **template_data)


@app.route('/cluster-data')
def cluster_data():
    entity_name = request.args.get('entity', None)
    entity = db.tables[entity_name]

    k = int(request.args.get('k', 5))


    clusters = predict.cluster(entity, k=k)

    payload = {
        "clusters": clusters,
        "k" : k
    }

    return flask.jsonify(**payload)




if __name__ == "__main__":
    db = database.Database.load("../features/models/Outcomes")
    app.run(debug=True)
