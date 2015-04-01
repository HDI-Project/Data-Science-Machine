from flask import Flask
import flask
from flask import request
from flask import render_template
import config
import datetime
import json
import static_data
import random


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
    features = utils.get_col_names(entity)

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

    using_names = request.args.get('using', [])
    using = [entity.get_col_by_name(c) for c in using_names]

    target_feature_name = request.args.get('target', None)
    target_feature = entity.get_col_by_name(target_feature_name)

    if using == []:
        score, using = predict.best_model(target_feature)
    else: 
        score, using = predict.model(target_feature, using)

    using = [(u.metadata["real_name"], u.name) for u in using]

    return flask.jsonify(score=score, using=using)


if __name__ == "__main__":
    db = database.Database.load("../features/Products")
    app.run(debug=True)
