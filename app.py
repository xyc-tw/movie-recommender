from flask import Flask, render_template, request
from recommender import recommend_random, recommend_with_nmf, recommend_with_cos_similarity
import pandas as pd
import numpy as np
# from utils import movies

movies_df = pd.read_csv("./processed-data/movies_df_processed_2.csv", sep=",")
print(len(sorted(movies_df.title.to_list())))

app = Flask(__name__)
@app.route('/')
def homepage():
    return render_template('index.html' ,movies = movies_df.title.to_list())
    
@app.route('/recommendation')
def recommendation():
    titles = request.args.getlist("title")
    ratings = request.args.getlist("rating")

    # make a input_rating dict with movieId and rating
    dic = dict(zip(titles, ratings))
    input_rating = dict()
    for key, value in dic.items():
        movieId = int(movies_df[movies_df['title'] == key]['movieId'].values[0])
        input_rating[movieId] = int(value)
    

    if request.args['method'] == 'Random':
        recs = recommend_random(k=3).to_list()
        return render_template('recommender.html',values = recs)
    elif request.args['method'] == 'Cosine':
        recs = recommend_with_cos_similarity(input_rating)
        return render_template('recommender.html',values = recs)
    else:
        recs = recommend_with_nmf(input_rating)
        return render_template('recommender.html',values = recs)
    

if __name__=='__main__':
    app.run(port=5000,debug=True)