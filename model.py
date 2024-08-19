import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from io import BytesIO
import pymongo
from flask import jsonify
from flask import Flask,request
from flask_cors import CORS

class CombinedRecommender:
    def __init__(self, new_df,user,df, users_pivot):
        self.df = df
        self.users_pivot = users_pivot
        self.new_df =new_df
        self.user = user
###user base
    def combined_recommendations(self, user_id):
    # Check if the user exists in the dataset
        if user_id not in self.df["User-ID"].values:
            print("❌ User NOT FOUND ❌")
            return None

        # Find the index of the user in the pivot table
        index = np.where(self.users_pivot.index == user_id)[0][0]

        # Calculate cosine similarity
        similarity = cosine_similarity(self.users_pivot)
        similar_users = list(enumerate(similarity[index]))
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:6]

        # Get recommended users
        user_rec = []
        for i in similar_users:
            data = self.df[self.df["User-ID"] == self.users_pivot.index[i[0]]]
            user_rec.extend(list(data.drop_duplicates("User-ID")["User-ID"].values))

        # Get book recommendations for the similar users
        x=self.df[self.df["User-ID"]==user_id]
        recommend_books=[]
    
        for i in user_rec:
            y=self.df[(self.df["User-ID"]==i)]
            books=y.loc[~y["ISBN"].isin(x["ISBN"]),:]
            books=books.sort_values(["Book-Rating"],ascending=False)[0:5]
            recommend_books.extend(books["ISBN"].values)

        return recommend_books[0:30]
###item base
    def item_based(self,bookTitle):
        bookTitle = str(bookTitle)
        if bookTitle in self.df["Book-Title"].values:
            rating_count = pd.DataFrame(self.df["Book-Title"].value_counts())
            rare_books = rating_count[rating_count["count"] <= 20].index
            common_books = self.df[~self.df["Book-Title"].isin(rare_books)]
            if bookTitle in rare_books:
                most_common = pd.Series(common_books["Book-Title"].unique()).sample(3).values
            else:
                common_books_pivot = common_books.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating")
                title = common_books_pivot[bookTitle]
                recommendation_df = pd.DataFrame(common_books_pivot.corrwith(title).sort_values(ascending=False)).reset_index(drop=False)
                if bookTitle in recommendation_df["Book-Title"].values:
                    recommendation_df = recommendation_df[recommendation_df["Book-Title"] != bookTitle]
                    # print("booktitle",recommendation_df)
                less_rating = []
                for i in recommendation_df["Book-Title"]:
                    if self.df[self.df["Book-Title"] == i]["Book-Rating"].mean() < 5:
                        less_rating.append(i)
                # print("kasdasd",len(less_rating))    
                if recommendation_df.shape[0] - len(less_rating) > 5:
                    recommendation_df = recommendation_df[~recommendation_df["Book-Title"].isin(less_rating)]
                    # print("LANH",recommendation_df)
                recommendation_df = recommendation_df.head(30)
                recommendation_df.columns = ["Book-Title", "Correlation"]
                # Lấy số ISBN từ tên sách
                isbn_list = []
                for title in recommendation_df["Book-Title"]:
                    isbn = self.df[self.df["Book-Title"] == title]["ISBN"].values[0]
                    # print("title",title)    
                    isbn_list.append(isbn)
                return isbn_list[0:30]
    ### popular_books
    def popular_books(self,n):
        rating_count=self.df.groupby("ISBN").count()["Book-Rating"].reset_index()
        rating_count.rename(columns={"Book-Rating":"NumberOfVotes"},inplace=True)
        
        rating_average=self.df.groupby("ISBN")["Book-Rating"].mean().reset_index()
        rating_average.rename(columns={"Book-Rating":"AverageRatings"},inplace=True)

        popularBooks=rating_count.merge(rating_average,on="ISBN")

        def weighted_rate(x):
            v=x["NumberOfVotes"]
            R=x["AverageRatings"]

            return ((v*R) + (m*C)) / (v+m)

        C=popularBooks["AverageRatings"].mean()
        m=popularBooks["NumberOfVotes"].quantile(0.90)

        popularBooks=popularBooks[popularBooks["NumberOfVotes"] >=100]
        popularBooks["Popularity"]=popularBooks.apply(weighted_rate,axis=1)
        popularBooks=popularBooks.sort_values(by="Popularity",ascending=False)
        # return popularBooks[["ISBN","NumberOfVotes","AverageRatings","Popularity"]].reset_index(drop=True).head(n)
        top_n_isbn_list = popularBooks["ISBN"].head(n).tolist()
        return top_n_isbn_list
    ###content base
    def content_based(self,bookTitle):
        bookTitle = str(bookTitle)

        if bookTitle in self.df["Book-Title"].values:
            rating_count = pd.DataFrame(self.df["Book-Title"].value_counts())
            rare_books = rating_count[rating_count["count"] <= 100].index
            common_books = self.df[~self.df["Book-Title"].isin(rare_books)]

            if bookTitle in rare_books:
                most_common = pd.Series(common_books["ISBN"].unique()).sample(20).values
                most_common = most_common.tolist()
                return most_common[0:20]
            else:
                common_books = common_books.drop_duplicates(subset=["Book-Title"])
                common_books.reset_index(inplace=True)
                common_books["index"] = [i for i in range(common_books.shape[0])]
                targets = ["Book-Title", "Book-Author", "Publisher"]
                common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values) for i in range(common_books[targets].shape[0])]
                vectorizer = CountVectorizer()
                common_booksVector = vectorizer.fit_transform(common_books["all_features"])
                similarity = cosine_similarity(common_booksVector)
                index = common_books[common_books["Book-Title"] == bookTitle]["index"].values[0]
                similar_books = list(enumerate(similarity[index]))
                similar_booksSorted = sorted(similar_books, key=lambda x: x[1], reverse=True)[0:20]
                books = []

                # Lấy số ISBN từ các sách tương tự
                for idx, _ in similar_booksSorted:
                    isbn = self.df[self.df["Book-Title"] == common_books.loc[idx, "Book-Title"]]["ISBN"].values[0]
                    books.append(isbn)

                return books

with open('model_public.pkl', 'rb') as f:
    model = pickle.load(f)
app = Flask(__name__)

CORS(app)

@app.route("/recommend_by_user")
def member():
    memberId = request.args.get('id')
    id = int(memberId)
    rcm = model.combined_recommendations(id)
    return rcm

@app.route("/recommend_by_item" , methods =['POST'])
def item():
    data = request.json
    if 'item' in data:
        item = data['item']
        rcm = model.item_based(str(item))
        return jsonify(rcm)
    else:
        return print('Khong có item')

@app.route("/recommend_by_content",  methods =['POST'])
def content():
    data = request.json
    if 'content' in data:
            content = data['content']
            rcm =model.content_based(content)
            return jsonify(rcm)
    else:
        return print('Khong có content')

@app.route("/recommend_by_popular")
def popular_id():
    popular_id = request.args.get('popular')
    popular_id = int(popular_id)
    rcm = model.popular_books(popular_id)
    return rcm

@app.route("/load_data" )

def load_data():
    client = pymongo.MongoClient("mongodb+srv://quelanh7412369:lanhpro101@cluster0.igg3ixr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["Data_Api"]

    rating = db["rating_vaild"]
    data_rating = []
    for i in rating.find():
        try:
            newdata = [i['User-ID'],i['ISBN'],i['Book-Rating']]
            data_rating.append(newdata)
        except Exception as e:
            print(e)

    book = db["book"]
    data_book = []
    for i in book.find():
        try:
            newdata = [i['ISBN'],i['Book-Title'],i['Book-Author'],i['Year-Of-Publication'],i['Publisher'],i['Image-URL-L']]
            data_book.append(newdata)
        except Exception as e:
            print(e)

    df_book = pd.DataFrame(data_book,columns=['ISBN','Book-Title','Book-Author','Year-Of-Publication','Publisher','Image-URL-L'])

    df_ratings = pd.DataFrame(data_rating,columns=['User-ID','ISBN','Book-Rating'])



    books_data=df_book.merge(df_ratings,on="ISBN")
    df=books_data.copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    df.drop(columns=["Year-Of-Publication"],axis=1,inplace=True)
    df.drop(index=df[df["Book-Rating"]==0].index,inplace=True)
    df["Book-Title"]=df["Book-Title"].apply(lambda x: re.sub("[\W_]+"," ",x).strip())
    users_pivot=df.pivot_table(index=["User-ID"],columns=["ISBN"],values="Book-Rating")
    users_pivot.fillna(0,inplace=True)
    new_df=df[df['User-ID'].map(df['User-ID'].value_counts()) > 40]
    user = new_df["User-ID"].tolist()

    recommender = CombinedRecommender(new_df,user,df, users_pivot) 
    with open('model_public.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    return user

app.run(debug=True)


# Đọc dữ liệu từ file pkl

