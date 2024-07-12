import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words='english')
ps = PorterStemmer()

# Read both the csv file and assign it to a variable creating a DATA FRAME
credits_df = pd.read_csv('credits.csv')
movies_df = pd.read_csv('movies.csv')


movies_df = movies_df.merge(credits_df, on='title')  # Merge the credits_df DATA FRAME to movies_df DATA FRAME using the 'title' column in both the both the DATA FRAMES
movies_df = movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]  # Taking only 7 columns value to the DATA FRAME 

# Extracting name from object:[{'id':  , 'name':  },{'id':  , 'name':  },...,{'id':  , 'name':  }]
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Extracting main cast name from object:[{"cast_id": 242, "character": "Jake Sully", "credit_id": "5602a8a7c3a3685532001c9a", "gender": 2, "id": 65731, "name": "Sam Worthington", "order": 0},....]
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
        return L

# Extracting Director name from object
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Function to preform the cell value split 
def split_overview(x):
    if isinstance(x, str):  # Check if x is a string
        return x.split()
    else:
        return x

# Remove space   
def clearSpace(x):
    if x is None:
        return x  # Return None if x is None
    else:
        return [i.replace(" ", "") for i in x]

# Remove brackets 
def join_tags(x):
    if isinstance(x, str):  # Check if x is already a string
        return x  # Return x unchanged
    elif isinstance(x, list):  # Check if x is a list
        return ' '.join(x)  # Join the elements of the list
    else:
        return '' 
    
# Apply Stemming
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Convert 'tags' value from UPPER case to LOWER case if there are any upper case letters
def lowercase_tags(x):
    if isinstance(x, str):
        return x.lower()
    else:
        return ''


movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df['cast'] = movies_df['cast'].apply(convert3)
movies_df['crew'] = movies_df['crew'].apply(fetch_director)

# Split the overview cell value 
movies_df['overview'] = movies_df['overview'].apply(split_overview)

# Remove space between the value, Example: 1st cell of 'genres' has a value 'Science Fiction' it updates to 'ScienceFiction'
movies_df[['genres', 'keywords', 'cast', 'crew']] = movies_df[['genres', 'keywords', 'cast', 'crew']].apply(lambda x: x.map(clearSpace))

movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']  # Create a column 'tags' with 5 columns value

new_df = movies_df[['movie_id','title','tags']]  # Creata a NEW DATA FRAME with 'movie_id', 'title' and 'tags'

new_df.loc[:,'tags'] = new_df['tags'].apply(join_tags)  # Convert 'tags' cell value from Array to String 
new_df.loc[:,'tags'] = new_df['tags'].apply(lowercase_tags)  # Convert 'tags' value from UPPER case to LOWER case if there are any upper case letters

vectors = cv.fit_transform(new_df['tags']).toarray()

new_df.loc[:,'tags'] = new_df['tags'].apply(stem)

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()
    
    # Check if the movie exists in the DataFrame
    if movie not in new_df['title'].str.lower().values:
        print("--------------------------------------------------------------------------\nMovie '{}' not found in the database. Please check the spelling or try another movie.".format(movie.title()))
        return []
    
    movie_index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(new_df.iloc[i[0]].title)
    return recommended_movies

def main():
    # print("List of Movies : \n",", ".join(new_df['title'].astype(str).tolist()))
    while True:
        movie = input("\nEnter the movie title (Hollywood movies only): ")
        if movie.lower() == 'exit':
            print("\nExiting program...")
            break
        recommended_movies = recommend(movie)
        if recommended_movies:
            print("\n\nRecommended movies for '{}':\n-------------------------------------".format(movie))
            for title in recommended_movies:
                print(title)
            print("-------------------------------------\n(Type 'Exit' to exit the program!!)\n-------------------------------------")
        else:
            print("\nNo recommendations available for '{}'.\n-------------------------------------\n(Type 'Exit' to exit the program!!)\n-------------------------------------".format(movie.title()))

if __name__ == "__main__":
    main()