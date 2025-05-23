import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the dataset
data_path = r"C:\Users\10759\Desktop\ml-100k\u.data"
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv(data_path, sep='\t', names=ratings_cols, encoding='latin-1')

# 2. Create utility matrix
utility_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')

# Center the ratings
user_means = utility_matrix.mean(axis=1)
centered_utility = utility_matrix.sub(user_means, axis=0)

# 3. Calculate cosine similarity
centered_utility_filled = centered_utility.fillna(0)
user_similarity = cosine_similarity(centered_utility_filled)
user_similarity_df = pd.DataFrame(user_similarity,
                                index=utility_matrix.index,
                                columns=utility_matrix.index)

# 4. Find similar users
target_user = 1
similar_users = user_similarity_df[target_user].sort_values(ascending=False)[1:11]

# 5. Visualization: Similar Users
plt.figure(figsize=(10, 6))
similar_users.plot(kind='bar')
plt.title(f'Top 10 Users Similar to User {target_user}')
plt.ylabel('Cosine Similarity Score')
plt.xlabel('User ID')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Calculate predicted rating
item_id = 508
similar_users_ratings = utility_matrix.loc[similar_users.index, item_id]

# 7. Visualization: Similar Users' Ratings
plt.figure(figsize=(10, 6))
similar_users_ratings.plot(kind='bar')
plt.axhline(y=similar_users_ratings.mean(), color='r', linestyle='--',
            label=f'Average: {similar_users_ratings.mean():.2f}')
plt.title(f'Ratings for Item {item_id} from Similar Users')
plt.ylabel('Rating')
plt.xlabel('User ID')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Final results
predicted_rating = similar_users_ratings.mean()
print(f"\nTop 10 most similar users to user {target_user}: {similar_users.index.tolist()}")
print(f"Average rating for item {item_id} from similar users: {predicted_rating:.2f}")
print(f"Predicted rating for user {target_user} on item {item_id}: {predicted_rating:.2f}")
