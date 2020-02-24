# Your Next Page Turner -  Book Recommender

### Project Objectives
- Introduce similar books to reader using Content-Based Recommendation System
- Add filters for better recommendations

### Dataset
- Top 200 from every year in the past 20 years with a total of 4100+ books.
- Essential information on every book including: title, authors, user rating, length, description and so on..

### Exploratory Data Analysis

![](Images/num_rating_per_book.png)

Using this graph, I decided to group the dataset into 3 groups of popularity: 
- "Super Popular": number of ratings < 27000
- "Well Known": 27000 < number of ratings < 800000
- "Deep Cut": number of ratings > 80000



![](Images/num_page_per_book.png)

Similar for book length, I divided the data set into 2 groups:
- Short: pages < 350
- Long: pages > 350


![](Images/genres.png)

Also, because there were a lot of sub-genres, I put them into 10 major group of genre. 
For example, "History", "Politics", and "Cooking" are considered "Non-Fiction". Or "Mystery", "Crime" and "Horror" are classified as "Thriller".

### Baseline Model - Simple Recommender
Features: Genre, Length, and Popularity
This is a simple model using the above features to generate book suggestions.

Example: Let's say you want a "Sci-Fi", "Long" and "Super Popular" book, the recommendations for you will be:
- Harry Porter and The Prisoner of Azkaban (Harry Porter, #3)
- Ender's Shadow (The Shadow Series, #1)
- Gardens of the Moond (Malazan Book of Fallen, #1)


### Description-Based Recommender
Feature: Description
This model uses only the description of the book to give recommendations.

Process: 
- Create Bag of Words for each book using the description of the book
- TF-IDF Vectorize the Bag of Words (BoWs)
- Compute Cosine Similarity for each BoWs 
- Use Cosine Similarity to generate recommendations

Example: I want to read similar books like "Gone Girl". 
*A little about "Gone Girl": it's about a seemingly happy married couple. Then things begin to get complicated when the wife mysteriously disappears.

The recommendations are:
- Into the Water
- The Silent Wife
- The Hunger Games (The Hunger Games, #1)
- The Lying Game (The Lying Game, #1)

Looking at these recommendations, I think this model does an ok job, because "Into the Water", and "The Silent Wife" are kind of similar with "Gone Girl". However, "The Hunger Games" is pretty different. 











