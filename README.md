# NBA Players Position Classifier

![banner](/docs/assets/positionless-nba.png)

[SOURCE](https://www.cbssports.com/nba/news/power-guard-point-center-the-nbas-positional-misfits-are-dismantling-an-antiquated-system/)

# Links
- [Website](https://jacquelinekclee.github.io/nba-players-position-classifier.github.io/)
- [Repository (with all relevant files/data)](https://github.com/jacquelinekclee/nba-players-position-classifier.github.io)

# Table of contents

- [Background](#background)
- [The Statistics](#the-statistics)
- [Statistics Source](#statistics-source)
- [The Methodology](#the-methodology)
  - [Training](#training)
  - [Final Model](#final-model)
- [Usage](#usage)
- [Findings](#findings)
- [Source File](#source-file)
- [Legality](#legality)

## Background
As someone who has been playing basketball since the second grade and has been a Warriors fan since birth, the NBA and basketball in general have always held a special place in my heart. As time goes on, statistics and analytics have played an increasingly larger role in the world of basketball. With this project, and my [NBA All Stars Classifier](https://jacquelinekclee.github.io/nba-all-stars-classifier.github.io/), I wanted to use my love of basketball in developing and praciticing new Data Science skills.

As basketball players have gotten more skilled and talented and as the game itself has revolutionized, the notion of **positions** increasingly becomes a dated aspect of the game. The commissioner of the NBA, Adam Silver, acknowledged that the NBA ["has moved increasingly to positionless basketball"](https://www.nba.com/news/nba-commissioner-adam-silver-discusses-leagues-positionless-basketball-at-annual-finals-press-conference) when discussing the possibility of removing positions from the All-NBA decision process, which honors the best players in the league. 

With this project, I hope to understand basketball positions using statistics and machine learning. If an ML model can predict a player's position based on his stats, then maybe this player doesn't play positionless basketball and adheres to his traditional role as a guard/forward/center. If a model gets it wrong, then maybe the player has a unique play style that doesn't conform to the historic statistics of players in his positions before him. 

Keep reading to learn more about the data used, the approach I took to building this classifier, and the cool findings the model yielded!

[(Back to top)](#table-of-contents)

## The Statistics
The NBA tracks almost 50 different statistics for every player in the league. Many statistics are often unknown to most basketball fans, so using only the common statistics will make the most sense for everyone. 
Here are some basic definitions of the statistics I will be using in my classifier:
- True shooting percentage (TS%): a metric that demonstrates how efficiently a player shoots the ball. Takes into consideration field goals, 3-pointers, and free throws (unlike other metrics like field goal percentage).
- Rebounds per game (RPG): a metric that shows how many total rebounds (both offensive and defensive) a player averages per game. 
- Assists per game (APG): a metric that shows how many total assists a player averages per game.
- Points per game (PPG): a metric that shows how many total points a player averages per game.
- Blocks per game (BPG): a metric that shows how many total blocks a player averages per game.
- Steals per game (SPG): a metric that shows how many total steals a player averages per game.

[(Back to top)](#table-of-contents)

## Statistics Source
The training data comes from [Kaggle](https://www.kaggle.com/drgilermo/nba-players-stats?select=Seasons_Stats.csv). The test data come from [Basketball Reference](https://www.basketball-reference.com/):
- [2018-19 Data](https://www.basketball-reference.com/leagues/NBA_2019_per_game.html)
- [2020-21 Data](https://www.basketball-reference.com/leagues/NBA_2021_per_game.html)
- [2021-22 Data](https://www.basketball-reference.com/leagues/NBA_2022_per_game.html)

[(Back to top)](#table-of-contents)

## Usage

Please refer to the [Jupyter Notebook Viewer](https://nbviewer.org/github/jacquelinekclee/nba-players-position-classifier/blob/0d3b4b07abce345b5b50566b95ac793e7ba10c5d/nba_positions_classifier.ipynb) or the [.ipynb file](https://github.com/jacquelinekclee/nba-players-position-classifier/blob/0d3b4b07abce345b5b50566b95ac793e7ba10c5d/nba_positions_classifier.ipynb) to view all the code for the classifier. The notebook with all the data cleaning can be seen [here](https://github.com/jacquelinekclee/nba-players-position-classifier/blob/0d3b4b07abce345b5b50566b95ac793e7ba10c5d/nba_players_data_cleaning.ipynb). 

The [source file](#source-file) contains all the functions used to clean/manipulate the data and DataFrames.

[(Back to top)](#table-of-contents)

## The Methodology

In the initial data cleaning process, the players' positions were simplified to include 5 classes: 

| Position   | Proportion in Training Data        |
|:---|-----------:|
| Forward  | 0.399   |
| Guard  | 0.396   |
| Center  | 0.199   |
| Guard/Forward | 0.003 |
| Forward/Center | 0.003 |

Notice that forwards and guards make up nearly 80% of the datasest, with centers making just under 20% and the hybrid positions making up hardly 1% altogether. Since this is a multiclass problem with some class imbalance, I wanted to test different models. Another consideration was the effect of including Year as a feature. Theoretically, if the way a player of a given position hasn't changed over time, then a classifier without year should perform better (or differently) than a classifier with year as a feature. 

I tried both random forest modesl and XGBoost models. I went with random forests as a better alternative to decision trees in that random forests are more robust to overfitting. I also chose to explore XGBoost classifiers as they might work better for the class imbalance. In total, 4 different models were originally trained: 2 random forests (one with year as a featuere and one without) and 2 XGBoost classifiers (one with year as a feature and one without). The features for both models were as follows:
- TS%: true shooting percentage
- RPG: rebounds per game
- APG: assists per game
- PPG: points per game
- BPG: blocks per game
- SPG: steals per game
- Year: year of that season (e.g, rows from the 1980-1981 season has 1981 as its year)
- All Star: whether that player was an All Star that season
- MVP: whether that player was the MVP that season

[(Back to top)](#table-of-contents)

## Training
Using Scikit's `GridSearchCV`, I tested out several combinations of different hyperparameters. The training times for both models were extremely *long*. See below for the paramters tested for each type of model:

Parameters tested for random forests:
`{'n_estimators': [300,500,700], 'max_features': ['sqrt', 'log2'], 'max_depth' : [5,10,15,20,25,None], 'criterion' :['gini', 'entropy'], 'random_state' : [18]}`

Parameters tested for XGBoost classifiers:
`{'max_depth': [3,6,10], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 500, 1000], 'colsample_bytree': [0.3, 0.7]}`

Given that this a multiclass classification problem, I found that accuracy was the most straightforward evaluation metric. Additionally, I looked at the proportion of players for a given position that were misclassified. For example, if 10 out of 40 centers in the test data set were *not* classified as centers by the model, then the proportion would be 0.25. See below for the results found with the 2018-19 season as the test set:

| model                      |   test accuracy |   prop wrong for centers |   prop wrong for forwards |   prop wrong for guards |
|:---------------------------|----------------:|-------------------------:|--------------------------:|------------------------:|
| random forest without year |        0.390 |                     0.48 |                      0.19 |                    0.26 |
| random forest with year    |        0.392 |                     0.47 |                      0.19 |                    0.23 |
| XGBoost without year       |        0.378 |                     0.53 |                      0.2  |                    0.27 |
| XGBoost with year          |        0.390 |                     0.55 |                      0.16 |                    0.25 |

Both types of model performed better when year was included in the feature set, with year boosting accuracy more for the XGBoost models than the random forests. What stuck out to me most was how much better the random forests were at classifying the minority class, centers. 

Additionally, the 2 types of classifiers found different features to be more important. The Random Forest classifiers thought RBG (rebounds per game) were more important than BPG (blocks per game), while the XGBoost classifiers didn't. Both classifiers had similar levels of feature importance for APG (assists per game) and PPG (points per game). See the notebook for feature importances. 

Some pitfalls of both classifiers include: 
- Neither classifier was able to classify the hybrid positions, GF and FC, correctly. This is likely because only about 0.006 of the training data have these hybrid positions. 
- Both the All Star and MVP features had 0 importance for all 4 models tested. Including irrelevant features could make cost (e.g., runtime) unnecessarily high. 
- Although XGBoost was used to try and combat the class imbalance (around 2x guards and forwards than centers), XGBoost did *worse* and classifying centers than Random Forest did. 

In effort to create a better performing classifier, a new position column will be created. Hopefully making this problem only 3 classes instead of 5 will yield a better classifier. Also, MVP and All Star will be removed from the feature list. It seems that the year feature is particularly useful for classifying guards. Lastly, although XGBoost yielded a slightly higher accuracy, it classified centers much worse than the random forest (which was unexpected). Since the XGBoost didn't provide the expected benefits and its training time is much slower, Random Forest will be used going forward. 

## Final Model

Based on the findings above, I went with a Random Forest model with TS%, RPG, APG, PPG, BPG, SPG, and Year as features. See below for the feature importances and test accuracies:

|   Feature   |          Importance |
|:-----|----------:|
| RPG  | 0.254  |
| APG  | 0.251  |
| BPG  | 0.233  |
| SPG  | 0.116  |
| PPG  | 0.077 |
| TS%  | 0.037 |
| Year | 0.031 |

| season   |   test accuracy |   prop wrong for centers |   prop wrong for forwards |   prop wrong for guards |
|:---------|----------------:|-------------------------:|--------------------------:|------------------------:|
| 2018-19  |           0.736 |                     0.47 |                      0.19 |                    0.24 |
| 2020-21  |           0.715 |                     0.46 |                      0.22 |                    0.27 |
| 2021-22  |           0.704 |                     0.54 |                      0.19 |                    0.27 |

Clearly, this model performed much better than the 1st round. However, it started performing worse for the more recent seasons. This may be some indication of a shift coming in basketball, where players' statistics and general playstyles don't reflect the typical notions of positions in seasons prior. 

[(Back to top)](#table-of-contents)

## Findings
Draymond Green, Ben Simmons, Giannis Antetokounmpo, LeBron James, Kevin Durant, Nikola Jokić, and Jayson Tatum are some consensus "positionless" NBA players (see this [CBS article]('https://bleacherreport.com/articles/2627364-5-unique-nba-players-who-dont-fit-in-a-category') and this [Blearcher Report article]('https://bleacherreport.com/articles/2627364-5-unique-nba-players-who-dont-fit-in-a-category')). One might expect the classifier to predict these players' posititions *incorrectly* if they are truly "positionless." As with all things basketball, several things transcend the stat sheet, but hopefully these results provide some interesting insights! The table below shows the correct position (Pos) and the model's prediction (pos_pred) for these "positionless" players:

| Player                | Pos_2019   | pos_pred_2019   | Pos_2021   | pos_pred_2021   | Pos   | pos_pred   |
|:----------------------|:-----------|:----------------|:-----------|:----------------|:------|:-----------|
| Giannis Antetokounmpo | F          | F               | F          | F               | F     | F          |
| Kevin Durant          | F          | F               | F          | F               | F     | F          |
| Draymond Green        | F          | F               | F          | G               | F     | F          |
| LeBron James          | F          | F               | G          | F               | F     | F          |
| Nikola Jokić          | C          | F               | C          | F               | C     | F          |
| Jayson Tatum          | F          | F               | F          | F               | F     | F          |

Giannis Antetokounmpo, Kevin Durant, and Jayson Tatum were always correctly classified as forwards for the 3 seasons used as test data. This may be because forwards were the most common position in the training data, so the classifier knows forwards particularly well. 

Basketball Reference has LeBron James listed as having played both the forward and guard positions. In the 2020-21 season, where James was listed primarily as a guard, the classifier predicted him incorrectly to be a forward. In the eyes of the classifier, it seems that James presents as a forward more than a guard.

In the 2020-21 season, Draymond Green was listed as a forward, but misclassified as a guard. Green had to step up that season considering Klay Thompson's absence that season and the fact that other guards like Jordan Poole (playing only his 3rd year professionaly after some time in the G-League) and Gary Payton II (who hardly played at all) were early in their development. 

Hopefully some of these insights were interesting! Please feel free to explore the Python notebooks on your own!

[(Back to top)](#table-of-contents)

## Source File
- [nba_players_classification.py](https://github.com/jacquelinekclee/nba-players-position-classifier/blob/0d3b4b07abce345b5b50566b95ac793e7ba10c5d/nba_players_classification.py)
  - Has all the functions used for mainly the data cleaning.

## Legality
This personal project was made for the sole intent of applying my skills in Python thus far and as a way to learn new ones. It is intended for non-commercial uses only.

[(Back to top)](#table-of-contents)
