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
- [The Results](#the-results)
- [Source File](#source-file)
- [Legality](#legality)

## Background
As someone who has been playing basketball since the second grade and has been a Warriors fan since birth, the NBA and basketball in general have always held a special place in my heart. As time goes on, statistics and analytics have played an increasingly larger role in the world of basketball. With this project, and my [NBA All Stars Classifier](https://jacquelinekclee.github.io/nba-all-stars-classifier.github.io/), I wanted to use my love of basketball in developing and praciticing new Data Science skills.

This classifier, specifically a K-Nearest Neighbors Classifier, uses statistics from nearly 19,000 players' seasons from 1980-2017. I will then test my classifier using statistics from the 2018-19 season, the 2020-21 season (skipping the season interrupted by the pandemic), and the most recent season, 2021-22. Given that all star voting involves non-expert (fan) voting and some level of subjectivity, hopefully the results will help reveal which players' All Star designation(s) match the data, which players were overlooked, and which players were perhaps overrated. 

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

Please refer to the [Jupyter Notebook Viewer](https://nbviewer.org/github/jacquelinekclee/nba-all-stars-classifier.github.io/blob/3779271415ee1cf80baaaf659555743dc93966e9/nba_all_stars_classifier.ipynb) or the [.ipynb file](https://github.com/jacquelinekclee/nba-all-stars-classifier.github.io/blob/3779271415ee1cf80baaaf659555743dc93966e9/nba_all_stars_classifier.ipynb) to view all the code for the classifier. The notebook with all the data cleaning can be seen [here](https://github.com/jacquelinekclee/nba-all-stars-classifier.github.io/blob/3779271415ee1cf80baaaf659555743dc93966e9/nba_players_data_cleaning.ipynb). 

The [source file](#source-file) contains all the functions used to clean/manipulate the data and DataFrames.

[(Back to top)](#table-of-contents)

## The Methodology
Since only about 5% of players in a given season are designated All Stars, a classifier that does well predicting the minority class is needed. Additionally, instead of using accuracy as the evaluation metric, recall was chosen instead given my interest in capturing all the All Star players in each season. In [this Medium article](https://towardsdatascience.com/classifying-rare-events-using-five-machine-learning-techniques-fab464573233), by Leihua Ye, PhD found that K-Nearest Neighbors performed best for prediciting rare events, especially when looking at ROC Curves. Given these findings, KNN's lack of of training time, and its simplicity, KNN was chosen. 

[(Back to top)](#table-of-contents)

## Training
The main hyperparamter in KNN is of course the number of neighbors `k`. Several sources indicate that `n ** 2`, where $n$ is the number of instances in the sample, is a good heuristic for best `k`. Other important hyperparameters are `weights`, or how the nearest neighbors are used in determining the final prediction, and `metric`, or which distance metric KNN uses. I used Scikit's `GridSearchCV` to combine both hyperparameter tuning and training the model with cross validation to determine the final model. See below for the parameters tested and the ultimate model parameters:

Parameters tested:
`{'clf__n_neighbors' : [19, 21, 23, 25, 27], 'clf__weights' : ['uniform','distance'], 'clf__metric' : ['minkowski','euclidean','manhattan']}`

Best paramters:
`{'clf__metric': 'minkowski', 'clf__n_neighbors': 19, 'clf__weights': 'distance'}`

[(Back to top)](#table-of-contents)

## Findings
Below is a table summarizing the recall scores:
| Season    |   Recall Score |
|:----------|---------------:|
| 2019-2018 |          0.615 |
| 2020-2021 |          0.741 |
| 2021-2022 |          0.667 |

The classifier was able to find a majority of each season's all stars, performing best for the 2020-21 season. 

## The Results
See below for some discussion on the model's predictions. See the table linked [here](/all_star_classifier_summary.csv) to see a full summary of the 3 NBA seasons used in testing and the results for each player. 

### "Properly Rated" All Stars: The True Positives
There were only 6 players who were voted as All Stars for all 3 seasons (2018-19, 2020-21, and 2021-22) *and* were deemed "properly rated" for all 3 seasons. In this case, "properly rated" means that the all star voting seemed to match the data and the KNN model's findings.
- Giannis Antetokounmpo
- James Harden
- Joel Embiid
- Kevin Durant
- LeBron James
- Stephen Curry

### "Overrated" All Stars: The False Negatives
The "overrated" players are instances where the classifier predicted a player *wasn't* an All Star when in reality they were. These cases could include players that perhaps contribute in ways that don't show up on the stat sheet or players that a particular fan favorites. For instance, in the 2018-19 season, future hall-of-famers and NBA champions Dirk Nowitzki and Dwyane Wade were voted as All Stars, but the classifier thought differently. That season was both Nowitzki and Wade's last season in the NBA, so while their stats maybe weren't up to par, their legendary careers earned them the designation. 

For 2 of the 3 seasons used in testing, Ben Simmons (2018-19 and 2020-21), Khris Middleton (2018-19 and 2021-22), Nikola Vučević (2018-19 and 2020-21), and Rudy Gobert (2020-21 and 2021-22) all fell under the "overrated" category. Middleton might not have the opportunity to shine in the stats playing alongside generational talent Giannis Antetokounmpo, thus resulting in a false negative. The same could be said about Ben Simmons who played with Joel Embiid, but his recent (lack of) play may make fans agree with the "overrated" designation. 

Another interesting case of "overrated" all stars is Nikola Jokić. The model deemed him to be "overrated" in 2018-19, but in the 2 previous seasons (during which he got consecutive MVP honors), the model correctly predicted him as an All Star. 

### "Underrated" Players: The False Positives
The "underrated" players are those that the classifier predicted to be all stars, but weren't voted as all stars in reality. Perhaps the stats of such a player were exceptional, but other aspects like their winning percentages weren't up to par. See below for the players from the 2018-19, 2020-21, and 2021-22 seasons that were deemed underrated and were never selected as an All Star in any of the 3 seasons:
- Brandon Ingram (2020-21)
- CJ McCollum (2020-21)
- John Wall (2018-19)
- Jrue Holiday (2019-19)
- Pascal Siakam (2021-22)
- Shai Gilgeous-Alexander (2020-21, 2021-22)

One interesting player here is Pascal Siakam, who was voted an All Star in the 2019-2020, the season after he won a championship. Siakam was perhaps overlooked by fans and/or the media playing in Toronto (as opposed to in the U.S. or a larger market team). For 2 seasons in a row, young guard Shai Gilgeous-Alexander was "underrated" in the eyes of the model. His team, the OKC Thunder, had a terrible record this past season and finished in 14th place in the Western Conference. This lack of winning, and the fact that OKC is a smaller market, may explain why Gilgeous-Alexander was overlooked in all star voting.

[(Back to top)](#table-of-contents)

## Source File
- [nba_players_classification.py](https://github.com/jacquelinekclee/nba-all-stars-classifier.github.io/blob/a6dd72fb58304c2a2c4c9125883913b60662731f/nba_players_classification.py)
  - Has all the functions used for mainly the data cleaning.

## Legality
This personal project was made for the sole intent of applying my skills in Python thus far and as a way to learn new ones. It is intended for non-commercial uses only.

[(Back to top)](#table-of-contents)
