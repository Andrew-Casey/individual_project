# Individual_project (PGA cut predictions)

### Goal:
- This data science project will attempt to predict whether or not a player will make the cut, based on previous weeks performance, coupled with season averages(drive distance, fairway percentage, putting average).

### Description:

- Is it possible to predict 'making the cut' or not at a PGA event?

- Making the cut is important for players and sponsors alike.
    - A typical PGA event is 4 rounds of golf. Thursday - Sunday
    - 'The top 70 players (including ties) make the cut. Any player in 71st place or worse (after Friday's round) is cut.' - pga.com

- From a players perspective, making the cut is important for many reasons.
    - If a player misses the weekend, they do not get paid.
    - Also, prized FEDEX cup points are missed. ('The FedEx Cup is the central professional golf competition that compares player performance across part of the PGA Tour season in a League and Playoffs structure.')
    - For continued poor performance, their tour eligibilty could be in jeopardy.

- From a sponsors perspective, making the cut equates to 'eyes on product'.
    - Players making the weekend are televised more. Sunday draws the largest viewership of the tournament days.
        - 2-4 million viewers for non-majors
        - 5 million for majors
        - 10 million for 'The Masters'



# Data Dictionary


| Variable               | Description                           |
|------------------------|---------------------------------------|
| date                   | Date of the event                       |
| course                 | Name of the golf course                 |
| tournament name        | Name of the tournament                   |
| made_cut               | Indicator of whether the player made the cut or not |
| sg_putt_prev           | Strokes gained/lost putting in previous events |
| sg_arg_prev            | Strokes gained/lost around-the-green in previous events |
| sg_app_prev            | Strokes gained/lost approach in previous events |
| sg_ott_prev            | Strokes gained/lost off-the-tee in previous events |
| sg_t2g_prev            | Strokes gained/lost tee-to-green in previous events |
| sg_total_prev          | Total strokes gained/lost in previous events |
| driving_avg            | Distance of the player's drives in yards |
| fairways_hit           | Percentage of fairways hit by the player |
| putting_avg            | Average number of putts per hole        |



# Project Planning:
- Acquire a dataset that peaks a personal interest - Golf
- Prepare/clean the data for exploration
- Split data into train, validate and test
- Explore the data for significant features and answer questions we have about the data set
    - Perform statistical tests that follow assumptive guidelines
        - number of observations > 30
        - test for equal variance and ensure 'equal_var' is properly set based on Levene's test
        - normal distribution (using the central limit theorm for this study)
    - Allow the data to guide us.
- Model on scaled train,validate data sets
    - Run model for accuracy on test dataset to see if there is predictive power 
        - Predict whether or not a player will make the cut at a PGA event
- Make recommendations and formulate next steps to continue the project


# Initial Questions:
- Do strokes gained putting in the previous week affect whether or not someone will make the cut this week?
- Do strokes gained around green in the previous week affect whether or not someone will make the cut this week?
- Do strokes gained approach in the previous week affect whether or not someone will make the cut this week?
- Do strokes gained off-the-tee in the previous week affect whether or not someone will make the cut this week?
- Do strokes gained tee-to-green in the previous week affect whether or not someone will make the cut this week?
- Do strokes gained total in the previous week affect whether or not someone will make the cut this week?
- Does driving average affect whether or not someone will make the cut?
- Does fairways hit percentage affect whether or not someone will make the cut?
- Does putting average affect whether or not someone will make the cut?


# Key findings:
- Players that make the cut gain .25 strokes on the field in 'strokes gained approach'
    - 'Approach measures a players shots after the initial drive, so it highlights which golfers have excelled in those crucial second and third shots.'
    - Second and third shot (accuracy) is a premium stat when trying to make the cut
- Players who make the cut gain .50 strokes on the field in 'strokes gained tee-to-green'
    - 'Tee-to-Green is perhaps the best metric of them all. It’s essentially Total Strokes Gained — which is the sum of off-the-tee, approach, around-the-green'
        - This stat covers complete game. Tee ball accuracy, iron play, and chipping around the green.
        - If players are lagging in this stat, it may uncover weakness in their game and provide insight for focused practice
- Driving average/Putting average
    - Those that make the cut are 2 yards longer on average (296 vs. 294)
    - Those that make the cut average 1.76 putts per hole vs. 1.77 of those that do not
        - Golf is a game of inches and on any given Friday, the total game is required to make the cut
            - And.... maybe a little luck
 


# Steps to reproduce:

- Aquire data from 'https://zenodo.org/record/5235684'
    - save to your working directory and read into a notebook using pd.read_csv(name_of_file)
- Prep the data for exploration: 
    - using function 'prep_pga()' in the prepare.py file included in the Github repository
- Split data
    - using function 'split_data()' in the preare.py file included in the Github repository 
    - data is split into train validate and test datasets using the 60,20,20 method.
- Explore the data using visualizations and stats tests.
- Using the best features run classification models on the train and validate to find the best model.
- Logistic Regression was the best/most consistent performing model.
- Read over the outputs and form a conclusion and summarize with recommendations/nextsteps.

# Summary
- The data analysis pipeline (Acquire, Prepare, Explore, Modeling) was completed.
- All features selected were statistically significant.
    - These features seemed to be the most important factors
        - Strokes gained approach
        - Strokes gained off-the-tee
        - Strokes gained tee-to-green
        - Strokes gained total
- Scaled data with selected features were ran through 4 different predictive models:
    - Decision Tree
    - Random Forest
    - KNN
    - Logistic Regression
        - Out of the four models, logistic regression performed the best.
            - The model scored 61% on the unseen test data set. 5% better than the baseline of 56%.
            
# Recommendations
- Using this model as a guide, players and sponsors can predict next weeks performance (with 61% accuracy) based on last weeks performance.
    - In preparation for a tournament, focus on the following to ensure best chances of making the cut:
        - Strokes gained approach
        - Strokes gained off-the-tee
        - Strokes gained tee-to-green
            - Tee ball accuracy and iron accuracy are premium stats. They add the most strokes gained to players scorecards in average
    - Caveat 
        - There are many unquantifiable variables to predict with 100% certainty whether or not a player will make the cut or not. 
        - This is a better than baseline prediction 
            - Players can use this a a guide to focus efforts in training
            - Sponsors can use this data to find players to support
        
# Next Steps
- To further investigate the possibility of a more powerful predictive model hidden within this dataset:
    - I will continue to work on this data set
        - One possible idea would be to identify good and bad weather players:
            - To do this, I would feature engineer weather attributes, possibly through clustering
        - Another direction for predictive modeling:
            - Create models that are course dependent
            - Courses are unique and set up to players various strengths and weaknesses 
        - Explore player demographics (age, height, weight).
