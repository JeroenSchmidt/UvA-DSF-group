# ASSIGNMENT
**Use 2 Case: Business, predicting Well-being from Instagram data**

 

## Introduction
 

A growing proportion of social interactions are now mediated by digital services and devices. Such digitally mediated behaviors can easily be recorded and analyzed, fueling the emergence of computational marketing and social science. Researchers have used social media to predict individual and aggregated measures of heart attacks, political preference, personality and perhaps most importantly well-being. Well-being, which is defined as peoples’ positive evaluations of their lives, includes positive emotion, engagement, satisfaction, and meaning (Diener and Seligman, 2004).
 

Previous research, however, has been based on text input, usually based on Facebook and Twitter. Your assignment is to assess the relationship between visual social media data, in this case Instagram, and well-being (Park et al 2016). While a recent study (Reece and Danforth 2016) demonstrated a relationship between Instagram user posts and clinical markers of depression, no study has however looked at the relation between Instagram posts and well-being. Well-being is measured through a survey using the PERMA scale (Seligman 2012). Just like the “state” of an airplane is not given by a single indicator but instead by a variety of different indicators (altitude, speed, head-ing, fuel consumption) — well-being is best measured as separate, correlated dimensions, Positive Emotions, Engagement, Relationships, Meaning, and Accomplish (PERMA).
 

## Data
 

Users from crowdsourcing platforms (Mechanical Turk and Microwork) where asked to login with their Instagram account and fill out the survey including the PERMA scale. Next we extracted all images a user posted and extracted different features for each image. Please see the data document for a description of individual variables.
 

## Assignment
 

Your assignment is to assess the nature of the relationship between user Instagram activity and the content of the images and their well-being. At your disposal you have data on the user, images and their metadata and features extracted from the images (e.g. sentiment, faces).
In your analysis consider the individual aspects of well-being (separate PERMA factors) as well a the PERMA score itself. Consider the hierarchical relationship between the predictor variables (user -> images -> image features) and please note that the dependent variable only has 160 observations. (= filled out questionnaires) The sample size thus might require additional procedures to ensure you can draw confident conclusions from your analysis.
 

References below might provide interesting methods for your analysis.

 

## Suggested starting point
 

* Split the data into training and testing splits
* Train a regressor to predict the PERMA scores on the test set using different sets of attributes (not all of them at once)
* Analyze which features (attributes) correlate well with each other and help in fitting the curve to the data better.
* Elaborate on the results.
 

## References
 

* Diener, Ed and Martin E. P. Seligman (2004), “Beyond Money: Toward an Economy of Well-Being,” Psychological Science in the Public Interest: A Journal of the American Psychological Society, 5, 1, 1–31.

* Park, G., D. Stillwell, and M. Kosinski (2016), “Predicting Individual Well-Being through the Language of Social Media,” : Proceedings of the …, davidstillwell.co.uk, http://www.davidstillwell.co.uk/articles/2016_predicting_wellbeing.pdf (Links to an external site.)Links to an external site..

* Reece, Andrew G. and Christopher M. Danforth (2016), “Instagram Photos Reveal Predictive Markers of Depression,” arXiv [cs.SI], arXiv. http://arxiv.org/abs/1608.03282 (Links to an external site.)Links to an external site..

* Seligman, Martin E. P. (2012), Flourish: A Visionary New Understanding of Happiness and Well-Being, Simon and Schuster.
