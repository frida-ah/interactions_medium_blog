# Interaction effects in linear regression
This code was created for the Medium blog "Interaction effects in linear regression".
https://medium.com/p/77ae7245eecd


### How to run code locally
In the Terminal:
- Make sure Miniconda (or Anaconda) is installed. If you prefer another option, you will not need to follow the following steps.
- Clone GitHub code
``` git clone git@github.com:frida-ah/interactions_medium_blog.git```
- Create a conda environment  
``` conda create -n interactions-blog python=3.9  ```
- Activate the new conda environment
``` conda activate interactions-blog ```
- Install the required python packages
``` pip install -r requirements.txt ```

### Data sources
#### Keyword searches
The timeseries within Google Trends search data mimics the real sales of a product. https://trends.google.com/trends/explore?cat=71&date=today%205-y&geo=NL&q=ijs 

#### Temperature in the Netherlands
In order to run the code, you will first need to donwload the weather data from https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/. The observation type TAVG which represents the average temperature of a timestamp, is measured in the tenth of degrees Celsius in the dataset. So, it needs to be converted to Celsius degrees by dividing with 10.

#### School holidays in the Netherlands
School holidays were scraped using the workalendar package. The north region was chosen for code simplicity to download the days when schools are closed in the Netherlands. However, in order to make a production-ready model, it is advised to take into account the rest of the regions, so it is a match with the average temperature in all of the regions in the Netherlands.

### Modeling
Two linear regression models are fit, one with and one without interaction effects. Also, a LightGBM model is fit with the most basic parameteres. The modeling is kept to the bare minimum for the sake of simplicity. A simple cross-validation was implemented to make the model accurate. 

### Outcome of the models being implemented
A linear regression with interactions achieves better accuracy than a linear regression without interactions. In this example, they are both more accurate than a flexible model such as LightGBM. Also,  the interaction effects assist in interpretability. We can easily explain how the model came to its decisions since we know which features and interactions were chosen and we have the power to adjust the model so it fits our problem better.