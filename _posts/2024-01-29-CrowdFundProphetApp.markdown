---
layout: post
title: CrowdFundProphet- A Crowdfunding Prediction App Powered by an Ensemble of Binary Classification Machine Learning Models  
image: CrowdFundProphetApp.png
date: 2024-01-29 6:21:20 +0400
tags: [Streamlit, App, Binary Classification, Machine Learning, Cloud,  Data Model, Data Science]
categories: [App, LLM, Businesses, Data_Science, Finance, Data_Analytics]
---
Checkout the deployed app in action: [http://ec2-40-176-10-24.ca-west-1.compute.amazonaws.com:8502](http://ec2-40-176-10-24.ca-west-1.compute.amazonaws.com:8502) 

The link to the github repo for this app is [here](https://github.com/apsinghAnalytics/CrowdFundProphetApp). Please check the readme in this repository for deployment instructions of this app to an aws ec2 instance and the requirements.txt file for the packages used.

I'm excited to share with you a bit about a prediction tool I've been working on – *the CrowdFundProphet*! It's powered by a sophisticated machine learning model that I developed as part of my [CrowdOfferingsStudy](https://github.com/apsinghAnalytics/CrowdOfferingsStudy) project. At its heart is a **stacked classifier**, crafted for binary classification. This nifty model is what gives CrowdFundProphet its superpower: making predictions with roughly **80% accuracy**! (Compared to a baseline majority class accuracy of **57.7%** )

In future edits/updates to this blog post, I'll dive deeper into the CrowdOfferingsStudy project. But if you're curious now, feel free to check out the detailed Jupyter notebooks linked below:

#### Data Gathering

 Our journey starts with a Python [web scraper](https://github.com/apsinghAnalytics/CrowdOfferingsStudy/blob/main/Scraper.py) that collected data from all the Form C SEC filings (Form C filings are for Crowdfunding campaigns launched at various Platforms or Intermediaries) from 2016 up to Q3, 2023.

#### Data Cleaning

Next, we gave our data a good scrub. You can see all the nitty-gritty in this [notebook](https://github.com/apsinghAnalytics/CrowdOfferingsStudy/blob/main/DataCleaning.ipynb).

#### Model Building 

The clean data then paved the way for building our ML models. Curious about how it was done? Check out this [notebook](https://github.com/apsinghAnalytics/CrowdOfferingsStudy/blob/main/DataModel.ipynb).

#### Visualization

The cleaned data was also used to create a sleek **Tableau dashboard** to visualize the trends. You can explore it [here](https://public.tableau.com/app/profile/aditya.prakash.singh/viz/CrowdfundingTrendsDashboard/Dashboard). (Quick note for those on smaller desktop screens: switch to **'Desktop Layout'** on Tableau Public for the best view)

Stay tuned for more updates and insights. 