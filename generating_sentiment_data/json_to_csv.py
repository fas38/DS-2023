import json
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

period_start_list = ["2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"]
period_end_list = ["2013-01-31", "2014-01-31", "2015-01-31", "2016-01-31", "2017-01-31", "2018-01-31", "2019-01-31", "2020-01-31", "2021-01-31", "2022-01-31", "2023-01-31"]

# ================= for company info =================
print("Processing company info...")
data_df_list = []
for item in zip(period_start_list, period_end_list):
    start_date = item[0]
    end_date = item[1]
    file_name = "dataset/company_" + start_date + "_" + end_date + ".json"
    
    
    print("processing time period: " + start_date + " to " + end_date)

    with open(file_name) as f:
        data = json.load(f)

    company_names = data.keys()
    company_names = list(company_names)

    # iterate over all companies
    for company_name in company_names:
        temp = data[company_name]
        df = pd.DataFrame(columns=["ticker", "start", "end", "sentiment"])
        for i in range(len(temp)):
            start = temp[i]["start"]
            end = temp[i]["end"]
            sentiment = temp[i]["sentiment"]
            df = df.append({"ticker": company_name, "start": start, "end": end, "sentiment": sentiment}, ignore_index=True)
        data_df_list.append(df)

print("Concatenating dataframes...")
data_df = pd.concat(data_df_list, ignore_index=True)
# print(data_df.head(20))

data_df = data_df.sort_values(by=["ticker", "start"])
data_df = data_df.reset_index(drop=True)
# print(data_df.head(20))

print("Saving to CSV...")
data_df.to_csv("../dataset/company_2012-01-01_2022-12-31.csv", index=False)



# ================= for industry info =================
print("Processing industry info...")
data_df_industry_list = []

for item in zip(period_start_list, period_end_list):
    start_date = item[0]
    end_date = item[1]
    file_name = "dataset/industry_" + start_date + "_" + end_date + ".json"

    print("processing time period: " + start_date + " to " + end_date)

    with open(file_name) as f:
        data = json.load(f)

    industry_names = data.keys()
    industry_names = list(industry_names)

    # iterate over all industries
    for industry_name in industry_names:
        temp = data[industry_name]
        df = pd.DataFrame(columns=["industry", "start", "end", "sentiment"])
        for i in range(len(temp)):
            start = temp[i]["start"]
            end = temp[i]["end"]
            sentiment = temp[i]["sentiment"]
            df = df.append({"industry": industry_name, "start": start, "end": end, "sentiment": sentiment}, ignore_index=True)
        data_df_industry_list.append(df)

print("Concatenating dataframes...")
data_df_industry = pd.concat(data_df_industry_list, ignore_index=True)
# print(data_df_industry.head(20))

data_df_industry = data_df_industry.sort_values(by=["industry", "start"])
data_df_industry = data_df_industry.reset_index(drop=True)
# print(data_df_industry.head(20))

print("Saving to CSV...")
data_df_industry.to_csv("../dataset/industry_2012-01-01_2022-12-31.csv", index=False)


# ================= combine company with industry score =================
print("Combining company with industry score...")

entertainment_df = pd.read_csv("dataset/stocks_entertainment.csv")
entertainment_df["industry_type"] = "entertainment"
internet_content_df = pd.read_csv("dataset/stocks_internet-content-and-information.csv")
internet_content_df["industry_type"] = "internet-content-and-information"
travel_df = pd.read_csv("dataset/stocks_travel-services.csv")
travel_df["industry_type"] = "travel-services"

combined_df = pd.concat([entertainment_df, internet_content_df, travel_df], ignore_index=True)

df_list = []
for _, item in combined_df.iterrows():
    industry_type = item["industry_type"]
    ticker = item["symbol"]
    company_name = item["company_name"]

    data_df_subset = data_df[data_df["ticker"] == ticker]
    data_df_industry_subset = data_df_industry[data_df_industry["industry"] == industry_type]

    # merge company and industry dataframes
    merged_df = pd.merge(data_df_subset, data_df_industry_subset, left_on=["start", "end"], right_on=["start", "end"], how="inner")
    merged_df["sentiment"] = merged_df["sentiment_x"] + merged_df["sentiment_y"]
    # merged_df = merged_df.drop(columns=["sentiment_x", "sentiment_y"], axis=1)
    merged_df = merged_df.rename(columns={"sentiment_x": "sentiment_company", "sentiment_y": "sentiment_industry"})
    merged_df["company_name"] = company_name
    df_list.append(merged_df)

combined_df = pd.concat(df_list, ignore_index=True)
combined_df = combined_df.sort_values(by=["ticker", "start"])
combined_df = combined_df.reset_index(drop=True)
# print(combined_df.head(20))

print("Saving to CSV...")
combined_df.to_csv("../dataset/combined_2012-01-01_2022-12-31.csv", index=False)


    
    
    
    