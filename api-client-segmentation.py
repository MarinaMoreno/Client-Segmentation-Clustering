#pip install fastapi
#pip install uvicorn[standard]
from fastapi import FastAPI
from pydantic import BaseModel #Guarantee that each attibute in the class correspond to the data type specified in the class Customer defined
from fastapi.encoders import jsonable_encoder #Transform class object to dict
from datetime import date
import pickle
import pandas as pd
import numpy as np

class Customer(BaseModel):
    ID: int
    Year_Birth: int
    Education: int
    Marital_Status: str
    Income: int
    Kidhome: int
    Teenhome: int
    Recency: int
    MntWines: int
    MntFruits: int
    MntMeatProducts: int
    MntFishProducts: int
    MntSweetProducts: int
    MntGoldProds: int
    NumDealsPurchases: int
    NumWebPurchases: int
    NumCatalogPurchases: int
    NumStorePurchases: int
    NumWebVisitsMonth: int
    AcceptedCmp3: int
    AcceptedCmp4: int
    AcceptedCmp5: int
    AcceptedCmp1: int
    AcceptedCmp2: int
    Complain: int
    Response: int
    EnrolledDays: int
    class Config:
            schema_extra = {
            "examples": [
                {
                    "ID": 1,
                    "Year_Birth": 1957,
                    "Education": 2,
                    "Marital_Status": "Single",
                    "Income": 57962,
                    "Kidhome": 0,
                    "Teenhome": 0,
                    "Recency": 58,
                    "MntWines": 635,
                    "MntFruits": 88,
                    "MntMeatProducts": 546,
                    "MntFishProducts": 172,
                    "MntSweetProducts": 88,
                    "MntGoldProds": 88,
                    "NumDealsPurchases": 3,
                    "NumWebPurchases": 8,
                    "NumCatalogPurchases": 10,
                    "NumStorePurchases": 4,
                    "NumWebVisitsMonth": 7,
                    "AcceptedCmp3": 0,
                    "AcceptedCmp4": 0,
                    "AcceptedCmp5": 0,
                    "AcceptedCmp1": 0,
                    "AcceptedCmp2": 0,
                    "Complain": 0,
                    "Z_CostContact": 3,
                    "Z_Revenue": 11,
                    "Response": 1,
                    "EnrolledDays": 55
                }
            ]
        }
    
app = FastAPI()

@app.get("/health")
def index():
    return "OK"

import pickle
clustering_model = pickle.load(open("clustering_clients_kmeans.pickle", 'rb'))

@app.post("/GetCustomerGroup")
def get_customer_group(customer: Customer):
    #Transform input class object to a dataframe
    df = pd.DataFrame([jsonable_encoder(customer)]) 
    print("Input received:\n")
    print(df.T)

    #Create new columns mandatory for the model
    df["Spent"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    df["Children"]=df["Kidhome"] + df["Teenhome"]
    df["Is_Parent"] = np.where(df.Children> 0, 1, 0)
    df["Customer_purchase_deal"] = df['NumDealsPurchases'] > 0
    df["Purchases"] = df["NumWebPurchases"]+ df["NumCatalogPurchases"]+ df["NumStorePurchases"]
    df["Total_Offers_Accepted"] = df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] + df["AcceptedCmp4"] + df["AcceptedCmp5"] + df["Response"]

    #Create dummy marital status feautures mandatory for the model
    df["Marital_Status_Divorced"] = 0
    df["Marital_Status_Single"] = 0
    df["Marital_Status_Together"] = 0
    df["Marital_Status_Widow"] = 0
    df["Marital_Status_" + df.loc[0,"Marital_Status"]] = 1

    # Save ID for the response
    id = df.loc[0,"ID"]
    # Remove features which the model wasn't trained
    df.drop(columns=["ID","Marital_Status"], inplace=True)

    #Sort columns in the same order the model was trained
    df = df.reindex(columns=sorted(df.columns))

    #Apply model and return respone
    cluster_id = clustering_model.predict(df)[0]
    cluster_name = "high income level" if cluster_id == 0 else "low income level"
    return("The customer with ID: " + str(id) + " belong to group of customers with " + cluster_name + ", with id group: " + str(cluster_id))

# To run the api, in the terminal: uvicorn api-client-segmentation:app --reload
#http://127.0.0.1:8000/docs
