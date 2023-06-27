## This branch is for the development of the app.


### Backend
The backend script is in server.py. To start the flask server just execute the script from terminal and it will receive post request on the following url: 

    localhost:8080/predict

The post request should contain a json object with the following structure:

    {
        "company": "aPPle", # company name - case insensitive
        "predict_type": "Day", # Day, Week, Month - case insensitive
        "predict_period": 1, # number of days, weeks or months to predict
    }

It will return a json object with the following structure:

    {
        "actual_values": [#list of actual daily returns for the company],
        "predicted_values": [#list of predicted daily returns for the company]
    }

Example output for next 10 days of Apple stock:

    {
        "actual_values": [
            0.4923712909221649,
            0.32823896408081055,
            -0.07704219967126846,
            -0.07704219967126846,
            -0.07704219967126846,
            -0.2694390118122101,
            -0.23343391716480255,
            1.0812219381332397,
            -0.1950482726097107,
            -0.744105875492096
        ],
        "predicted_values": [
            -0.07204285264015198,
            -0.07138592004776001,
            -0.08207884430885315,
            -0.05768775939941406,
            -0.06906194239854813,
            -0.07096996158361435,
            -0.054937347769737244,
            -0.06236839294433594,
            -0.06878367811441422,
            -0.05906323343515396
        ]
    }

Example output for next day of Apple stock:

    {
        "actual_values": [
            0.4923712909221649
        ],
        "predicted_values": [
            -0.032223597168922424
        ]
    }