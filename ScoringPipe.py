import luigi
from luigi_slack import SlackBot, notify
import pandas as pd
import numpy as np
from numpy import newaxis
import datetime
import pickle
import time
import os
import heapq
import glob
from sklearn.externals import joblib
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from slackclient import SlackClient
from pytz import timezone
import quandl

# Run with : python ScoringPipe.py --scheduler-host localhost StartScoringPipe --PredictionTimepoints 2
#####################
# NOTE FH : START AT preparedataann -> must bring input to correct shape for socring
#######################
class GetData(luigi.Task):
    PredictionTimepoints = luigi.Parameter()
    root = luigi.Parameter()

    def getalphadata(self, companiesAlpha, ts):
        finaldatacomp, metadata = ts.get_daily_adjusted(symbol='ATVI',outputsize="compact")
        finaldata = pd.DataFrame(finaldatacomp["5. adjusted close"])
        finaldata.columns = ["ATVI"]
        i = 0
        for company in companiesAlpha:
            i += 1
            data, metadata = ts.get_daily_adjusted(symbol=company,outputsize="compact")
            datatemp = pd.DataFrame(data["5. adjusted close"])
            datatemp.columns = [company]
            finaldata = finaldata.join(datatemp)
            if i == 1:
                time.sleep(30)
                i = 0
        return finaldata

    def requires(self):
        None
    def run(self):

        companiesAlpha = ["ADBE", "AKAM", "ALXN", "GOOGL", "AMZN", "AAL", "AMGN", "ADI", "AAPL", "AMAT", "ADSK", "ADP",
                          "BIDU", "BIIB", "BMRN", "CA", "CELG", "CERN", "CHKP", "CTAS", "CSCO", "CTXS", "CTSH", "CMCSA",
                          "COST", "CSX", "XRAY", "DISCA", "DISH", "DLTR", "EBAY", "EA", "EXPE", "ESRX", "FAST", "FISV",
                          "GILD", "HAS", "HSIC", "HOLX", "IDXX", "ILMN", "INCY", "INTC", "INTU", "ISRG", "JBHT", "KLAC",
                          "LRCX", "LBTYA", "MAR", "MAT", "MXIM", "MCHP", "MU", "MDLZ", "MSFT", "MNST", "MYL", "NFLX",
                          "NVDA", "ORLY", "PCAR", "PAYX", "QCOM", "REGN", "ROST", "STX", "SIRI", "SWKS", "SBUX", "SYMC",
                          "TSLA", "TXN", "TSCO", "TMUS", "FOX", "ULTA", "VRSK", "VRTX", "VIAB", "VOD", "WBA", "WDC",
                          "WYNN", "XLNX"]  # "PCLN",
        # Download via API
        tickerstart = time.time()

        print("Getting alphavantage data...")

        with open(self.root+"/Meta/alphavantage.txt", "r") as myfile:
            alphatoken = myfile.readlines()
        ts = TimeSeries(key=alphatoken, output_format='pandas', retries=5)
        self.mydata = self.getalphadata(companiesAlpha, ts)

        tickerend = time.time()
        print("Data downloaded in {} s".format((tickerend - tickerstart)))
        print("No Companies: {}".format(self.mydata.shape[1]))

        # save as csv with current date
        with self.output().open('w') as outfile:
                self.mydata.to_csv(outfile)

    def output(self):
        self.now = datetime.datetime.now(timezone("America/New_York"))

        return luigi.LocalTarget(self.root+"/Data/ScoringData-"+str(self.now.year)+str(self.now.month)+str(self.now.day)+".csv")


class CheckInputforNans(luigi.Task):
    PredictionTimepoints = luigi.Parameter()
    root = luigi.Parameter()

    def requires(self):
        return GetData(self.PredictionTimepoints,self.root)

    def run(self):
        #self.now = datetime.datetime.now()
        # NASDAQ time
        self.now = datetime.datetime.now(timezone("America/New_York"))
        # load from target
        self.mydata = pd.read_csv(self.input().path, parse_dates = True, index_col = "date")

        # Slice data to get last  day (PredictionTimepoints)
        Delta = datetime.timedelta(days=1)
        self.mydata = self.mydata[self.now-Delta : self.now]

        if len(self.mydata)<1:
            raise ValueError("No Data for yesterday!")

        # Print number of NaNs, throw exception if more than 5
        NumberofRowNans = len(self.mydata[self.mydata.isnull().any(axis=1)])
        if NumberofRowNans > 0:
            raise ValueError("NaNs detected in scoring data.")


    def output(self):
        return  self.input()

class PrepareDataForScoring(luigi.Task):
    PredictionTimepoints = luigi.Parameter()
    root = luigi.Parameter()

    def requires(self):
        return CheckInputforNans(self.PredictionTimepoints,self.root)

    def run(self):
        self.PredictionTimepoints = int(self.PredictionTimepoints)
        # Data preparation
        self.mydata = pd.read_csv(self.input().path, index_col=0, parse_dates=True)

        NumberofCompanies = self.mydata.shape[1]

        FirstIndex = self.PredictionTimepoints
        MaxPoints = self.mydata.shape[0] - FirstIndex

        MLP = True
        normalization = True

        # create np array for Data Collection
        DataCollection = np.empty([1, self.PredictionTimepoints, NumberofCompanies])

        # Create copy of data frame for handling
        mydataPP = self.mydata.copy(deep=True)

        # Normalization if required
        Scalers = glob.glob(self.root+"/saved_models_pipe/*Scaler*.pkl")
        RecentScaler = Scalers[-1]
        scaler = joblib.load(RecentScaler)
        if (normalization == True):
            mydataNP = self.mydata.values
            mydataNormalizedNP = scaler.transform(mydataNP)
            mydataPP = pd.DataFrame(mydataNormalizedNP)
            mydataPP.columns = self.mydata.columns
            mydataPP.index = self.mydata.index

        # START CREATE INPUT VECTORS (IMAGES)

        AdjCloseTemp = mydataPP.iloc[len(mydataPP)-(self.PredictionTimepoints) : ]  # e.g. 0 - 249 inclusive, as last index is not sliced

        AdjCloseTemp_Array = AdjCloseTemp.values
        ###########################################
        ###SLACK MESSAGE POST
        ###########################################
        SlackMsg = "Timestamps used for scoring (N="+ str(len(AdjCloseTemp)) + "):"
        sc = SlackClient(token)
        sc.api_call(
            "chat.postMessage",
            channel="predictions",
            text=SlackMsg
        )
        for Timestamp in AdjCloseTemp.index:
             # Post data to slack for control
             SlackMsg = Timestamp.strftime("%d.%m.%y")
             sc = SlackClient(token)
             sc.api_call(
                 "chat.postMessage",
                 channel="predictions",
                 text=SlackMsg
             )
        ###########################################

        arrayAdjClosedTemp = np.array(AdjCloseTemp_Array, np.float32)[newaxis, :, :]
        DataCollection = np.append(DataCollection, arrayAdjClosedTemp, axis=0)

        # END CREATE IMAGES
        DataCollection = DataCollection[1:DataCollection.shape[0], :, :]

        print("#############################")
        print("Scoring Data Shape prepared for ANN:")
        print(DataCollection.shape)
        print("#############################")


        with open(self.output()["X_score"].path, 'wb') as save_file:
            pickle.dump(DataCollection, save_file)


    def output(self):
        self.now = datetime.datetime.now(timezone("America/New_York"))
        return {
                 "X_score": luigi.LocalTarget(self.root + "/Data/X_score-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "RawData": self.input()
                }


class ScoreModel(luigi.Task):
    PredictionTimepoints = luigi.Parameter()
    root = luigi.Parameter()

    def MLP_B2(self):
        model = Sequential()
        model.add(Flatten(input_shape=(int(self.PredictionTimepoints), self.NumberofCompanies)))

        # model.add(Dense(5000, activation='relu'))
       # model.add(Dense(2000, activation='relu'))
       # model.add(Dropout(0.2))
       # model.add(Dense(1000, activation='relu'))
       # model.add(Dropout(0.2))
       # model.add(Dense(500, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(87))
        model.compile(loss='mean_squared_error', optimizer="adamax", metrics=['mse'])

        return model

    def requires(self):
        return PrepareDataForScoring(self.PredictionTimepoints,self.root)

    def run(self):
        # Load required (prepared) data
        pickle_in = open(self.input()["X_score"].path, "rb")
        self.X_score = pickle.load(pickle_in)
        self.NumberofCompanies = self.X_score.shape[2]

        # Check for newest model
        Models = glob.glob(self.root+"/saved_models_pipe/*MLPtype2_B2*.hdf5")
        RecentModel = Models[-1]

        model = self.MLP_B2()
        model.load_weights(RecentModel)

        companiesAlpha = ["ADBE", "AKAM", "ALXN", "GOOGL", "AMZN", "AAL", "AMGN", "ADI", "AAPL", "AMAT", "ADSK", "ADP",
                          "BIDU", "BIIB", "BMRN", "CA", "CELG", "CERN", "CHKP", "CTAS", "CSCO", "CTXS", "CTSH", "CMCSA",
                          "COST", "CSX", "XRAY", "DISCA", "DISH", "DLTR", "EBAY", "EA", "EXPE", "ESRX", "FAST", "FISV",
                          "GILD", "HAS", "HSIC", "HOLX", "IDXX", "ILMN", "INCY", "INTC", "INTU", "ISRG", "JBHT", "KLAC",
                          "LRCX", "LBTYA", "MAR", "MAT", "MXIM", "MCHP", "MU", "MDLZ", "MSFT", "MNST", "MYL", "NFLX",
                          "NVDA", "ORLY", "PCAR", "PAYX", "QCOM", "REGN", "ROST", "STX", "SIRI", "SWKS", "SBUX", "SYMC",
                          "TSLA", "TXN", "TSCO", "TMUS", "FOX", "ULTA", "VRSK", "VRTX", "VIAB", "VOD", "WBA", "WDC",
                          "WYNN", "XLNX"]  # "PCLN",

        # Apply the model
        self.y_pred = model.predict(self.X_score)
        # Rescale to original values
        Scalers = glob.glob(self.root+"/saved_models_pipe/*Scaler*.pkl")
        RecentScaler = Scalers[-1]
        scaler = joblib.load(RecentScaler)
        self.y_pred_rescaled = scaler.inverse_transform(self.y_pred)
        self.X_score_rescaled = scaler.inverse_transform(self.X_score[0])
        #self.X_score[0, 1, :]
        # Calculate percentual difference between current value and prediction
        self.Delta = (self.y_pred_rescaled[0] - self.X_score_rescaled[1,:])/self.X_score_rescaled[1,:]
        # find top 5 companies in percentual increase
        top5indices = heapq.nlargest(5, range(len(self.Delta)), self.Delta.take)
        top5companies = [companiesAlpha[i] for i in top5indices]
        # Acquire New York Time
        self.now = datetime.datetime.now(timezone("America/New_York"))
        tomorrow = self.now + datetime.timedelta(days=1)
        tomorrow = tomorrow.strftime("%y%m%d")
        # Generate Headline Message
        SlackMsg = "Top 5 predictions for " + tomorrow +" :"
        # read in slack token
        with open(self.root+"/Meta/slacktoken.txt", "r") as myfile:
            token = myfile.readlines()
        # Post Headline Message
        sc = SlackClient(token)
        sc.api_call(
            "chat.postMessage",
            channel="predictions",
            text=SlackMsg
        )
        # Post predicted increase for top5 companies
        for company in top5companies:
            SlackMsg = "- " + company + " (+" + str(100*self.Delta[companiesAlpha.index(company)]) + " %)"
            sc = SlackClient(token)
            sc.api_call(
                "chat.postMessage",
                channel="predictions",
                text=SlackMsg
            )

        with open(self.output()["y_pred"].path, 'wb') as save_file:
            pickle.dump(self.y_pred, save_file)

    def output(self):
        self.now = datetime.datetime.now(timezone("America/New_York"))
        return {
        "X_score": self.input()["X_score"],
        "RawData": self.input()["RawData"],
        "y_pred": luigi.LocalTarget(self.root+'/Predictions/y_pred' +
             "_" + str(self.now.year) + str(self.now.month) + str(self.now.day) +'.pickle')
    }

###############################################
# This Task removes temporary and input files
class CleanUp(luigi.Task):
    PredictionTimepoints = luigi.Parameter()
    root = luigi.Parameter()
    def requires(self):
        return ScoreModel(self.PredictionTimepoints,self.root)

    def run(self):
        print("Nothing to clean")
        # Delete Files from input folder
        #os.remove(self.input()["X_score"].path)
        #os.remove(self.input()["NoNaNs"].path)

    def output(self):
        return {"y_pred": self.input()["y_pred"]}

class StartScoringPipe(luigi.WrapperTask):
    PredictionTimepoints = luigi.Parameter()
    root = "C:/Users/Fabian/Documents/FinancialForecasting"
    def requires(self):
        return CleanUp(self.PredictionTimepoints,self.root)


if __name__ == '__main__':
    ##############################
   # OPTIONAL for Slackbot => sends notification to slack
    with open("C:/Users/Fabian/Documents/FinancialForecasting/Meta/slacktoken.txt", "r") as myfile:
        token = myfile.readlines()
    slacker = SlackBot(token=token,
                       channels=['pipenews', '@FM Hecht'], events = ["SUCCESS", "FAILURE"])
    with notify(slacker):
        luigi.run()
   # luigi.run()