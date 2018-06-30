import luigi
from luigi_slack import SlackBot, notify
import pandas as pd
import numpy as np
from numpy import newaxis
import datetime
import pickle
import time
import os
import glob

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import quandl

# Run with : python ScoringPipe.py --scheduler-host localhost StartScoringPipe --PredictionTimepoints 2
#####################
# NOTE FH : START AT preparedataann -> must bring input to correct shape for socring
#######################
class GetData(luigi.Task):
    PredictionTimepoints = luigi.Parameter()

    def requires(self):
        None
    def run(self):

        quandl.ApiConfig.api_key = "U5cJSsnv4Ad7UUnHNGu8"
        companies = ["WIKI/ATVI.11", "WIKI/ADBE.11", "WIKI/AKAM.11", "WIKI/ALXN.11", "WIKI/GOOGL.11", "WIKI/AMZN.11",
                     "WIKI/AAL.11", "WIKI/AMGN.11", "WIKI/ADI.11", "WIKI/AAPL.11", "WIKI/AMAT.11", "WIKI/ADSK.11",
                     "WIKI/ADP.11", "WIKI/BIDU.11", "WIKI/BIIB.11", "WIKI/BMRN.11", "WIKI/CA.11", "WIKI/CELG.11",
                     "WIKI/CERN.11", "WIKI/CHKP.11", "WIKI/CTAS.11", "WIKI/CSCO.11", "WIKI/CTXS.11", "WIKI/CTSH.11",
                     "WIKI/CMCSA.11", "WIKI/COST.11", "WIKI/CSX.11", "WIKI/XRAY.11", "WIKI/DISCA.11", "WIKI/DISH.11",
                     "WIKI/DLTR.11", "WIKI/EBAY.11", "WIKI/EA.11", "WIKI/EXPE.11", "WIKI/ESRX.11", "WIKI/FAST.11",
                     "WIKI/FISV.11", "WIKI/GILD.11", "WIKI/HAS.11", "WIKI/HSIC.11", "WIKI/HOLX.11", "WIKI/IDXX.11",
                     "WIKI/ILMN.11", "WIKI/INCY.11", "WIKI/INTC.11", "WIKI/INTU.11", "WIKI/ISRG.11", "WIKI/JBHT.11",
                     "WIKI/KLAC.11", "WIKI/LRCX.11", "WIKI/LBTYA.11", "WIKI/MAR.11", "WIKI/MAT.11", "WIKI/MXIM.11",
                     "WIKI/MCHP.11", "WIKI/MU.11", "WIKI/MDLZ.11", "WIKI/MSFT.11", "WIKI/MNST.11", "WIKI/MYL.11",
                     "WIKI/NFLX.11", "WIKI/NVDA.11", "WIKI/ORLY.11", "WIKI/PCAR.11", "WIKI/PAYX.11", "WIKI/QCOM.11",
                     "WIKI/REGN.11", "WIKI/ROST.11", "WIKI/STX.11", "WIKI/SIRI.11", "WIKI/SWKS.11", "WIKI/SBUX.11",
                     "WIKI/SYMC.11", "WIKI/TSLA.11", "WIKI/TXN.11", "WIKI/TSCO.11", "WIKI/TMUS.11", "WIKI/FOX.11",
                     "WIKI/ULTA.11", "WIKI/VRSK.11", "WIKI/VRTX.11", "WIKI/VIAB.11", "WIKI/VOD.11", "WIKI/WBA.11",
                     "WIKI/WDC.11", "WIKI/WYNN.11", "WIKI/XLNX.11"]  # "WIKI/PCLN.11",
        # Download via API
        tickerstart = time.time()

        # Generate todays date autmatically
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now()#-Delta
        PredictionDelta = datetime.timedelta(days=int(self.PredictionTimepoints)+5)
        self.past = datetime.datetime.now()-PredictionDelta

        RequestDate = str(self.now.year) + "-"
        if int(self.now.month) < 10:
            RequestDate += str(0)
        RequestDate += str(self.now.month) + "-"
        if int(self.now.day) < 10:
            RequestDate += str(0)
        RequestDate += str(self.now.day)
        print("RequestDate {}".format(RequestDate))
        RequestDateStart = str(self.past.year) + "-"
        if int(self.past.month) < 10:
            RequestDateStart += str(0)
        RequestDateStart += str(self.past.month) + "-"
        if int(self.past.day) < 10:
            RequestDateStart += str(0)
        RequestDateStart += str(self.past.day)
        print("RequestDatestart {}".format(RequestDateStart))

        print("Getting quandl data...")
        self.mydata = quandl.get(companies, start_date=RequestDateStart, end_date=RequestDate)
        tickerend = time.time()
        print("Data downloaded in {} s".format((tickerend - tickerstart)))
        print("No Companies: {}".format(self.mydata.shape[1]))

        print(self.mydata.head())

        # save as csv with current date
        with self.output().open('w') as outfile:
                self.mydata.to_csv(outfile)

    def output(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta

        return luigi.LocalTarget("Data/ScoringData-"+str(self.now.year)+str(self.now.month)+str(self.now.day)+".csv")


class CheckInputforNans(luigi.Task):
    PredictionTimepoints = luigi.Parameter()

    def requires(self):
        return GetData(self.PredictionTimepoints)

    def run(self):
        # load from target
        self.mydata = pd.read_csv(self.input().path)

        # Print number of NaNs, throw exception if more than 5
        NumberofRowNans = len(self.mydata[self.mydata.isnull().any(axis=1)])
        if NumberofRowNans > 0:
            raise ValueError("NaNs detected in scoring data")
        # If fewer Nans, fill with bfill method

        with self.output()["NoNaNs"].open('w') as outfile:
            self.mydata.to_csv(outfile)

    def output(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta

        return {"NoNaNs": luigi.LocalTarget("Temp/ScoringData-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + "_NoNans.csv"),
                "RawData" : self.input() }


class PrepareDataForANN(luigi.Task):
    PredictionTimepoints = luigi.Parameter()

    def requires(self):
        return CheckInputforNans(self.PredictionTimepoints)

    def run(self):
        self.PredictionTimepoints = int(self.PredictionTimepoints)
        # Data preparation
        self.mydata = pd.read_csv(self.input()["NoNaNs"].path, index_col=0, parse_dates=True)
        self.mydata[['Date']] = self.mydata[['Date']].apply(pd.to_datetime, errors='ignore')
        self.mydata = self.mydata.set_index(self.mydata["Date"])
        self.mydata = self.mydata.drop("Date", axis=1)

        NumberofCompanies = self.mydata.shape[1]

        FirstIndex = self.PredictionTimepoints
        MaxPoints = self.mydata.shape[0] - FirstIndex

        MLP = True  # False => CNN
        normalization = True

        # create np array for Data Collection
        DataCollection = np.empty([1, self.PredictionTimepoints, NumberofCompanies])
        # Create np array for Target Collection
        TargetCollection = np.empty([1, NumberofCompanies])

        # Create copy of data frame for handling
        mydataPP = self.mydata.copy(deep=True)
        # Set StartIndex to FirstIndex
        toPredictIndex = FirstIndex

        # Normalization if required
        if (normalization == True):
            mydataNP = self.mydata.values
            scaler = MinMaxScaler()
            mydataNormalizedNP = scaler.fit_transform(mydataNP)
            mydataPP = pd.DataFrame(mydataNormalizedNP)


        # START CREATE INPUT VECTORS (IMAGES)

        AdjCloseTemp = mydataPP.iloc[0 : self.PredictionTimepoints]  # e.g. 0 - 249 inclusive, as last index is not sliced

        AdjCloseTemp_Array = AdjCloseTemp.values

        arrayAdjClosedTemp = np.array(AdjCloseTemp_Array, np.float32)[newaxis, :, :]
        DataCollection = np.append(DataCollection, arrayAdjClosedTemp, axis=0)

        # END CREATE IMAGES
        DataCollection = DataCollection[1:DataCollection.shape[0], :, :]

        print("#############################")
        print("Data Shape prepared for ANN:")
        print(DataCollection.shape)
        print("#############################")

        print("#############################")
        print("Scoring Samples: {}".format(self.X_train.shape))
        print("#############################")

        with open(self.output()["X_score"].path, 'wb') as save_file:
            pickle.dump(self.X_train, save_file)


    def output(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta
        return {
                 "X_score": luigi.LocalTarget("Temp/X_score-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "RawData": self.input()["RawData"],
                 "NoNaNs": self.input()["NoNaNs"]
                }



class ScoreModel(luigi.Task):
    PredictionTimepoints = luigi.Parameter()

    def MLP_B2(self):
        model = Sequential()
        model.add(Flatten(input_shape=(int(self.PredictionTimepoints), self.NumberofCompanies)))

        # model.add(Dense(5000, activation='relu'))
        model.add(Dense(2000, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(87))
        model.compile(loss='mean_squared_error', optimizer="adamax", metrics=['mse'])

        return model

    def requires(self):
        return PrepareDataForANN(self.PredictionTimepoints)

    def run(self):
        # Load required (prepared) data
        pickle_in = open(self.input()["X_score"].path, "rb")
        self.X_score = pickle.load(pickle_in)
        self.NumberofCompanies = self.X_train.shape[2]

        # Check for newest model
        Models = glob.glob("saved_models_pipe/*MLPtype2_B2*.hdf5")
        RecentModel = Models[-1]

        model = self.MLP_B2()
        model.load_weights(RecentModel)

        self.y_pred = model.predict(self.X_score)

        print(y_pred)

        with open(self.output()["y_pred"].path, 'wb') as save_file:
            pickle.dump(self.y_pred, save_file)

    def output(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta
        return {
        "X_score": self.input()["X_score"],
        "RawData": self.input()["RawData"],
        "NoNaNs": self.input()["NoNaNs"],
        "y_pred": luigi.LocalTarget('Predictions/y_pred' +
             "_" + str(self.now.year) + str(self.now.month) + str(self.now.day) +'.pickle')
    }

###############################################
# This Task removes temporary and input files
class CleanUp(luigi.Task):
    PredictionTimepoints = luigi.Parameter()

    def requires(self):
        return ScoreModel(self.PredictionTimepoints)

    def run(self):

        # Delete Files from input folder
        os.remove(self.input()["X_score"].path)
        os.remove(self.input()["NoNaNs"].path)

    def output(self):
        return {"y_pred": self.input()["y_pred"]}

class StartScoringPipe(luigi.WrapperTask):
    PredictionTimepoints = luigi.Parameter()

    def requires(self):
        return CleanUp(self.PredictionTimepoints)


if __name__ == '__main__':
    ##############################
   # OPTIONAL for Slackbot => sends notification to slack
    with open("Meta/slacktoken.txt", "r") as myfile:
        token = myfile.readlines()
    slacker = SlackBot(token=token,
                       channels=['pipenews', '@FM Hecht'], events = ["SUCCESS", "FAILURE"])
    with notify(slacker):
        luigi.run()
   # luigi.run()