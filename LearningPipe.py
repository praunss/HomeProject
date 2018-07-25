import luigi
from luigi_slack import SlackBot, notify
import pandas as pd
import numpy as np
from numpy import newaxis
import datetime
import pickle
import time
import os
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint
import quandl
from alpha_vantage.timeseries import TimeSeries

# Run with : python LearningPipe.py --scheduler-host localhost StartLearningPipe --PredictionTimepoints 2 --Epochs 2000
#####################
# NOTE FH : TO FIX:
#######################
class GetData(luigi.Task):
    root = luigi.Parameter()

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

        # Generate todays date autmatically
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now()-Delta

        print("Getting alphavantage data...")

        with open(self.root + "/Meta/alphavantage.txt", "r") as myfile:
            alphatoken = myfile.readlines()
        ts = TimeSeries(key=alphatoken, output_format='pandas', retries=5)
        self.mydata = self.getalphadata(companiesAlpha, ts)

        tickerend = time.time()
        print("Data downloaded in {} s".format((tickerend - tickerstart)))
        print("No Companies: {}".format(self.mydata.shape[1]))



        # save as csv with current date
        with self.output().open('w') as outfile:
                self.mydata.to_csv(outfile)

    def getalphadata(self, companiesAlpha, ts):
        finaldatacomp, metadata = ts.get_daily_adjusted(symbol='ATVI',outputsize="full")
        finaldata = pd.DataFrame(finaldatacomp["5. adjusted close"])
        finaldata.columns = ["ATVI"]
        i = 0
        n = 0
        for company in companiesAlpha:
            i += 1
            n += 1
            data, metadata = ts.get_daily_adjusted(symbol=company,outputsize="full")
            datatemp = pd.DataFrame(data["5. adjusted close"])
            datatemp.columns = [company]
            finaldata = finaldata.join(datatemp)
            if i == 1:
                time.sleep(30)
                i = 0
        return finaldata


    def output(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta

        return luigi.LocalTarget(self.root + "/Data/Data-"+str(self.now.year)+str(self.now.month)+str(self.now.day)+".csv")


class CheckInputforNans(luigi.Task):
    root = luigi.Parameter()

    def requires(self):
        return GetData(self.root)

    def run(self):
        # load from target
        self.mydata = pd.read_csv(self.input().path, parse_dates = True, index_col = "date")
        # Optional: Slice data

        self.mydata = self.mydata[datetime.datetime(2012, 1, 1):self.now]

        # Print number of NaNs, throw exception if more than 5
        NumberofRowNans = len(self.mydata[self.mydata.isnull().any(axis=1)])
        if NumberofRowNans > 5:
            raise ValueError("More than 5 NaNs in downloaded data")
        # If fewer Nans, fill with bfill method
        self.mydata = self.mydata.fillna(method="bfill")

        NumberofRowNansAC = len(self.mydata[self.mydata.isnull().any(axis=1)])
        if NumberofRowNansAC > 0:
            raise ValueError("Nans in data, cannot be filled artificially")
        with self.output()["NoNaNs"].open('w') as outfile:
            self.mydata.to_csv(outfile)

    def output(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta

        return {"NoNaNs": luigi.LocalTarget(self.root + "/Temp/Data-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + "_NoNans.csv"),
                "RawData" : self.input() }


class PrepareDataForANN(luigi.Task):
    PredictionTimepoints = luigi.Parameter()
    root = luigi.Parameter()

    def requires(self):
        return CheckInputforNans(self.root)

    def run(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta

        self.PredictionTimepoints = int(self.PredictionTimepoints)
        # Data preparation
        self.mydata = pd.read_csv(self.input()["NoNaNs"].path, index_col=0, parse_dates=True)

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
            joblib.dump(scaler, self.root + '/saved_models_pipe/Scaler_' +  str(self.now.year) + str(self.now.month) + str(self.now.day) + '.pkl')
        # FIRST define target day, THEN extract image of past data

        for i in range(MaxPoints):
            # START CREATE OUTPUT VECTORS

            PredictTemp = mydataPP.iloc[toPredictIndex]

            arrayPredtemp = np.array(PredictTemp, np.float32)[newaxis, :]
            TargetCollection = np.append(TargetCollection, arrayPredtemp, axis=0)

            # START CREATE INPUT VECTORS (IMAGES)
            end = toPredictIndex  # e.g. first = 250
            start = end - self.PredictionTimepoints  # e.g. first = 0

            AdjCloseTemp = mydataPP.iloc[start: end]  # e.g. 0 - 249 inclusive, as last index is not sliced

            AdjCloseTemp_Array = AdjCloseTemp.values

            arrayAdjClosedTemp = np.array(AdjCloseTemp_Array, np.float32)[newaxis, :, :]
            DataCollection = np.append(DataCollection, arrayAdjClosedTemp, axis=0)

            # END CREATE IMAGES

            toPredictIndex += 1

        DataCollection = DataCollection[1:DataCollection.shape[0], :, :]
        TargetCollection = TargetCollection[1:TargetCollection.shape[0], :]

        print("#############################")
        print("Data Shape prepared for ANN:")
        print(DataCollection.shape)
        print("#############################")

        # Define number of Training samples (85 %), Validation (15%)
        TrainingSamples = int(MaxPoints * 0.85)

        self.X_train = np.copy(DataCollection[:TrainingSamples, :])
        self.y_train = np.copy(TargetCollection[:TrainingSamples, :])
        self.X_valid = np.copy(DataCollection[TrainingSamples - 1:, :])
        self.y_valid = np.copy(TargetCollection[TrainingSamples - 1:, :])


        print("#############################")
        print("Trainingsamples: {}".format(self.X_train.shape))
        print("#############################")

        with open(self.output()["X_train"].path, 'wb') as save_file:
            pickle.dump(self.X_train, save_file)
        with open(self.output()["y_train"].path, 'wb') as save_file:
            pickle.dump(self.y_train, save_file)
        with open(self.output()["X_valid"].path, 'wb') as save_file:
            pickle.dump(self.X_valid, save_file)
        with open(self.output()["y_valid"].path, 'wb') as save_file:
            pickle.dump(self.y_valid, save_file)


    def output(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta
        return {
                 "X_train": luigi.LocalTarget(self.root + "/Temp/X_train-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "y_train": luigi.LocalTarget(self.root + "/Temp/y_train-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "X_valid": luigi.LocalTarget(self.root + "/Temp/X_valid-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "y_valid": luigi.LocalTarget(self.root + "/Temp/y_valid-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "RawData": self.input()["RawData"],
                 "NoNaNs": self.input()["NoNaNs"]
                }



class LearnModel(luigi.Task):
    PredictionTimepoints = luigi.Parameter()
    Epochs = luigi.Parameter()
    root  = luigi.Parameter()

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
        return PrepareDataForANN(self.PredictionTimepoints, self.root)

    def run(self):

        # Load required (prepared) data
        pickle_in = open(self.input()["X_train"].path, "rb")
        self.X_train = pickle.load(pickle_in)
        pickle_in = open(self.input()["y_train"].path, "rb")
        self.y_train = pickle.load(pickle_in)
        pickle_in = open(self.input()["X_valid"].path, "rb")
        self.X_valid = pickle.load(pickle_in)
        pickle_in = open(self.input()["y_valid"].path, "rb")
        self.y_valid = pickle.load(pickle_in)

        self.NumberofCompanies = self.X_train.shape[2]
        epochs = int(self.Epochs)

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # build estimator
        estimator = KerasRegressor(build_fn=self.MLP_B2, epochs=epochs, batch_size=self.X_train.shape[0], verbose=1)

        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta

        # Enter checkpoint filename here
        checkpointer = ModelCheckpoint(filepath=self.root + '/saved_models_pipe/weights.best.pipe_MLPtype2_B2_Timepoints' +
            self.PredictionTimepoints + "_" + str(self.now.year) + str(self.now.month) + str(self.now.day) +'.hdf5',
                                       verbose=1, save_best_only=True)

        estimator.fit(self.X_train, self.y_train, validation_data=(self.X_valid, self.y_valid), callbacks=[checkpointer])
    def output(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta
        return {
        "X_train": self.input()["X_train"],
        "y_train": self.input()["y_train"],
        "X_valid": self.input()["X_valid"],
        "y_valid": self.input()["y_valid"],
        "RawData": self.input()["RawData"],
        "NoNaNs": self.input()["NoNaNs"],
        "Model": luigi.LocalTarget(self.root + '/saved_models_pipe/weights.best.pipe_MLPtype2_B2_Timepoints' +
            self.PredictionTimepoints + "_" + str(self.now.year) + str(self.now.month) + str(self.now.day) +'.hdf5')
    }

###############################################
# This Task removes temporary and input files
class CleanUp(luigi.Task):
    PredictionTimepoints = luigi.Parameter()
    Epochs = luigi.Parameter()
    root = luigi.Parameter()

    def requires(self):
        return LearnModel(self.PredictionTimepoints, self.Epochs, self.root)

    def run(self):

        # Delete Files from input folder
        os.remove(self.input()["X_train"].path)
        os.remove(self.input()["y_train"].path)
        os.remove(self.input()["X_valid"].path)
        os.remove(self.input()["y_valid"].path)
        os.remove(self.input()["NoNaNs"].path)


    def output(self):
        return {"Model": self.input()["Model"]}

class StartLearningPipe(luigi.WrapperTask):
    PredictionTimepoints = luigi.Parameter()
    Epochs = luigi.Parameter()
    root = "C:/Users/Fabian/Documents/FinancialForecasting"
    def requires(self):
        return CleanUp(self.PredictionTimepoints, self.Epochs, self.root)


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