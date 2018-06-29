import luigi

import pandas as pd
import numpy as np
from numpy import newaxis
import datetime
import pickle
import time

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint
import quandl

# Run with : python LearningPipe.py --scheduler-host localhost LearnModel --PredictionTimepoints 2
#####################
# NOTE FH : TO FIX:
#######################
class GetData(luigi.Task):
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
        self.now = datetime.datetime.now()-Delta
        RequestDate = str(self.now.year) + "-"
        if int(self.now.month) < 10:
            RequestDate += str(0)
        RequestDate += str(self.now.month) + "-"
        if int(self.now.day) < 10:
            RequestDate += str(0)
        RequestDate += str(self.now.day)
        print("Getting quandl data...")
        self.mydata = quandl.get(companies, start_date="2018-01-01", end_date=RequestDate)
        tickerend = time.time()
        print("Data downloaded in {} s".format((tickerend - tickerstart)))

        # save as csv with current date
        with self.output().open('w') as outfile:
                self.mydata.to_csv(outfile)

    def output(self):
        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta

        return luigi.LocalTarget("Data/Data-"+str(self.now.year)+str(self.now.month)+str(self.now.day)+".csv")


class CheckInputforNans(luigi.Task):
    def requires(self):
        return GetData()

    def run(self):
        # load from target
        self.mydata = pd.read_csv(self.input().path)

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

        return {"NoNaNs": luigi.LocalTarget("Data/Data-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + "_NoNans.csv"),
                "RawData" : self.input() }


class PrepareDataForANN(luigi.Task):
    PredictionTimepoints = luigi.Parameter()

    def requires(self):
        return CheckInputforNans()

    def run(self):
        self.PredictionTimepoints = int(self.PredictionTimepoints)
        # Data preparation
        self.mydata = pd.read_csv(self.input()["NoNaNs"].path)
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
        ValidationSamples = int(MaxPoints - TrainingSamples)
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
                 "X_train": luigi.LocalTarget("Temp/X_train-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "y_train": luigi.LocalTarget("Temp/y_train-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "X_valid": luigi.LocalTarget("Temp/X_valid-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "y_valid": luigi.LocalTarget("Temp/y_valid-" + str(self.now.year) + str(self.now.month) + str(self.now.day) + ".pickle"),
                 "RawData": self.input()["RawData"],
                 "NoNaNs": self.input()["NoNaNs"]
                }

class LearnModel(luigi.WrapperTask):
    PredictionTimepoints = luigi.Parameter()

    def requires(self):
        return PrepareDataForANN(self.PredictionTimepoints)


class LearnModel2(luigi.Task):
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
        pickle_in = open(self.input()["X_train"].path, "rb")
        self.X_train = pickle.load(pickle_in)
        pickle_in = open(self.input()["y_train"].path, "rb")
        self.y_train = pickle.load(pickle_in)
        pickle_in = open(self.input()["X_valid"].path, "rb")
        self.X_valid = pickle.load(pickle_in)
        pickle_in = open(self.input()["y_valid"].path, "rb")
        self.y_valid = pickle.load(pickle_in)

        print(type(self.X_train))
        self.NumberofCompanies = self.X_train.shape[2]
        epochs = 5

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # build estimator
        estimator = KerasRegressor(build_fn=self.MLP_B2, epochs=epochs, batch_size=self.X_train.shape[0], verbose=1)

        Delta = datetime.timedelta(days=1)
        self.now = datetime.datetime.now() - Delta

        # Enter checkpoint filename here
        checkpointer = ModelCheckpoint(filepath='saved_models_pipe/weights.best.pipe_MLPtype2_B2_Timepoints' +
            self.PredictionTimepoints + "_" + str(self.now.year) + str(self.now.month) + str(self.now.day) +'.hdf5',
                                       verbose=1, save_best_only=True)

        estimator.fit(self.X_train, self.y_train, validation_data=(self.X_valid, self.y_valid), callbacks=[checkpointer])
    def output(self):
        return None

if __name__ == '__main__':
                    ###############################
                    # OPTIONAL for Slackbot => sends notification to slack
                    # slacker = SlackBot(token='xoxp-379158436963-378561401968-379040477684-de68e92cda6785ba0b27bb8d9e1c66d5',
                    #                    channels=['anomaly-pipe', '@Fabian Hecht'], events = ["SUCCESS", "FAILURE"])
                    # with notify(slacker):
                    #     luigi.run()
    luigi.run()