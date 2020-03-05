import math
import numpy
import matplotlib.pyplot as plt
import seaborn

import pandas

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, auc

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K


def generateVariableSets(dataframe1, dataframe2, pearson_thresholds = -1, cutoff_values = -1, variables = -1, verbose = False):

    #print()
    if pearson_thresholds == -1:
        pearson_thresholds = set(numpy.around(numpy.arange(0.1, 1.0, 0.05), decimals=2))
    else:
        pearson_thresholds = set(pearson_thresholds)

    if cutoff_values == -1:
        cutoff_values = set(numpy.arange(2, 60, 2))
    else:
        cutoff_values = set(cutoff_values)

    if variables == -1:
        variables = set(dataframe1.columns).intersection(set(dataframe2.columns))
    else:
        variables = set(variables) 


    if verbose:
        print()
        print('Generating variable sets using:')
        print(['Variables:', variables])
        print(['Pearson coefficient thresholds:', pearson_thresholds])
        print(['SP cutoff values:', cutoff_values])
        print()
        print()


    data1 = dict.fromkeys(variables, None)
    data2 = dict.fromkeys(variables, None)
    separation = []

    for variable in variables:

        data1[variable] = dataframe1[variable].to_numpy(dtype=float, copy=True)
        data2[variable] = dataframe2[variable].to_numpy(dtype=float, copy=True)

        data1[variable] = (data1[variable] - data1[variable].mean()) / data1[variable].std()
        data2[variable] = (data2[variable] - data2[variable].mean()) / data2[variable].std()

        
        minimum = min(data1[variable].min(), data2[variable].min())
        maximum = max(data1[variable].max(), data2[variable].max())

        bincontent1, _ = numpy.histogram(data1[variable], 1000, range=[minimum,maximum])
        bincontent2, _ = numpy.histogram(data2[variable], 1000, range=[minimum,maximum])

        separation.append(round(100*0.5*sum(map(abs, bincontent1 / sum(bincontent1) - bincontent2 / sum(bincontent2))), 2))


    separation, variables = zip(*sorted(zip(separation, variables), key=lambda x: x[0], reverse=True))

    correlation_matrix1 = numpy.corrcoef([data1[var] for var in variables])
    correlation_matrix2 = numpy.corrcoef([data2[var] for var in variables])



    if verbose:

        print('All variables sorted by separation:')
        for sep, var in zip(separation, variables):
            print([sep, var])
        print()
        print()


        f, ax = plt.subplots(figsize=(20,20))
        nrows = math.floor(math.sqrt(len(variables)))
        ncols = math.ceil(len(variables) / nrows)

        for index, variable in enumerate(variables):
            f.add_subplot(nrows, ncols, index + 1)
            seaborn.distplot(data1[variable], kde=False, norm_hist = True).set_title(variable)
            seaborn.distplot(data2[variable], kde=False, norm_hist = True).set_title(variable)


        mask = numpy.triu(numpy.ones_like(correlation_matrix1, dtype=numpy.bool))
        cmap = seaborn.diverging_palette(10, 220, as_cmap=True)

        f, ax = plt.subplots(figsize=(20, 20))
        seaborn.heatmap(correlation_matrix1, annot=correlation_matrix1, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=variables, yticklabels=variables).set_title('dataset 1')

        f, ax = plt.subplots(figsize=(20, 20))
        seaborn.heatmap(correlation_matrix2, annot=correlation_matrix2, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=variables, yticklabels=variables).set_title('dataset 2')

        plt.show()



    output = {
        'variable names' : variables,
        'separation values' : separation,
        'variable sets' : [],
        'custom sets' : []
    }

    output['variable sets'].append({'pearson threshold' : 1.0, 'cutoff value' : 0, 'number of variables' : len(variables), 'mask' : numpy.ones_like(variables, dtype=bool)})


    for pearson in pearson_thresholds:
        for cutoff in cutoff_values:

            mask = numpy.ones_like(variables, dtype=bool)

            for lower_separation_index in reversed(range(len(separation))):

                if separation[lower_separation_index] <= cutoff:
                    mask[lower_separation_index] = False
                    continue

                for higher_separation_index in range(lower_separation_index):

                    if math.fabs(correlation_matrix1[lower_separation_index][higher_separation_index]) >= pearson or math.fabs(correlation_matrix2[lower_separation_index][higher_separation_index]) >= pearson: 
                        mask[lower_separation_index] = False
                        break
            


            duplicate = False
            for variable_set in output['variable sets']:
                if(numpy.all(numpy.equal(variable_set['mask'], mask))):
                    variable_set['pearson threshold'] = min(variable_set['pearson threshold'], pearson)
                    variable_set['cutoff value'] = max(variable_set['cutoff value'], cutoff)
                    duplicate = True
                    break

            if not duplicate:
                output['variable sets'].append({'pearson threshold' : pearson, 'cutoff value' : cutoff, 'number of variables' : numpy.sum(mask), 'mask' : mask})


    output['variable sets'].sort(key=lambda x: x['number of variables'], reverse=True)

    if verbose:
        print('Generated ' + str(len(output['variable sets'])) + ' variable sets.')
        print(output)

    return output


def printVariableSets(variable_sets):

    print('Full list of variables:')
    for sep, var in zip(variable_sets['separation values'], variable_sets['variable names']):
        print([sep, var])

    print()
    print()

    print('Variable sets (' + str(len(variable_sets['variable sets'])) + '):')
    print()
    for varset in variable_sets['variable sets']:
        print(['pearson threshold', varset['pearson threshold']])
        print(['cutoff value', varset['cutoff value']])
        print(['number of variables', varset['number of variables']])
        print(['variables:'])
        for index, mask_value in enumerate(varset['mask']):
            if mask_value:
                print([variable_sets['separation values'][index], variable_sets['variable names'][index]])
        print()
        print()
        print()

    if variable_sets['custom sets']:
        print('Custom sets (' + str(len(variable_sets['custom sets'])) + '):')
        print()
        for varset in variable_sets['custom sets']:
            print(['name', varset['name']])
            print(['number of variables', varset['number of variables']])
            print(['variables:'])
            for index, mask_value in enumerate(varset['mask']):
                if mask_value:
                    print([variable_sets['separation values'][index], variable_sets['variable names'][index]])
            print()
            print()
            print()

def addCustomVariableSet(variable_sets, name, variables):

    mask = numpy.zeros_like(variable_sets['variable names'], dtype=bool)

    for index, variable in enumerate(variable_sets['variable names']):

        if variable in variables:
            mask[index] = True

    variable_sets['custom sets'].append({'name' : name, 'number of variables' : numpy.sum(mask), 'mask' : mask})


def testVariableSets(dataframe1, dataframe2, variable_sets, verbose=True):

    X1 = pandas.DataFrame(data=dataframe1)
    Y1 = pandas.DataFrame(data=numpy.ones(X1.shape[0], dtype=bool))

    X2 = pandas.DataFrame(data=dataframe2)
    Y2 = pandas.DataFrame(data=numpy.zeros(X2.shape[0], dtype=bool))

    X = X1.append(X2, ignore_index=True)
    Y = Y1.append(Y2, ignore_index=True)

    if verbose:
        print(X)
        print(Y)


    def testSet(X, Y, varset, verbose):

        metrics = {}

        X_train, X_test, Y_train, Y_test = train_test_split(X.loc[:, varset['mask']], Y, test_size=0.2, shuffle=True)

        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim = varset['number of variables']))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        callback = [EarlyStopping(monitor='val_loss', patience=3)]

        history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs=50, batch_size=256, callbacks=callback, verbose=verbose)
  
        metrics['accuracy'] = history.history['val_accuracy'][-5]
        metrics['loss'] = history.history['val_loss'][-5]

        predicted = model.predict(X_test)

        metrics['roc'] = roc_curve(Y_test, predicted)
        metrics['roc integral'] = auc(metrics['roc'][0], metrics['roc'][1])

        predicted_class = predicted > 0.5

        cfm = confusion_matrix(Y_test, predicted_class, normalize = 'all')
        metrics['confusion matrix'] = cfm

        #true positives + true negatives normalized to event number
        metrics['classification accuracy'] = cfm[1][1] + cfm[0][0]

        #metrics['model purity'] = ()

        varset['metrics'] = metrics

        K.clear_session()

    for varset in variable_sets['variable sets']:
        testSet(X, Y, varset, verbose)

    for varset in variable_sets['custom sets']:
        testSet(X, Y, varset, verbose)

