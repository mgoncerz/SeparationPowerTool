import math
import numpy
import matplotlib.pyplot as plt
import seaborn


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