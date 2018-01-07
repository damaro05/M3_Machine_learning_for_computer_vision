import gzip
import cPickle

filename = 'grid_2018-01-07 18-58-56.416684.pklz'

path='grids'
with gzip.open(path+'/'+filename) as f:
    grid = cPickle.load(f)

for i in range(len(grid.cv_results_['mean_test_score'])):
    print str(grid.cv_results_['params'][i]) + ' -> ' + str(100 * grid.cv_results_['mean_test_score'][i]) + '%'

print 'Best score: ' + str(grid.best_score_)
print 'Best params: ' + str(grid.best_params_)

'''
BoVW:
grid_2018-01-03 16-41-19.817478.pklz

Spatial Pyramids:
grid_2018-01-03 13-13-59.576644.pklz
grid_2018-01-03 16-14-52.871637.pklz
grid_2018-01-03 16-35-32.620042.pklz

Intersection Kernel + Spatial Pyramids:
grid_2018-01-07 18-58-56.416684.pklz
'''