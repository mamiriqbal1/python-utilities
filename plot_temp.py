import numpy as np
import matplotlib.pyplot as plt


digit = "4"
algo1 = "XCS-IMG"
run1 = "01"
max_pop1 = 14000
algo2 = "XCS-IMG"
run2 = "18"
max_pop2 = 14000




test_performance1 = np.loadtxt("../remote/output/" + algo1 +"/output-" + digit + "-digit/" + digit + "-digits-" + run1 + "/test_performance.txt")
test_performance2 = np.loadtxt("../remote/output/"+algo2+"/output-"+digit+"-digit/"+digit+"-digits-"+run2+"/test_performance.txt")
title = algo1 + "_" + digit + "_" + run1 + " vs " + algo2 + "_" + digit + "_" + run2


plt.plot(test_performance1[:, 0], test_performance1[:, 1], label=algo1 + ' ' + run1 + ' Training')
plt.plot(test_performance2[:, 0], test_performance2[:, 1], label=algo2 + ' ' + run2 + ' Training')
plt.plot(test_performance1[:, 0], test_performance1[:, 4], label=algo1 + ' ' + run1 + ' Validation')
plt.plot(test_performance2[:, 0], test_performance2[:, 4], label=algo2 + ' ' + run2 + ' Validation')
plt.title('Error ' + title)
plt.legend(loc='lower right')
plt.savefig('plots/' + title + ' Error' + '.png')
plt.show()


plt.plot(test_performance2[:, 0], test_performance1[:, 6]/max_pop1, label=algo1 + ' ' + run1 + ' Population')
plt.plot(test_performance2[:, 0], test_performance2[:, 6]/max_pop2, label=algo2 + ' ' + run2 + ' Population')
plt.title('Population Size ' + title)
plt.legend(loc='lower left')
plt.savefig('plots/' + title + ' Population Size' + '.png')
plt.show()


print('done')
