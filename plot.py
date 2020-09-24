import numpy as np
import matplotlib.pyplot as plt


digit = "2"
algo1 = "RXCSi"
run1 = "14"
algo2 = "RXCSi"
run2 = "19"
# algo2 = "XCSRCFC"
# run2 = "01"

if digit == "2":
    max_pop = 2000
if digit == "4":
    max_pop = 4000
if digit == "10":
    max_pop = 10000




# training_performance1 = np.loadtxt('../remote/output/"+algo1+"/output-"+digit+"-digit/"+digit+"-digits-01/training_performance1.txt')
test_performance1 = np.loadtxt("../remote/output/" + algo1 +"/output-" + digit + "-digit/" + digit + "-digits-" + run1 + "/test_performance.txt")

# training_performance2 = np.loadtxt('../remote/output/"+algo2+"/output-"+digit2+"-digit/"+digit2+"-digits-01/training_performance1.txt')
test_performance2 = np.loadtxt("../remote/output/"+algo2+"/output-"+digit+"-digit/"+digit+"-digits-"+run2+"/test_performance.txt")
title = algo1 + "_" + digit + "_" + run1 + " vs " + algo2 + "_" + digit + "_" + run2


plt.plot(test_performance1[:, 0], test_performance1[:, 1], label=algo1 + ' ' + run1 + ' Training')
plt.plot(test_performance2[:, 0], test_performance2[:, 1], label=algo2 + ' ' + run2 + ' Training')
plt.title('Training Accuracy ' + title)
plt.legend(loc='lower right')
plt.savefig('plots/' + title + ' Training Accuracy' + '.png')
plt.show()


plt.plot(test_performance1[:, 0], test_performance1[:, 4], label=algo1 + ' ' + run1 + ' Validation')
plt.plot(test_performance2[:, 0], test_performance2[:, 4], label=algo2 + ' ' + run2 + ' Validation')
plt.title('Validation Accuracy ' + title)
plt.legend(loc='lower right')
plt.savefig('plots/' + title + ' Validation Accuracy' + '.png')
plt.show()

plt.plot(test_performance2[:, 0], test_performance1[:, 6], label=algo1 + ' ' + run1 + ' Population')
plt.plot(test_performance2[:, 0], test_performance2[:, 6], label=algo2 + ' ' + run2 + ' Population')
plt.title('Population Size ' + title)
plt.legend(loc='lower left')
plt.savefig('plots/' + title + ' Population Size' + '.png')
plt.show()

plt.plot(test_performance1[:, 0], test_performance1[:, 2], label=algo1 + ' ' + run1 + ' Training')
plt.plot(test_performance2[:, 0], test_performance2[:, 2], label=algo2 + ' ' + run2 + ' Training')
plt.title('Training Error ' + title)
plt.legend(loc='upper right')
plt.savefig('plots/' + title + ' Training Error' + '.png')
plt.show()

plt.plot(test_performance1[:, 0], test_performance1[:, 5], label=algo1 + ' ' + run1 + ' Validation')
plt.plot(test_performance2[:, 0], test_performance2[:, 5], label=algo2 + ' ' + run2 + ' Validation')
plt.title('Validation Error ' + title)
plt.legend(loc='upper right')
plt.savefig('plots/' + title + ' Validation Error' + '.png')
plt.show()


print('done')
