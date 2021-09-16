import numpy as np
import matplotlib.pyplot as plt


digit = "10"
algo1 = "RXCSi"
algo1_label = "XCS-IMG"
max_pop1 = 35000
algo2 = "XCSRCFC"
max_pop2 = 35000




# test_performance1 = np.loadtxt("../remote/output/" + algo1 +"/output-" + digit + "-digit/" + digit + "-digits-" + run1 + "/test_performance.txt")
# test_performance2 = np.loadtxt("../remote/output/"+algo2+"/output-"+digit+"-digit/"+digit+"-digits-"+run2+"/test_performance.txt")
# title = algo1 + "_" + digit + "_" + run1 + " vs " + algo2 + "_" + digit + "_" + run2

test_performance1 = np.loadtxt("../remote/output/" + algo1 +"/output-" + digit + "-digit/" + "summary/" + algo1 + "-" + digit + "-01-30-validation_performance_avg.txt")
test_performance1_sd = np.loadtxt("../remote/output/" + algo1 +"/output-" + digit + "-digit/" + "summary/" + algo1 + "-" + digit + "-01-30-validation_performance_sd.txt")
test_performance2 = np.loadtxt("../remote/output/" + algo2 + "/output-" +digit + "-digit/" + "summary/" + algo2 + "-" + digit + "-01-30-validation_performance_avg.txt")
test_performance2_sd = np.loadtxt("../remote/output/" + algo2 + "/output-" +digit + "-digit/" + "summary/" + algo2 + "-" + digit + "-01-30-validation_performance_sd.txt")

title = 'Average Training Accuracy for ' + digit + ' digits'
plt.errorbar(test_performance1[:, 0], test_performance1[:, 1], yerr=test_performance1_sd[:, 1], label=algo1_label, color='k', linestyle='solid')
plt.errorbar(test_performance2[:, 0], test_performance2[:, 1], yerr=test_performance2_sd[:, 1], label=algo2, color='0.5', linestyle='solid')
plt.title(title)
plt.legend(loc='lower right')
plt.ylim(0.8, 1)
plt.ylabel('Performance')
plt.xlabel('Instances')
plt.savefig('plots/' + title + '.png')
plt.show()

title = 'Average Validation Accuracy for ' + digit + ' digits'
plt.errorbar(test_performance1[:, 0], test_performance1[:, 4], yerr=test_performance1_sd[:, 4], label=algo1_label, color='k', linestyle='solid')
plt.errorbar(test_performance2[:, 0], test_performance2[:, 4], yerr=test_performance2_sd[:, 4], label=algo2, color='0.5', linestyle='solid')
plt.title(title)
plt.legend(loc='lower right')
plt.ylim(0.8, 1)
plt.ylabel('Performance')
plt.xlabel('Instances')
plt.savefig('plots/' + title + '.png')
plt.show()


# plt.plot(test_performance1[:, 0], test_performance1[:, 2], label=algo1 + ' ' + run1 + ' Training')
# plt.plot(test_performance2[:, 0], test_performance2[:, 2], label=algo2 + ' ' + run2 + ' Training')
# plt.plot(test_performance1[:, 0], test_performance1[:, 5], label=algo1 + ' ' + run1 + ' Validation')
# plt.plot(test_performance2[:, 0], test_performance2[:, 5], label=algo2 + ' ' + run2 + ' Validation')
# plt.title('Error ' + title)
# plt.legend(loc='upper right')
# plt.savefig('plots/' + title + ' Error' + '.png')
# plt.show()
#
#
# plt.plot(test_performance2[:, 0], test_performance1[:, 6]/max_pop1, label=algo1 + ' ' + run1 + ' Population')
# plt.plot(test_performance2[:, 0], test_performance2[:, 6]/max_pop2, label=algo2 + ' ' + run2 + ' Population')
# plt.title('Population Size ' + title)
# plt.legend(loc='lower left')
# plt.savefig('plots/' + title + ' Population Size' + '.png')
# plt.show()


print('done')
