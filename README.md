# Identification of network users by profiling their behavior - experiments
Thesis abstract
--------
The precise identification of users in the network at different moments in time is a well known
and difficult problem. Identifying users by their actions (and not their IP addresses) allows
administrators to apply policy controls on users, to find intruders that are impersonating le-
gitimate users, and to find anomalous user behaviors that could be due to malware infections.
More importantly, the behavioral analysis of users actions raises important moral questions
about the power to identify users in unknown networks. This thesis explores this question by
trying to identify users by converting the user’s behavior into user’s profiles. These profiles
are time-dependent and they have dozen of features. By using the traffic of known past users
in our dataset, it was possible to create and store their behavioral profiles. The profiles were
created by extracting features from NetFlow data, and therefore no payload was used. The
decision to only use NetFlows made this research much more challenging since there were less
data. After studying the behaviors, we designed a comparison model that it is a similarity
metric between users profiles. The profiles are compared one-to-one and also in sequential
groups. The comparison of groups of profiles is the base for the user to user classifier. These
methods were verified on experiments that used one of the largest labeled datasets currently
available in the area, consisting in more than one month of real traffic from 19 known and
verified normal users. All our tools were published online, including the tools to visualize
and compare users. Results show that we can identify our users with 60% of accuracy and
90% precision. The success of this method mostly depends on how well we can compare two
user profiles. A small improvement can lead to improvement in user detection.

Thesis text
--------
```
Identification_of_network_users_by_profiling_their_behavior.pdf
```
Installation
--------
Clone the repository
```
git clone -b thesis https://github.com/Kobtul/diploma-profiling-experiments.git
```

create the virtual enviroment:
```
python3 -m venv venv/diplomaexperiments
```
Activate the virtual enviroment
```
source venv/diplomaexperiments/bin/activate
```
Go to the project directory
```
cd diplomaexperiments
```
Install required dependencies
```
pip install -r requirements.txt
```
Test the program by running:
```
python mainsimple.py -h
```
