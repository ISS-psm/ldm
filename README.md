## Magnetic discontinuities in the solar wind (LDM) ##

####  1. OBJECTIVE ####
  
This project is based on:
  Classifying Interplanetary Discontinuities Using Supervised Machine Learning
  The associated article can be found here: 10.22541/essoar.168056803.33578278/v1
	
 #### 2. INTRODUCTION ####
  
Abrupt changes in the orientation of the interplanetary magnetic field (IMF) are referred
to us as directional discontinuities (DDs), and are contained in the structure of the the solar wind.  DDs are known to trigger geomagnetic storms and magnetospheric substorms, with significant impact on ground-based and spaceborne technologies (e.g., Tsurutani et al., 2011). They can be used, for example, to estimate the solar wind propagation time from an upstream solar wind monitor to a downstream target (e.g., Mailyan et al., 2008; Haaland et al., 2010; Munteanu et al., 2013), kinetic modelling of the solar wind (Artemyev et al., 2019).

 #### 3. DATA ####
  
It was used in-situ magnetic field measurements from ESA’s Cluster mission in 2007 and 2008. Cluster is a constellation of four spacecraft flying in tetrahedral formation around Earth, launched in 2000 to study Earth’s magnetosphere, it is still active and gather important data regarding the complex plasma interactions of the solar wind with the magnetosphere. 
   In matlab frame it was compiled a database of events that contains directional discontinuities: 4216 events in January-April 2007, and 5194 in January-April 2008.
   A preliminary classification algorithm is designed to distinguish between: simple - isolated events, and complex - multiple overlapping events. In 2007, 1806 events are pre-classified as simple, and 2410 as complex; in 2008, 1997 events are simple, and 3197 are complex. This data is later used to train a hybrid CNN-SVM model in order to automatic detect these events. Thus one can find these events organized in the binary_dataset/ directory in the corresponding 2007 and 2008 folder. 
   The supervised machine learning model uses 2007 events as training data to predict the events for 2008 and vice-versa, the predictions for 2007 is done  with the training on 2008.

####   4. FURTHER WORK ####
   
This projects aims to be extended on all the database of the Cluster mission using trained machine learning model to identify magnetic field discontinuities. 

####   5. TEAM: ####
   
This project is maintained and developed by:

Munteanu Costel, costelm@spacescience.ro

Dumitru Daniel, daniel.dumitru@infpr.ro   


