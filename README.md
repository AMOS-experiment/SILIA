# SILIA - Software Implementation of a Lock-In Amplifier

A software implementation of a multi channel and multi frequency lock-in amplifier to extract periodic features from data

To install, open terminal as admin, navigate to the package directory and run 
~~~ 
python setup.py install
~~~

For a tutorial on how to use the software, look at the jupyter notebook in the 'Example/' directory. In the notebook, we simulate a dataset but the data inputs can be readings from any device or even old video given that they are formatted the correct way. 
 
Below, an overview of the lock-in process. The reference fitting step can also be skipped with a toggle. For more information, see 'Manuscript Draft.pdf' which is an early and rudimentary version of the paper we hope to eventually submit.

![Alt text](images/general_code_diagram.png?raw=true "General Code Summary")

Please ask Amrut Nadgir (amrut.nadgir@gmail.com) or Richard Thurston (rthurston@lbl.gov) if you have any questions. 
