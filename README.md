# SILIA - Software Implementation of a Lock-In Amplifier

A software implementation of a multi channel and multi frequency lock-in amplifier to extract periodic features from data

To install, open terminal as admin, navigate to the package directory and run 
~~~ 
python setup.py install
~~~

For a tutorial on how to use the software, look at the jupyter notebook in the 'Example/' directory. In the notebook, we simulate a dataset but the data inputs can be readings from any experimental device or even old video given that they are formatted the correct way.

Run the scripts in the 'paper' folder to replicate our results in figures 2-5, 9. There is a quick script that should run within 30 minutes - 1 hour on a laptop computer and will approximately replicate our figures. The full replication script will take longer, but will more exactly replicate the figures seen in the paper. There is a script to replicate our experimental results in figures 7 and 8 along with our experimental data in the following google drive folder - https://drive.google.com/drive/folders/1jAotoEDoEJWXhmoS94Wsp2U8re8pXLOL?usp=sharing
 
Below, an overview of the lock-in process. The reference fitting step can also be skipped with a toggle. For more information, see 'Manuscript Draft.pdf' which is an early and rudimentary version of the paper we hope to eventually submit. Note that this draft is not our current draft.

![Alt text](images/general_code_diagram.png?raw=true "General Code Summary")


When designing SILIA, we had two principles in mind. The first was that no software can be designed with the foresight to include features for all of its possible use cases without being overwhelmingly complex. And the second was, a user-friendly interface is essential for a software package to be widely accepted. As a result, we focused on writing code that can easily be understood and edited for specific use cases while also ensuring SILIA is easy to use and minimizes data preparation time.

Please ask Amrut Nadgir (amrut.nadgir@gmail.com) or Richard Thurston (rthurston@lbl.gov) if you have any questions. 
