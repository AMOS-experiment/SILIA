# SILIA - Software Implementation of a Lock-In Amplifier

A software implementation of a multi channel and multi frequency lock-in amplifier to extract periodic features from data

To install you can clone the repository and ```cd``` into the SILIA folder. 

~~~
$ git clone https://github.com/AMOS-experiment/SILIA.git
$ cd SILIA
~~~

After cloning the repo, run
~~~ 
$ python setup.py install
~~~


### Dependencies

This software requires Python >= 3.7 as well as working installations of numpy, scipy, tqdm, and pytest. 


## Examples/Tutorial

For a tutorial on how to use the software, look at the jupyter notebook in the 'Example/' directory. In the examples, we simulate a dataset but the data inputs can be readings from any experimental device or even old video given that they are formatted the correct way.

## Testing

To run our unit tests, navigate to the package directory and run
~~~ 
$ pytest
~~~
These unit tests ensure certain functions are working correctly, but are not comprehensive. To confirm that SILIA runs properly, we recommend running the tutorial and examples. 

## Paper

For more information, see 'paper/manuscript.pdf' which is a draft of our paper. Note that this manuscript has not yet been published and is not the final version of the paper, it is only on the GitHub for reference. 

Run the scripts in the 'paper' folder to replicate our results in figures 2-5, 9. There is a quick script that should run within 15 minutes - 1 hour on a laptop computer and will approximately replicate our figures (figures 4 and 5 might not be exact). The full replication script will take longer, but will more exactly replicate the figures seen in the paper. There is a script to replicate our experimental results in figures 7 and 8 along with our experimental data in the following google drive folder - https://drive.google.com/drive/folders/1jAotoEDoEJWXhmoS94Wsp2U8re8pXLOL?usp=sharing
 
Below, an overview of the lock-in process. The reference fitting step can also be skipped with a toggle. 

![Alt text](images/general_code_diagram.png?raw=true "General Code Summary")

## Design
When designing SILIA, we had two principles in mind. The first was that no software can be designed with the foresight to include features for all of its possible use cases without being overwhelmingly complex. And the second was, a user-friendly interface is essential for a software package to be widely accepted. As a result, we focused on writing code that can easily be understood and edited for specific use cases while also ensuring SILIA is easy to use and minimizes data preparation time. Given some software experience and basic knoweldge of lock-in amplifiers, the code in the main SILIA folder should be relatively simple to understand and/or customize. 

Please ask Amrut Nadgir (amrut.nadgir@gmail.com) or Richard Thurston (rthurston@lbl.gov) if you have any questions. 
