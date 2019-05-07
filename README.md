# Auto Explore

**This library is in alpha stages.  Contributions are welcome**

The goal of this Python library is to create a reliable tool for performing a first-pass exploratory data analysis.  The hope is that ML developers & data analysts will shorten their iteration cycle time by using this tool.  

The earliest stages of a machine learning project require exploratory analysis to uncover raw features and insights that can be exploited in the modeling process.  Exploratory analysis typically follows a somewhat tree-like (and at times recursive) process where task-patterns emerge across projects.  By specifying certain parameters *a priori* about the data in question, a process that adheres to these task-patterns can be designed using open source tools to automate the majority of the "first pass" data analysis work -- freeing up time for deep-dive analyses, modeling, and deployment.

The open source projects that will be relied upon most for this project include:

- [`featexp`](https://github.com/abhayspawar/featexp)
- [`pandas`](https://pandas.pydata.org/)
- [`matplotlib`](https://matplotlib.org/)

While the term "automated" data analysis sounds difficult, the heavy lifting has been done by these library authors.  This project will simply be extending good work that already exists, meaning I will not need to spend considerable time re-inventing the wheel on already established techniques.  

True automation is still a ways out, but this library can be very helpful in exploring a new dataset. 
